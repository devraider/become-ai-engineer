"""
Week 13 - Project Solution: Semantic Document Search Engine

Complete implementation of a semantic search engine with document processing,
multiple chunking strategies, vector indexing, and intelligent retrieval.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Iterator
from abc import ABC, abstractmethod
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
import hashlib
import json
import math
import re
import time
import uuid

# Check optional dependencies
HAS_CHROMADB = find_spec("chromadb") is not None
HAS_SENTENCE_TRANSFORMERS = find_spec("sentence_transformers") is not None

if HAS_CHROMADB:
    import chromadb

if HAS_SENTENCE_TRANSFORMERS:
    from sentence_transformers import SentenceTransformer


# =============================================================================
# Document Models
# =============================================================================
class DocumentType(Enum):
    """Supported document types."""

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    CODE = "code"


@dataclass
class DocumentSource:
    """Source information for a document."""

    uri: str
    type: DocumentType = DocumentType.TEXT
    metadata: dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        """Extract filename from URI."""
        return Path(self.uri).name if self.uri else ""


@dataclass
class DocumentChunk:
    """A chunk of a document with its embedding."""

    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

    def __len__(self) -> int:
        return len(self.content)


@dataclass
class Document:
    """Complete document with source info and chunks."""

    id: str
    content: str
    source: DocumentSource
    chunks: list[DocumentChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str, source_uri: str = "memory://") -> "Document":
        """Create document from plain text."""
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        return cls(id=doc_id, content=text, source=DocumentSource(uri=source_uri))

    @classmethod
    def from_file(cls, path: str) -> "Document":
        """Create document from file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = file_path.read_text(encoding="utf-8")
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        # Detect document type
        suffix = file_path.suffix.lower()
        doc_type = {
            ".md": DocumentType.MARKDOWN,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".py": DocumentType.CODE,
            ".js": DocumentType.CODE,
            ".ts": DocumentType.CODE,
        }.get(suffix, DocumentType.TEXT)

        return cls(
            id=doc_id,
            content=content,
            source=DocumentSource(uri=path, type=doc_type),
            metadata={"filename": file_path.name, "size": len(content)},
        )


# =============================================================================
# Chunking Strategies
# =============================================================================
class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[tuple[str, int, int]]:
        """
        Split text into chunks.

        Returns list of (chunk_text, start_pos, end_pos) tuples.
        """
        pass


class FixedSizeChunker(ChunkingStrategy):
    """Chunk text into fixed-size pieces with overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            chunk = text[start:end]
            chunks.append((chunk, start, end))

            # Move start position with overlap
            start = end - self._overlap
            if start <= chunks[-1][1]:  # Prevent infinite loop
                start = end

        return chunks


class SentenceChunker(ChunkingStrategy):
    """Chunk text by sentences, respecting max chunk size."""

    def __init__(self, max_chunk_size: int = 500, min_sentences: int = 1):
        self._max_chunk_size = max_chunk_size
        self._min_sentences = min_sentences

    def chunk(self, text: str) -> list[tuple[str, int, int]]:
        """Split text by sentence boundaries."""
        # Simple sentence splitting
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_start = 0
        current_pos = 0

        for sentence, pos in sentences:
            # Start new chunk if adding this sentence exceeds max size
            if current_chunk:
                new_size = sum(len(s) for s, _ in current_chunk) + len(sentence)
                if new_size > self._max_chunk_size:
                    # Save current chunk
                    chunk_text = " ".join(s for s, _ in current_chunk)
                    chunks.append((chunk_text, current_start, pos))

                    current_chunk = []
                    current_start = pos

            current_chunk.append((sentence, pos))
            current_pos = pos + len(sentence)

        # Save final chunk
        if current_chunk:
            chunk_text = " ".join(s for s, _ in current_chunk)
            chunks.append((chunk_text, current_start, current_pos))

        return chunks

    def _split_sentences(self, text: str) -> list[tuple[str, int]]:
        """Split text into sentences with positions."""
        # Simple regex-based sentence splitting
        pattern = r"[.!?]+[\s\n]+"
        sentences = []
        last_end = 0

        for match in re.finditer(pattern, text):
            sentence = text[last_end : match.end()].strip()
            if sentence:
                sentences.append((sentence, last_end))
            last_end = match.end()

        # Handle remaining text
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append((remaining, last_end))

        return sentences


class SemanticChunker(ChunkingStrategy):
    """Chunk text by semantic boundaries (paragraphs, sections)."""

    def __init__(self, max_chunk_size: int = 1000):
        self._max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[tuple[str, int, int]]:
        """Split text by semantic boundaries."""
        # Split by double newlines (paragraphs)
        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk = []
        current_start = 0
        current_size = 0

        for para, pos in paragraphs:
            para_size = len(para)

            # If single paragraph exceeds max size, split it
            if para_size > self._max_chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append((chunk_text, current_start, pos))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph
                for i in range(0, para_size, self._max_chunk_size):
                    sub_chunk = para[i : i + self._max_chunk_size]
                    chunks.append((sub_chunk, pos + i, pos + i + len(sub_chunk)))

                current_start = pos + para_size
                continue

            # Check if adding paragraph exceeds max size
            if current_size + para_size > self._max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append((chunk_text, current_start, pos))
                current_chunk = []
                current_size = 0
                current_start = pos

            current_chunk.append(para)
            current_size += para_size

        # Save final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            end_pos = len(text)
            chunks.append((chunk_text, current_start, end_pos))

        return chunks

    def _split_paragraphs(self, text: str) -> list[tuple[str, int]]:
        """Split text into paragraphs."""
        paragraphs = []
        current_pos = 0

        for match in re.split(r"\n\s*\n", text):
            para = match.strip()
            if para:
                # Find actual position
                pos = text.find(para, current_pos)
                if pos == -1:
                    pos = current_pos
                paragraphs.append((para, pos))
                current_pos = pos + len(para)

        return paragraphs


class MarkdownChunker(ChunkingStrategy):
    """Chunk markdown by headers and sections."""

    def __init__(self, max_chunk_size: int = 1000, min_header_level: int = 2):
        self._max_chunk_size = max_chunk_size
        self._min_header_level = min_header_level

    def chunk(self, text: str) -> list[tuple[str, int, int]]:
        """Split markdown by header boundaries."""
        # Find all headers
        header_pattern = rf"^(#{{{self._min_header_level},}})\s+(.+)$"

        sections = []
        last_pos = 0
        current_header = ""

        for match in re.finditer(header_pattern, text, re.MULTILINE):
            if last_pos > 0 or match.start() > 0:
                # Save previous section
                section = text[last_pos : match.start()].strip()
                if section:
                    sections.append((section, last_pos, match.start(), current_header))

            current_header = match.group(2)
            last_pos = match.start()

        # Handle remaining content
        if last_pos < len(text):
            section = text[last_pos:].strip()
            if section:
                sections.append((section, last_pos, len(text), current_header))

        # If no headers found, use semantic chunking fallback
        if not sections:
            semantic = SemanticChunker(self._max_chunk_size)
            return semantic.chunk(text)

        # Convert sections to chunks, splitting large ones
        chunks = []
        for content, start, end, header in sections:
            if len(content) <= self._max_chunk_size:
                chunks.append((content, start, end))
            else:
                # Split large sections
                for i in range(0, len(content), self._max_chunk_size):
                    sub = content[i : i + self._max_chunk_size]
                    chunks.append((sub, start + i, start + i + len(sub)))

        return chunks


# =============================================================================
# Embedding Providers
# =============================================================================
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate deterministic mock embedding."""
        h = hashlib.md5(text.encode()).hexdigest()
        embedding = []
        for i in range(self._dimension):
            idx = i % 32
            val = int(h[idx], 16) / 15.0
            embedding.append(val - 0.5)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts."""
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerProvider:
    """Sentence Transformers embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dimension = 384

        if HAS_SENTENCE_TRANSFORMERS:
            self._model = SentenceTransformer(model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embed single text."""
        if self._model is None:
            # Mock fallback
            mock = MockEmbeddingProvider(self._dimension)
            return mock.embed(text)

        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts."""
        if self._model is None:
            mock = MockEmbeddingProvider(self._dimension)
            return mock.embed_batch(texts)

        return self._model.encode(texts).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


# =============================================================================
# Document Stores
# =============================================================================
class DocumentStore(ABC):
    """Abstract base class for document storage."""

    @abstractmethod
    def save(self, document: Document) -> None:
        """Save a document."""
        pass

    @abstractmethod
    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        pass

    @abstractmethod
    def list_all(self) -> list[str]:
        """List all document IDs."""
        pass


class InMemoryDocumentStore(DocumentStore):
    """In-memory document store."""

    def __init__(self):
        self._documents: dict[str, Document] = {}

    def save(self, document: Document) -> None:
        """Save document to memory."""
        self._documents[document.id] = document

    def get(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Delete document."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    def list_all(self) -> list[str]:
        """List all document IDs."""
        return list(self._documents.keys())

    def count(self) -> int:
        """Get document count."""
        return len(self._documents)


class FileDocumentStore(DocumentStore):
    """File-based document store."""

    def __init__(self, base_dir: str):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, document: Document) -> None:
        """Save document to file."""
        doc_path = self._base_dir / f"{document.id}.json"

        data = {
            "id": document.id,
            "content": document.content,
            "source": {
                "uri": document.source.uri,
                "type": document.source.type.value,
                "metadata": document.source.metadata,
            },
            "metadata": document.metadata,
            "chunks": [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                for chunk in document.chunks
            ],
        }

        doc_path.write_text(json.dumps(data, indent=2))

    def get(self, doc_id: str) -> Optional[Document]:
        """Load document from file."""
        doc_path = self._base_dir / f"{doc_id}.json"

        if not doc_path.exists():
            return None

        data = json.loads(doc_path.read_text())

        source = DocumentSource(
            uri=data["source"]["uri"],
            type=DocumentType(data["source"]["type"]),
            metadata=data["source"].get("metadata", {}),
        )

        doc = Document(
            id=data["id"],
            content=data["content"],
            source=source,
            metadata=data.get("metadata", {}),
            chunks=[
                DocumentChunk(
                    id=c["id"],
                    content=c["content"],
                    metadata=c.get("metadata", {}),
                    start_char=c.get("start_char", 0),
                    end_char=c.get("end_char", 0),
                )
                for c in data.get("chunks", [])
            ],
        )

        return doc

    def delete(self, doc_id: str) -> bool:
        """Delete document file."""
        doc_path = self._base_dir / f"{doc_id}.json"

        if doc_path.exists():
            doc_path.unlink()
            return True
        return False

    def list_all(self) -> list[str]:
        """List all document IDs."""
        return [p.stem for p in self._base_dir.glob("*.json")]


# =============================================================================
# Vector Index
# =============================================================================
class VectorIndex(ABC):
    """Abstract base class for vector indices."""

    @abstractmethod
    def add(self, chunk_id: str, embedding: list[float], metadata: dict) -> None:
        """Add a vector to the index."""
        pass

    @abstractmethod
    def search(
        self, query_embedding: list[float], k: int, filter: Optional[dict] = None
    ) -> list[dict]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """Delete a vector from the index."""
        pass


class InMemoryVectorIndex(VectorIndex):
    """Simple in-memory vector index."""

    def __init__(self):
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict] = {}

    def add(self, chunk_id: str, embedding: list[float], metadata: dict) -> None:
        """Add vector to index."""
        self._vectors[chunk_id] = embedding
        self._metadata[chunk_id] = metadata

    def search(
        self, query_embedding: list[float], k: int, filter: Optional[dict] = None
    ) -> list[dict]:
        """Search for similar vectors."""
        results = []

        for chunk_id, vec in self._vectors.items():
            meta = self._metadata.get(chunk_id, {})

            # Apply filter
            if filter:
                if not all(meta.get(key) == value for key, value in filter.items()):
                    continue

            score = self._cosine_similarity(query_embedding, vec)
            results.append({"chunk_id": chunk_id, "score": score, "metadata": meta})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def delete(self, chunk_id: str) -> bool:
        """Delete vector from index."""
        if chunk_id in self._vectors:
            del self._vectors[chunk_id]
            del self._metadata[chunk_id]
            return True
        return False

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    def count(self) -> int:
        """Get vector count."""
        return len(self._vectors)


class ChromaVectorIndex(VectorIndex):
    """ChromaDB-based vector index."""

    def __init__(
        self, collection_name: str = "documents", persist_dir: Optional[str] = None
    ):
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None

        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        if not HAS_CHROMADB:
            self._fallback = InMemoryVectorIndex()
            return

        if self._persist_dir:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunk_id: str, embedding: list[float], metadata: dict) -> None:
        """Add vector to ChromaDB."""
        if self._collection is None:
            self._fallback.add(chunk_id, embedding, metadata)
            return

        # ChromaDB doesn't support all metadata types
        clean_metadata = {
            k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))
        }

        self._collection.add(
            ids=[chunk_id], embeddings=[embedding], metadatas=[clean_metadata]
        )

    def search(
        self, query_embedding: list[float], k: int, filter: Optional[dict] = None
    ) -> list[dict]:
        """Search ChromaDB."""
        if self._collection is None:
            return self._fallback.search(query_embedding, k, filter)

        where = filter if filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding], n_results=k, where=where
        )

        output = []
        for i, chunk_id in enumerate(results["ids"][0]):
            output.append(
                {
                    "chunk_id": chunk_id,
                    "score": 1.0
                    - (results["distances"][0][i] if results["distances"] else 0),
                    "metadata": (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    ),
                }
            )

        return output

    def delete(self, chunk_id: str) -> bool:
        """Delete from ChromaDB."""
        if self._collection is None:
            return self._fallback.delete(chunk_id)

        try:
            self._collection.delete(ids=[chunk_id])
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Get vector count."""
        if self._collection is None:
            return self._fallback.count()
        return self._collection.count()


# =============================================================================
# Search Results
# =============================================================================
@dataclass
class SearchHit:
    """A single search hit."""

    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Complete search result with hits and metadata."""

    query: str
    hits: list[SearchHit]
    total_hits: int
    search_time_ms: float
    metadata: dict = field(default_factory=dict)

    def top_k(self, k: int) -> list[SearchHit]:
        """Get top k hits."""
        return self.hits[:k]

    def filter_by_score(self, min_score: float) -> list[SearchHit]:
        """Filter hits by minimum score."""
        return [h for h in self.hits if h.score >= min_score]


# =============================================================================
# Document Processors
# =============================================================================
class DocumentProcessor(ABC):
    """Abstract base class for document processors."""

    @abstractmethod
    def process(self, document: Document) -> Document:
        """Process a document."""
        pass


class TextCleaner(DocumentProcessor):
    """Clean and normalize document text."""

    def __init__(
        self,
        lowercase: bool = False,
        remove_extra_whitespace: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
    ):
        self._lowercase = lowercase
        self._remove_extra_whitespace = remove_extra_whitespace
        self._remove_urls = remove_urls
        self._remove_emails = remove_emails

    def process(self, document: Document) -> Document:
        """Clean document text."""
        text = document.content

        if self._remove_urls:
            text = re.sub(r"https?://\S+", "", text)

        if self._remove_emails:
            text = re.sub(r"\S+@\S+\.\S+", "", text)

        if self._remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

        if self._lowercase:
            text = text.lower()

        document.content = text
        return document


class MetadataExtractor(DocumentProcessor):
    """Extract metadata from document content."""

    def process(self, document: Document) -> Document:
        """Extract metadata from document."""
        content = document.content

        # Extract basic stats
        document.metadata["word_count"] = len(content.split())
        document.metadata["char_count"] = len(content)
        document.metadata["line_count"] = content.count("\n") + 1

        # Extract title (first line or header)
        lines = content.split("\n")
        if lines:
            first_line = lines[0].strip()
            if first_line.startswith("#"):
                document.metadata["title"] = first_line.lstrip("#").strip()
            elif len(first_line) < 100:
                document.metadata["title"] = first_line

        # Extract language hints for code
        if document.source.type == DocumentType.CODE:
            if ".py" in document.source.uri:
                document.metadata["language"] = "python"
            elif ".js" in document.source.uri:
                document.metadata["language"] = "javascript"
            elif ".ts" in document.source.uri:
                document.metadata["language"] = "typescript"

        return document


# =============================================================================
# Ingestion Pipeline
# =============================================================================
class IngestionPipeline:
    """Pipeline for ingesting documents into the search engine."""

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy,
        embedding_provider: EmbeddingProvider,
        document_store: DocumentStore,
        vector_index: VectorIndex,
    ):
        self._chunker = chunking_strategy
        self._embeddings = embedding_provider
        self._store = document_store
        self._index = vector_index
        self._processors: list[DocumentProcessor] = []

    def add_processor(self, processor: DocumentProcessor) -> "IngestionPipeline":
        """Add a document processor."""
        self._processors.append(processor)
        return self

    def ingest(self, document: Document) -> Document:
        """Ingest a single document."""
        # Run processors
        for processor in self._processors:
            document = processor.process(document)

        # Chunk document
        chunks = self._chunker.chunk(document.content)

        document.chunks = []
        texts_to_embed = []

        for i, (chunk_text, start, end) in enumerate(chunks):
            chunk_id = f"{document.id}_chunk_{i}"

            chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_text,
                metadata={
                    "doc_id": document.id,
                    "chunk_index": i,
                    "source_uri": document.source.uri,
                    **document.metadata,
                },
                start_char=start,
                end_char=end,
            )

            document.chunks.append(chunk)
            texts_to_embed.append(chunk_text)

        # Batch embed
        embeddings = self._embeddings.embed_batch(texts_to_embed)

        # Add to index
        for chunk, embedding in zip(document.chunks, embeddings):
            chunk.embedding = embedding
            self._index.add(chunk.id, embedding, chunk.metadata)

        # Save to store
        self._store.save(document)

        return document

    def ingest_batch(self, documents: list[Document]) -> list[Document]:
        """Ingest multiple documents."""
        return [self.ingest(doc) for doc in documents]

    def delete(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        document = self._store.get(doc_id)

        if not document:
            return False

        # Delete chunks from index
        for chunk in document.chunks:
            self._index.delete(chunk.id)

        # Delete from store
        return self._store.delete(doc_id)


# =============================================================================
# Semantic Search Engine
# =============================================================================
class SemanticSearchEngine:
    """Complete semantic search engine."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        document_store: DocumentStore,
    ):
        self._embeddings = embedding_provider
        self._index = vector_index
        self._store = document_store

    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> SearchResult:
        """Perform semantic search."""
        start_time = time.time()

        # Embed query
        query_embedding = self._embeddings.embed(query)

        # Search index
        results = self._index.search(
            query_embedding, k * 2, filter
        )  # Over-fetch for filtering

        # Build hits
        hits = []
        for result in results:
            if result["score"] < min_score:
                continue

            doc_id = result["metadata"].get("doc_id", "")
            content = ""

            # Get chunk content from document
            if doc_id:
                doc = self._store.get(doc_id)
                if doc:
                    for chunk in doc.chunks:
                        if chunk.id == result["chunk_id"]:
                            content = chunk.content
                            break

            hit = SearchHit(
                chunk_id=result["chunk_id"],
                doc_id=doc_id,
                content=content,
                score=result["score"],
                metadata=result["metadata"],
            )
            hits.append(hit)

            if len(hits) >= k:
                break

        search_time = (time.time() - start_time) * 1000

        return SearchResult(
            query=query, hits=hits, total_hits=len(results), search_time_ms=search_time
        )

    def search_with_context(
        self, query: str, k: int = 5, context_window: int = 1
    ) -> SearchResult:
        """Search and include surrounding chunks for context."""
        result = self.search(query, k)

        # Expand hits with context
        expanded_hits = []
        seen_chunks = set()

        for hit in result.hits:
            doc = self._store.get(hit.doc_id)
            if not doc:
                expanded_hits.append(hit)
                continue

            # Find chunk index
            chunk_idx = None
            for i, chunk in enumerate(doc.chunks):
                if chunk.id == hit.chunk_id:
                    chunk_idx = i
                    break

            if chunk_idx is None:
                expanded_hits.append(hit)
                continue

            # Get surrounding chunks
            start_idx = max(0, chunk_idx - context_window)
            end_idx = min(len(doc.chunks), chunk_idx + context_window + 1)

            combined_content = []
            for i in range(start_idx, end_idx):
                chunk = doc.chunks[i]
                if chunk.id not in seen_chunks:
                    combined_content.append(chunk.content)
                    seen_chunks.add(chunk.id)

            # Update hit with expanded content
            hit.content = "\n\n".join(combined_content)
            expanded_hits.append(hit)

        result.hits = expanded_hits
        return result

    def find_similar_documents(self, doc_id: str, k: int = 5) -> SearchResult:
        """Find documents similar to a given document."""
        doc = self._store.get(doc_id)
        if not doc:
            return SearchResult(
                query=f"similar:{doc_id}", hits=[], total_hits=0, search_time_ms=0
            )

        # Use first chunk as query
        if doc.chunks and doc.chunks[0].embedding:
            query_embedding = doc.chunks[0].embedding
        else:
            query_embedding = self._embeddings.embed(doc.content[:1000])

        start_time = time.time()
        results = self._index.search(query_embedding, k + len(doc.chunks))

        # Filter out same document
        hits = []
        for result in results:
            if result["metadata"].get("doc_id") == doc_id:
                continue

            hit = SearchHit(
                chunk_id=result["chunk_id"],
                doc_id=result["metadata"].get("doc_id", ""),
                content="",
                score=result["score"],
                metadata=result["metadata"],
            )
            hits.append(hit)

            if len(hits) >= k:
                break

        search_time = (time.time() - start_time) * 1000

        return SearchResult(
            query=f"similar:{doc_id}",
            hits=hits,
            total_hits=len(hits),
            search_time_ms=search_time,
        )


# =============================================================================
# Search Engine Factory
# =============================================================================
@dataclass
class SearchEngineConfig:
    """Configuration for the search engine."""

    embedding_model: str = "all-MiniLM-L6-v2"
    chunking_strategy: str = "sentence"
    chunk_size: int = 500
    chunk_overlap: int = 50
    use_chromadb: bool = False
    chromadb_persist_dir: Optional[str] = None
    document_store_dir: Optional[str] = None


class SearchEngineFactory:
    """Factory for creating search engine components."""

    @staticmethod
    def create_embedding_provider(config: SearchEngineConfig) -> EmbeddingProvider:
        """Create embedding provider."""
        return SentenceTransformerProvider(config.embedding_model)

    @staticmethod
    def create_chunking_strategy(config: SearchEngineConfig) -> ChunkingStrategy:
        """Create chunking strategy."""
        strategies = {
            "fixed": lambda: FixedSizeChunker(config.chunk_size, config.chunk_overlap),
            "sentence": lambda: SentenceChunker(config.chunk_size),
            "semantic": lambda: SemanticChunker(config.chunk_size),
            "markdown": lambda: MarkdownChunker(config.chunk_size),
        }

        factory = strategies.get(config.chunking_strategy, strategies["sentence"])
        return factory()

    @staticmethod
    def create_document_store(config: SearchEngineConfig) -> DocumentStore:
        """Create document store."""
        if config.document_store_dir:
            return FileDocumentStore(config.document_store_dir)
        return InMemoryDocumentStore()

    @staticmethod
    def create_vector_index(config: SearchEngineConfig) -> VectorIndex:
        """Create vector index."""
        if config.use_chromadb:
            return ChromaVectorIndex(
                collection_name="documents", persist_dir=config.chromadb_persist_dir
            )
        return InMemoryVectorIndex()

    @classmethod
    def create_pipeline(cls, config: SearchEngineConfig) -> IngestionPipeline:
        """Create complete ingestion pipeline."""
        return IngestionPipeline(
            chunking_strategy=cls.create_chunking_strategy(config),
            embedding_provider=cls.create_embedding_provider(config),
            document_store=cls.create_document_store(config),
            vector_index=cls.create_vector_index(config),
        )

    @classmethod
    def create_search_engine(cls, config: SearchEngineConfig) -> SemanticSearchEngine:
        """Create complete search engine."""
        return SemanticSearchEngine(
            embedding_provider=cls.create_embedding_provider(config),
            vector_index=cls.create_vector_index(config),
            document_store=cls.create_document_store(config),
        )


# =============================================================================
# Batch Ingestor
# =============================================================================
class BatchIngestor:
    """Batch document ingestion with progress tracking."""

    def __init__(self, pipeline: IngestionPipeline, batch_size: int = 10):
        self._pipeline = pipeline
        self._batch_size = batch_size

    def ingest_directory(
        self, directory: str, pattern: str = "*.txt", recursive: bool = True
    ) -> dict:
        """Ingest all matching files in a directory."""
        path = Path(directory)

        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        results = {"total": len(files), "succeeded": 0, "failed": 0, "documents": []}

        for file_path in files:
            try:
                doc = Document.from_file(str(file_path))
                self._pipeline.ingest(doc)
                results["succeeded"] += 1
                results["documents"].append(doc.id)
            except Exception as e:
                results["failed"] += 1

        return results

    def ingest_texts(self, texts: list[tuple[str, str, dict]]) -> dict:
        """
        Ingest multiple texts.

        Args:
            texts: List of (text, source_uri, metadata) tuples
        """
        results = {"total": len(texts), "succeeded": 0, "failed": 0, "documents": []}

        for text, source_uri, metadata in texts:
            try:
                doc = Document.from_text(text, source_uri)
                doc.metadata.update(metadata)
                self._pipeline.ingest(doc)
                results["succeeded"] += 1
                results["documents"].append(doc.id)
            except Exception:
                results["failed"] += 1

        return results


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Configure search engine
    config = SearchEngineConfig(
        embedding_model="all-MiniLM-L6-v2", chunking_strategy="sentence", chunk_size=500
    )

    # Create components
    embeddings = SearchEngineFactory.create_embedding_provider(config)
    chunker = SearchEngineFactory.create_chunking_strategy(config)
    store = SearchEngineFactory.create_document_store(config)
    index = SearchEngineFactory.create_vector_index(config)

    # Create pipeline
    pipeline = IngestionPipeline(chunker, embeddings, store, index)
    pipeline.add_processor(TextCleaner())
    pipeline.add_processor(MetadataExtractor())

    # Create search engine
    search = SemanticSearchEngine(embeddings, index, store)

    # Sample documents
    docs = [
        """
        Python is a high-level programming language known for its clear syntax
        and readability. It supports multiple programming paradigms including
        procedural, object-oriented, and functional programming.
        """,
        """
        Machine learning is a subset of artificial intelligence that enables
        computers to learn from data without being explicitly programmed.
        It uses statistical techniques to find patterns in data.
        """,
        """
        Vector databases are specialized databases designed to store and
        query high-dimensional vectors. They are essential for semantic
        search and recommendation systems.
        """,
    ]

    # Ingest documents
    for i, text in enumerate(docs):
        doc = Document.from_text(text.strip(), f"memory://doc_{i}")
        pipeline.ingest(doc)

    # Search
    result = search.search("programming language", k=2)

    print("Search Results:")
    print(f"Query: {result.query}")
    print(f"Time: {result.search_time_ms:.2f}ms")
    print(f"Hits: {result.total_hits}")

    for hit in result.hits:
        print(f"\n  Score: {hit.score:.4f}")
        print(f"  Content: {hit.content[:100]}...")
