"""
Week 8 Project: Document Q&A System - SOLUTIONS
===============================================
"""

import os
import json
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re
import hashlib

try:
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Document:
    """Represents a document with metadata."""

    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.source}:{self.content[:100]}".encode()
            ).hexdigest()[:12]


@dataclass
class Chunk:
    """A chunk of a document."""

    content: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.document_id}_chunk_{self.chunk_index}"


@dataclass
class SearchResult:
    """A search result with relevance score."""

    chunk: Chunk
    score: float


@dataclass
class QAResponse:
    """A Q&A response with sources."""

    question: str
    answer: str
    sources: List[SearchResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# =============================================================================
# PART 1: Document Loader
# =============================================================================


class DocumentLoader:
    """Load documents from various sources."""

    def load_text(self, file_path: str) -> Document:
        """Load a text file."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        metadata = {
            "filename": path.name,
            "file_size": path.stat().st_size,
            "file_type": path.suffix,
        }

        return Document(content=content, source=str(path), metadata=metadata)

    def load_directory(
        self, directory_path: str, extensions: List[str] = None
    ) -> List[Document]:
        """Load all documents from a directory."""
        if extensions is None:
            extensions = [".txt", ".md"]

        path = Path(directory_path)
        documents = []

        for ext in extensions:
            for file_path in path.glob(f"**/*{ext}"):
                if file_path.is_file():
                    doc = self.load_text(str(file_path))
                    documents.append(doc)

        return documents

    def load_from_string(self, content: str, source: str = "inline") -> Document:
        """Create a Document from a string."""
        return Document(
            content=content,
            source=source,
            metadata={"type": "inline", "length": len(content)},
        )


# =============================================================================
# PART 2: Text Chunker
# =============================================================================


class TextChunker:
    """Split documents into chunks for embedding."""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_size: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = ["\n\n", "\n", ". ", " "]

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document into chunks."""
        text = document.content

        if len(text) <= self.chunk_size:
            if len(text) >= self.min_chunk_size:
                return [
                    Chunk(
                        content=text,
                        document_id=document.id,
                        chunk_index=0,
                        metadata={"source": document.source},
                    )
                ]
            return []

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                # Find best break point
                chunk_text = text[start:end]
                best_break = len(chunk_text)

                for sep in self.separators:
                    idx = chunk_text.rfind(sep)
                    if idx > self.chunk_size * 0.3:
                        best_break = idx + len(sep)
                        break

                end = start + best_break

            chunk_content = text[start:end].strip()

            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        document_id=document.id,
                        chunk_index=chunk_idx,
                        metadata={
                            "source": document.source,
                            "start": start,
                            "end": end,
                        },
                    )
                )
                chunk_idx += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks


# =============================================================================
# PART 3: Embedding Manager
# =============================================================================


class EmbeddingManager:
    """Manage text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        if self.model:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        else:
            # Simple fallback: hash-based embeddings
            dim = 128
            embeddings = []
            for text in texts:
                hash_val = hashlib.sha256(text.encode()).digest()
                emb = [float(b) / 255 - 0.5 for b in hash_val[:dim]]
                # Normalize
                mag = math.sqrt(sum(x**2 for x in emb))
                if mag > 0:
                    emb = [x / mag for x in emb]
                embeddings.append(emb)
            return embeddings

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        return self.embed([text])[0]


# =============================================================================
# PART 4: Vector Store
# =============================================================================


class VectorStore:
    """Store and retrieve document chunks."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # In-memory storage
        self._chunks: List[Chunk] = []
        self._embeddings: List[List[float]] = []

        # Try ChromaDB
        self._chroma_client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB if available."""
        if CHROMADB_AVAILABLE:
            try:
                if self.persist_directory:
                    self._chroma_client = chromadb.PersistentClient(
                        path=self.persist_directory
                    )
                else:
                    self._chroma_client = chromadb.Client()

                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self._chroma_client = None

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks with their embeddings."""
        if self._collection:
            self._collection.add(
                documents=[c.content for c in chunks],
                embeddings=embeddings,
                ids=[c.id for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
        else:
            self._chunks.extend(chunks)
            self._embeddings.extend(embeddings)

    def search(
        self,
        query_embedding: List[float],
        k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar chunks."""
        if self._collection:
            results = self._collection.query(
                query_embeddings=[query_embedding], n_results=k, where=filters
            )

            search_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, dist, meta, id_) in enumerate(
                    zip(
                        results["documents"][0],
                        (
                            results["distances"][0]
                            if results["distances"]
                            else [0] * len(results["documents"][0])
                        ),
                        (
                            results["metadatas"][0]
                            if results["metadatas"]
                            else [{}] * len(results["documents"][0])
                        ),
                        (
                            results["ids"][0]
                            if results["ids"]
                            else ["unknown"] * len(results["documents"][0])
                        ),
                    )
                ):
                    chunk = Chunk(
                        content=doc,
                        document_id=(
                            id_.split("_chunk_")[0] if "_chunk_" in id_ else id_
                        ),
                        chunk_index=i,
                        metadata=meta,
                    )
                    score = 1 - dist  # Convert distance to similarity
                    search_results.append(SearchResult(chunk=chunk, score=score))

            return search_results
        else:
            # In-memory search
            results = []
            for chunk, emb in zip(self._chunks, self._embeddings):
                # Apply filters
                if filters:
                    if not all(chunk.metadata.get(k) == v for k, v in filters.items()):
                        continue

                score = cosine_similarity(query_embedding, emb)
                results.append(SearchResult(chunk=chunk, score=score))

            results.sort(key=lambda r: r.score, reverse=True)
            return results[:k]

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        if self._collection:
            # ChromaDB delete with where filter
            try:
                self._collection.delete(where={"source": source})
                return 1  # Can't know exact count
            except Exception:
                return 0
        else:
            original = len(self._chunks)
            indices = [
                i
                for i, c in enumerate(self._chunks)
                if c.metadata.get("source") == source
            ]
            for i in reversed(indices):
                del self._chunks[i]
                del self._embeddings[i]
            return original - len(self._chunks)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self._collection:
            count = self._collection.count()
            return {"total_chunks": count, "backend": "chromadb"}
        else:
            sources = set(c.metadata.get("source", "unknown") for c in self._chunks)
            return {
                "total_chunks": len(self._chunks),
                "unique_sources": len(sources),
                "backend": "in-memory",
            }


# =============================================================================
# PART 5: Response Generator
# =============================================================================


class ResponseGenerator:
    """Generate responses using retrieved context."""

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the LLM."""
        if GENAI_AVAILABLE and load_dotenv:
            try:
                load_dotenv()
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel(self.model_name)
            except Exception:
                self.model = None

    def generate(
        self,
        question: str,
        context_chunks: List[SearchResult],
        include_sources: bool = True,
    ) -> str:
        """Generate an answer using retrieved context."""
        context = self._format_context(context_chunks)
        prompt = self._create_prompt(question, context, include_sources)

        if self.model:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Error generating response: {e}"
        else:
            # Fallback: return context summary
            if not context_chunks:
                return "No relevant information found."

            answer = f"Based on the available context:\n\n"
            for i, result in enumerate(context_chunks[:3], 1):
                answer += f"{i}. {result.chunk.content[:200]}...\n\n"
            return answer

    def _format_context(self, chunks: List[SearchResult]) -> str:
        """Format chunks into context string."""
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, result in enumerate(chunks, 1):
            source = result.chunk.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {source}]\n{result.chunk.content}")

        return "\n\n---\n\n".join(parts)

    def _create_prompt(self, question: str, context: str, include_sources: bool) -> str:
        """Create the prompt for the LLM."""
        prompt = f"""Use the following context to answer the question accurately.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

"""
        if include_sources:
            prompt += "Include source references in your answer where relevant.\n\n"

        prompt += "Answer:"
        return prompt


# =============================================================================
# PART 6: RAG Pipeline
# =============================================================================


class RAGPipeline:
    """Complete RAG pipeline for document Q&A."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "rag_docs",
        persist_directory: Optional[str] = None,
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingManager(embedding_model)
        self.vector_store = VectorStore(collection_name, persist_directory)
        self.generator = ResponseGenerator()

    def ingest_text(self, text: str, source: str = "inline") -> int:
        """Ingest text content."""
        doc = self.loader.load_from_string(text, source)
        chunks = self.chunker.chunk_document(doc)

        if not chunks:
            return 0

        embeddings = self.embedder.embed([c.content for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)

        return len(chunks)

    def ingest_file(self, file_path: str) -> int:
        """Ingest a file."""
        doc = self.loader.load_text(file_path)
        chunks = self.chunker.chunk_document(doc)

        if not chunks:
            return 0

        embeddings = self.embedder.embed([c.content for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)

        return len(chunks)

    def ingest_directory(
        self, directory_path: str, extensions: List[str] = None
    ) -> int:
        """Ingest all files from a directory."""
        docs = self.loader.load_directory(directory_path, extensions)

        total_chunks = 0
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            if chunks:
                embeddings = self.embedder.embed([c.content for c in chunks])
                self.vector_store.add_chunks(chunks, embeddings)
                total_chunks += len(chunks)

        return total_chunks

    def query(
        self, question: str, k: int = 3, include_sources: bool = True
    ) -> QAResponse:
        """Query the RAG system."""
        # Embed question
        query_embedding = self.embedder.embed_single(question)

        # Search
        results = self.vector_store.search(query_embedding, k)

        # Generate response
        answer = self.generator.generate(question, results, include_sources)

        return QAResponse(
            question=question,
            answer=answer,
            sources=results,
            metadata={"k": k, "num_sources": len(results)},
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        store_stats = self.vector_store.get_stats()
        return {
            **store_stats,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
        }


# =============================================================================
# PART 7: Query Interface
# =============================================================================


class DocumentQA:
    """High-level interface for document Q&A."""

    def __init__(self, **kwargs):
        self.pipeline = RAGPipeline(**kwargs)
        self.history: List[QAResponse] = []

    def add_document(self, file_path: str) -> int:
        """Add a document to the knowledge base."""
        return self.pipeline.ingest_file(file_path)

    def add_documents(self, directory_path: str, extensions: List[str] = None) -> int:
        """Add all documents from a directory."""
        return self.pipeline.ingest_directory(directory_path, extensions)

    def add_text(self, text: str, source: str = "inline") -> int:
        """Add text content to the knowledge base."""
        return self.pipeline.ingest_text(text, source)

    def ask(
        self, question: str, k: int = 3, include_sources: bool = True
    ) -> QAResponse:
        """Ask a question about the documents."""
        response = self.pipeline.query(question, k, include_sources)
        self.history.append(response)
        return response

    def get_history(self) -> List[QAResponse]:
        """Get conversation history."""
        return self.history

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self.pipeline.get_stats()
        stats["questions_asked"] = len(self.history)
        return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Week 8 Project: Document Q&A System - SOLUTIONS")
    print("=" * 60)

    qa = DocumentQA(chunk_size=200, chunk_overlap=20)

    # Add sample content
    sample_docs = [
        """
        Retrieval Augmented Generation (RAG) is a technique that combines 
        information retrieval with text generation. It helps LLMs provide 
        more accurate and up-to-date responses by grounding them in 
        retrieved documents.
        """,
        """
        Vector databases store embeddings, which are numerical representations 
        of text. They enable semantic search, where queries find documents 
        based on meaning rather than just keyword matching.
        """,
        """
        Chunking is the process of splitting documents into smaller pieces.
        Good chunking strategies preserve context while keeping chunks 
        small enough for effective retrieval.
        """,
    ]

    print("\nIngesting documents...")
    for i, doc in enumerate(sample_docs):
        chunks = qa.add_text(doc.strip(), f"doc_{i+1}")
        print(f"  Document {i+1}: {chunks} chunks")

    # Ask questions
    print("\nAsking questions...")
    questions = ["What is RAG?", "How do vector databases work?"]

    for q in questions:
        print(f"\nQ: {q}")
        response = qa.ask(q)
        print(f"A: {response.answer[:300]}...")

    # Print stats
    print("\n\nStats:", qa.get_stats())
