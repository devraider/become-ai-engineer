"""
Week 13 - Solution 2: ChromaDB Integration

Complete implementations for ChromaDB document search system.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from abc import ABC, abstractmethod
from importlib.util import find_spec
import hashlib
import json
import re

# Check optional dependencies
HAS_CHROMADB = find_spec("chromadb") is not None

if HAS_CHROMADB:
    import chromadb
    from chromadb.utils import embedding_functions as chroma_embedding_functions


# =============================================================================
# Part 1: Document Model
# =============================================================================
@dataclass
class Document:
    """Represents a document to be stored in ChromaDB."""

    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    @classmethod
    def from_text(cls, text: str, metadata: Optional[dict] = None) -> "Document":
        """Create a document from text, generating an ID from content hash."""
        doc_id = hashlib.md5(text.encode()).hexdigest()[:16]
        return cls(id=doc_id, content=text, metadata=metadata or {})

    def to_chroma_format(self) -> dict:
        """Convert to format expected by ChromaDB add()."""
        result = {"id": self.id, "document": self.content, "metadata": self.metadata}
        if self.embedding is not None:
            result["embedding"] = self.embedding
        return result

    def validate(self) -> bool:
        """Validate that document is properly formatted."""
        # Check ID
        if not self.id or not self.id.strip():
            return False

        # Check content
        if not self.content or not self.content.strip():
            return False

        # Check metadata values
        allowed_types = (str, int, float, bool)
        for key, value in self.metadata.items():
            if not isinstance(value, allowed_types):
                return False

        return True


# =============================================================================
# Part 2: Collection Configuration
# =============================================================================
@dataclass
class CollectionConfig:
    """Configuration for a ChromaDB collection."""

    name: str
    embedding_function: Optional[str] = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"
    distance_metric: str = "cosine"
    hnsw_space: str = "cosine"
    hnsw_construction_ef: int = 100
    hnsw_m: int = 16

    def to_metadata(self) -> dict:
        """Convert to ChromaDB collection metadata format."""
        return {
            "hnsw:space": self.hnsw_space,
            "hnsw:construction_ef": self.hnsw_construction_ef,
            "hnsw:M": self.hnsw_m,
        }

    def get_embedding_function(self):
        """Get the ChromaDB embedding function based on config."""
        if not HAS_CHROMADB:
            return None

        if self.embedding_function == "sentence-transformers":
            return chroma_embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        elif self.embedding_function == "openai":
            return chroma_embedding_functions.OpenAIEmbeddingFunction()
        else:
            return chroma_embedding_functions.DefaultEmbeddingFunction()


# =============================================================================
# Part 3: Query Builder
# =============================================================================
@dataclass
class QueryFilter:
    """Represents a filter condition for ChromaDB queries."""

    field: str
    operator: str
    value: Any

    def to_chroma_format(self) -> dict:
        """Convert to ChromaDB where clause format."""
        return {self.field: {self.operator: self.value}}


class QueryBuilder:
    """Builder for constructing ChromaDB queries."""

    def __init__(self):
        self._query_texts: Optional[list[str]] = None
        self._query_embeddings: Optional[list[list[float]]] = None
        self._n_results: int = 10
        self._where: Optional[dict] = None
        self._where_document: Optional[dict] = None
        self._include: list[str] = ["documents", "metadatas", "distances"]

    def with_text(self, text: str) -> "QueryBuilder":
        """Set query text."""
        self._query_texts = [text]
        return self

    def with_texts(self, texts: list[str]) -> "QueryBuilder":
        """Set multiple query texts."""
        self._query_texts = texts
        return self

    def with_embedding(self, embedding: list[float]) -> "QueryBuilder":
        """Set query embedding directly."""
        self._query_embeddings = [embedding]
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """Set number of results."""
        self._n_results = n
        return self

    def where(self, filter: QueryFilter) -> "QueryBuilder":
        """Add a metadata filter."""
        self._where = filter.to_chroma_format()
        return self

    def where_and(self, filters: list[QueryFilter]) -> "QueryBuilder":
        """Add multiple filters with AND logic."""
        self._where = {"$and": [f.to_chroma_format() for f in filters]}
        return self

    def where_or(self, filters: list[QueryFilter]) -> "QueryBuilder":
        """Add multiple filters with OR logic."""
        self._where = {"$or": [f.to_chroma_format() for f in filters]}
        return self

    def where_document_contains(self, text: str) -> "QueryBuilder":
        """Filter by document content."""
        self._where_document = {"$contains": text}
        return self

    def include_distances(self) -> "QueryBuilder":
        """Include distances in results."""
        if "distances" not in self._include:
            self._include.append("distances")
        return self

    def include_embeddings(self) -> "QueryBuilder":
        """Include embeddings in results."""
        if "embeddings" not in self._include:
            self._include.append("embeddings")
        return self

    def build(self) -> dict:
        """Build the query dictionary for ChromaDB."""
        query = {"n_results": self._n_results, "include": self._include}

        if self._query_texts:
            query["query_texts"] = self._query_texts

        if self._query_embeddings:
            query["query_embeddings"] = self._query_embeddings

        if self._where:
            query["where"] = self._where

        if self._where_document:
            query["where_document"] = self._where_document

        return query


# =============================================================================
# Part 4: Search Result
# =============================================================================
@dataclass
class ChromaSearchResult:
    """Represents a single search result from ChromaDB."""

    id: str
    document: str
    metadata: dict
    distance: float
    embedding: Optional[list[float]] = None

    @property
    def score(self) -> float:
        """Convert distance to similarity score (0-1)."""
        return 1.0 - self.distance

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "document": self.document,
            "metadata": self.metadata,
            "distance": self.distance,
            "score": self.score,
        }
        if self.embedding:
            result["embedding"] = self.embedding
        return result


class SearchResults:
    """Container for ChromaDB search results."""

    def __init__(self, raw_results: dict):
        """Parse raw ChromaDB results into SearchResult objects."""
        self._results: list[ChromaSearchResult] = []
        self._parse_results(raw_results)

    def _parse_results(self, raw: dict) -> None:
        """Parse raw results into ChromaSearchResult objects."""
        if not raw.get("ids") or not raw["ids"][0]:
            return

        ids = raw["ids"][0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        embeddings = (
            raw.get("embeddings", [[]])[0]
            if raw.get("embeddings")
            else [None] * len(ids)
        )

        for i in range(len(ids)):
            self._results.append(
                ChromaSearchResult(
                    id=ids[i],
                    document=documents[i] if i < len(documents) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    distance=distances[i] if i < len(distances) else 0.0,
                    embedding=(
                        embeddings[i] if embeddings and i < len(embeddings) else None
                    ),
                )
            )

    def __iter__(self):
        """Iterate over results."""
        return iter(self._results)

    def __len__(self) -> int:
        """Get number of results."""
        return len(self._results)

    def __getitem__(self, index: int) -> ChromaSearchResult:
        """Get result by index."""
        return self._results[index]

    def filter_by_score(self, min_score: float) -> "SearchResults":
        """Filter results by minimum score."""
        filtered = SearchResults({"ids": [[]]})
        filtered._results = [r for r in self._results if r.score >= min_score]
        return filtered

    def to_documents(self) -> list[Document]:
        """Convert results back to Document objects."""
        return [
            Document(
                id=r.id, content=r.document, metadata=r.metadata, embedding=r.embedding
            )
            for r in self._results
        ]


# =============================================================================
# Part 5: ChromaDB Client Wrapper
# =============================================================================
class ChromaDBClient:
    """Wrapper around ChromaDB client."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """Initialize ChromaDB client."""
        self._client = None
        self._collections: dict = {}

        if HAS_CHROMADB:
            if host and port:
                self._client = chromadb.HttpClient(host=host, port=port)
            elif persist_directory:
                settings = chromadb.Settings(
                    chroma_db_impl="duckdb+parquet", persist_directory=persist_directory
                )
                self._client = chromadb.Client(settings)
            else:
                self._client = chromadb.Client()

    def create_collection(
        self, config: CollectionConfig, overwrite: bool = False
    ) -> None:
        """Create a new collection."""
        if self._client is None:
            # Mock implementation
            self._collections[config.name] = {"config": config, "documents": {}}
            return

        if overwrite:
            try:
                self._client.delete_collection(config.name)
            except Exception:
                pass

        embedding_fn = config.get_embedding_function()

        self._client.get_or_create_collection(
            name=config.name,
            metadata=config.to_metadata(),
            embedding_function=embedding_fn,
        )

    def get_collection(self, name: str):
        """Get a collection by name."""
        if self._client is None:
            return self._collections.get(name)
        return self._client.get_collection(name)

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if self._client is None:
            if name in self._collections:
                del self._collections[name]
                return True
            return False

        try:
            self._client.delete_collection(name)
            return True
        except Exception:
            return False

    def list_collections(self) -> list[str]:
        """List all collection names."""
        if self._client is None:
            return list(self._collections.keys())

        return [c.name for c in self._client.list_collections()]

    def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        return name in self.list_collections()


# =============================================================================
# Part 6: Document Collection Manager
# =============================================================================
class DocumentCollection:
    """High-level interface for managing documents in a ChromaDB collection."""

    def __init__(self, client: ChromaDBClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name
        self._collection = client.get_collection(collection_name)

    def add(self, document: Document) -> str:
        """Add a single document."""
        if self._collection is None:
            return document.id

        self._collection.add(
            ids=[document.id],
            documents=[document.content],
            metadatas=[document.metadata],
            embeddings=[document.embedding] if document.embedding else None,
        )
        return document.id

    def add_batch(self, documents: list[Document]) -> list[str]:
        """Add multiple documents efficiently."""
        if not documents:
            return []

        if self._collection is None:
            return [d.id for d in documents]

        embeddings = None
        if documents[0].embedding is not None:
            embeddings = [d.embedding for d in documents]

        self._collection.add(
            ids=[d.id for d in documents],
            documents=[d.content for d in documents],
            metadatas=[d.metadata for d in documents],
            embeddings=embeddings,
        )

        return [d.id for d in documents]

    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        if self._collection is None:
            return None

        result = self._collection.get(
            ids=[doc_id], include=["documents", "metadatas", "embeddings"]
        )

        if not result["ids"]:
            return None

        return Document(
            id=result["ids"][0],
            content=result["documents"][0] if result.get("documents") else "",
            metadata=result["metadatas"][0] if result.get("metadatas") else {},
            embedding=result["embeddings"][0] if result.get("embeddings") else None,
        )

    def get_batch(self, doc_ids: list[str]) -> list[Document]:
        """Get multiple documents by IDs."""
        if self._collection is None:
            return []

        result = self._collection.get(
            ids=doc_ids, include=["documents", "metadatas", "embeddings"]
        )

        documents = []
        for i, doc_id in enumerate(result["ids"]):
            documents.append(
                Document(
                    id=doc_id,
                    content=result["documents"][i] if result.get("documents") else "",
                    metadata=result["metadatas"][i] if result.get("metadatas") else {},
                    embedding=(
                        result["embeddings"][i] if result.get("embeddings") else None
                    ),
                )
            )

        return documents

    def update(self, document: Document) -> bool:
        """Update an existing document."""
        if self._collection is None:
            return False

        try:
            self._collection.update(
                ids=[document.id],
                documents=[document.content],
                metadatas=[document.metadata],
                embeddings=[document.embedding] if document.embedding else None,
            )
            return True
        except Exception:
            return False

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if self._collection is None:
            return False

        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False

    def delete_batch(self, doc_ids: list[str]) -> int:
        """Delete multiple documents."""
        if self._collection is None:
            return 0

        try:
            self._collection.delete(ids=doc_ids)
            return len(doc_ids)
        except Exception:
            return 0

    def search(self, query: QueryBuilder) -> SearchResults:
        """Search the collection using a QueryBuilder."""
        if self._collection is None:
            return SearchResults({"ids": [[]]})

        query_dict = query.build()
        result = self._collection.query(**query_dict)
        return SearchResults(result)

    def search_text(
        self, text: str, n_results: int = 10, filter: Optional[QueryFilter] = None
    ) -> SearchResults:
        """Simple text search interface."""
        builder = QueryBuilder().with_text(text).limit(n_results)

        if filter:
            builder = builder.where(filter)

        return self.search(builder)

    def count(self) -> int:
        """Get the number of documents in the collection."""
        if self._collection is None:
            return 0
        return self._collection.count()


# =============================================================================
# Part 7: Document Preprocessor
# =============================================================================
class DocumentPreprocessor:
    """Preprocessor for documents before adding to ChromaDB."""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: Document) -> list[Document]:
        """Split a document into chunks with overlap."""
        content = document.content

        if len(content) <= self.chunk_size:
            return [document]

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]

            # Don't create tiny final chunks
            if len(chunk_content) < self.min_chunk_size and chunks:
                # Append to previous chunk
                chunks[-1] = Document(
                    id=f"{document.id}_chunk_{chunk_idx-1}",
                    content=chunks[-1].content + chunk_content,
                    metadata={
                        **document.metadata,
                        "parent_id": document.id,
                        "chunk_index": chunk_idx - 1,
                    },
                )
            else:
                chunks.append(
                    Document(
                        id=f"{document.id}_chunk_{chunk_idx}",
                        content=chunk_content,
                        metadata={
                            **document.metadata,
                            "parent_id": document.id,
                            "chunk_index": chunk_idx,
                        },
                    )
                )
                chunk_idx += 1

            start = end - self.chunk_overlap
            if start >= len(content) - self.chunk_overlap:
                break

        return chunks

    def clean_text(self, text: str) -> str:
        """Clean text content."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def extract_metadata(self, document: Document) -> dict:
        """Extract additional metadata from document."""
        content = document.content

        return {
            "word_count": len(content.split()),
            "char_count": len(content),
            "has_urls": bool(re.search(r"https?://", content)),
            "has_code_blocks": bool(re.search(r"```", content)),
        }

    def process(self, document: Document) -> list[Document]:
        """Full preprocessing pipeline."""
        # Clean text
        cleaned_content = self.clean_text(document.content)

        # Extract metadata
        extracted_metadata = self.extract_metadata(document)

        # Create processed document
        processed = Document(
            id=document.id,
            content=cleaned_content,
            metadata={**document.metadata, **extracted_metadata},
            embedding=document.embedding,
        )

        # Chunk if needed
        return self.chunk_document(processed)


# =============================================================================
# Part 8: Metadata Filter Builder
# =============================================================================
class MetadataFilterBuilder:
    """Builder for complex metadata filters."""

    def __init__(self):
        self._filters: list[dict] = []
        self._logic: str = "and"

    def equals(self, field: str, value: Any) -> "MetadataFilterBuilder":
        """Add equality filter."""
        self._filters.append({field: {"$eq": value}})
        return self

    def not_equals(self, field: str, value: Any) -> "MetadataFilterBuilder":
        """Add inequality filter."""
        self._filters.append({field: {"$ne": value}})
        return self

    def greater_than(self, field: str, value: Any) -> "MetadataFilterBuilder":
        """Add greater than filter."""
        self._filters.append({field: {"$gt": value}})
        return self

    def less_than(self, field: str, value: Any) -> "MetadataFilterBuilder":
        """Add less than filter."""
        self._filters.append({field: {"$lt": value}})
        return self

    def in_list(self, field: str, values: list) -> "MetadataFilterBuilder":
        """Add in-list filter."""
        self._filters.append({field: {"$in": values}})
        return self

    def not_in_list(self, field: str, values: list) -> "MetadataFilterBuilder":
        """Add not-in-list filter."""
        self._filters.append({field: {"$nin": values}})
        return self

    def use_and(self) -> "MetadataFilterBuilder":
        """Use AND logic for combining filters."""
        self._logic = "and"
        return self

    def use_or(self) -> "MetadataFilterBuilder":
        """Use OR logic for combining filters."""
        self._logic = "or"
        return self

    def build(self) -> dict:
        """Build the ChromaDB where clause."""
        if len(self._filters) == 0:
            return {}

        if len(self._filters) == 1:
            return self._filters[0]

        if self._logic == "and":
            return {"$and": self._filters}
        else:
            return {"$or": self._filters}


# =============================================================================
# Part 9: Collection Statistics
# =============================================================================
class CollectionStats:
    """Compute statistics about a ChromaDB collection."""

    def __init__(self, collection: DocumentCollection):
        self._collection = collection

    def get_count(self) -> int:
        """Get total document count."""
        return self._collection.count()

    def get_metadata_distribution(self, field: str) -> dict[str, int]:
        """Get distribution of values for a metadata field."""
        # Would need to fetch all documents
        # This is a simplified implementation
        return {}

    def get_storage_estimate(self) -> dict:
        """Estimate storage usage."""
        count = self.get_count()

        # Rough estimates
        avg_vector_size = 384 * 4  # 384 dims * 4 bytes per float
        avg_metadata_size = 500  # bytes

        return {
            "document_count": count,
            "estimated_vectors_bytes": count * avg_vector_size,
            "estimated_metadata_bytes": count * avg_metadata_size,
        }

    def sample_documents(self, n: int = 10) -> list[Document]:
        """Get a random sample of documents."""
        # ChromaDB doesn't have native random sampling
        # This would need to fetch and sample
        return []


# =============================================================================
# Part 10: Complete Document Search System
# =============================================================================
class DocumentSearchSystem:
    """Complete document search system using ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "documents",
    ):
        self._client = ChromaDBClient(persist_directory=persist_directory)
        self._collection_name = collection_name
        self._preprocessor = DocumentPreprocessor()
        self._collection: Optional[DocumentCollection] = None

    def initialize(self, config: Optional[CollectionConfig] = None) -> None:
        """Initialize the search system."""
        if config is None:
            config = CollectionConfig(name=self._collection_name)

        self._client.create_collection(config, overwrite=False)
        self._collection = DocumentCollection(self._client, self._collection_name)

    def add_document(
        self, content: str, metadata: Optional[dict] = None, chunk: bool = True
    ) -> list[str]:
        """Add a document to the system."""
        if self._collection is None:
            self.initialize()

        doc = Document.from_text(content, metadata)

        if chunk:
            chunks = self._preprocessor.process(doc)
            return self._collection.add_batch(chunks)
        else:
            self._collection.add(doc)
            return [doc.id]

    def add_documents(
        self, documents: list[tuple[str, dict]], chunk: bool = True
    ) -> list[str]:
        """Add multiple documents."""
        all_ids = []
        for content, metadata in documents:
            ids = self.add_document(content, metadata, chunk)
            all_ids.extend(ids)
        return all_ids

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter: Optional[dict] = None,
        min_score: Optional[float] = None,
    ) -> list[dict]:
        """Search for relevant documents."""
        if self._collection is None:
            return []

        builder = QueryBuilder().with_text(query).limit(n_results)

        if filter:
            # Convert dict filter to QueryFilter objects
            filters = [
                QueryFilter(field=k, operator="$eq", value=v) for k, v in filter.items()
            ]
            if len(filters) == 1:
                builder = builder.where(filters[0])
            else:
                builder = builder.where_and(filters)

        results = self._collection.search(builder)

        if min_score:
            results = results.filter_by_score(min_score)

        return [
            {
                "id": r.id,
                "content": r.document,
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in results
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks."""
        if self._collection is None:
            return False

        # Delete the main document
        self._collection.delete(doc_id)

        # Also try to delete any chunks
        # This is a simplified approach - in production you'd track chunks
        for i in range(100):
            chunk_id = f"{doc_id}_chunk_{i}"
            if not self._collection.delete(chunk_id):
                break

        return True

    def get_stats(self) -> dict:
        """Get system statistics."""
        if self._collection is None:
            return {"count": 0}

        stats = CollectionStats(self._collection)
        return {"count": stats.get_count(), **stats.get_storage_estimate()}

    def clear(self) -> None:
        """Clear all documents."""
        self._client.delete_collection(self._collection_name)
        self.initialize()


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Initialize search system
    search = DocumentSearchSystem(
        persist_directory="./chroma_data", collection_name="knowledge_base"
    )
    search.initialize()

    # Add some documents
    docs = [
        (
            "Python is a versatile programming language.",
            {"topic": "programming", "year": 2024},
        ),
        (
            "Machine learning enables computers to learn from data.",
            {"topic": "ai", "year": 2024},
        ),
        (
            "Vector databases store embeddings for similarity search.",
            {"topic": "databases", "year": 2024},
        ),
    ]

    for content, metadata in docs:
        search.add_document(content, metadata, chunk=False)

    # Search
    results = search.search("What is Python?", n_results=3)

    print("Search results:")
    for result in results:
        print(f"  Score: {result['score']:.4f}")
        print(f"  Content: {result['content'][:50]}...")
        print()

    # Get stats
    stats = search.get_stats()
    print(f"\nTotal documents: {stats['count']}")
