"""
Week 13 - Solution 3: Advanced Vector Search

Complete implementations for multi-backend vector search with
hybrid search, reranking, and query processing.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Callable
from abc import ABC, abstractmethod
from enum import Enum
from importlib.util import find_spec
import hashlib
import math
import time

import numpy as np

# Check optional dependencies
HAS_FAISS = find_spec("faiss") is not None
HAS_SENTENCE_TRANSFORMERS = find_spec("sentence_transformers") is not None

if HAS_FAISS:
    import faiss

if HAS_SENTENCE_TRANSFORMERS:
    from sentence_transformers import SentenceTransformer, CrossEncoder


# =============================================================================
# Part 1: Vector Backend Protocol
# =============================================================================
class VectorBackend(Protocol):
    """Protocol for vector database backends."""

    def add(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict]
    ) -> None: ...

    def search(
        self, vector: list[float], k: int, filter: Optional[dict] = None
    ) -> list[dict]: ...

    def delete(self, ids: list[str]) -> int: ...

    def count(self) -> int: ...


class BackendType(Enum):
    """Supported vector database backends."""

    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    FAISS = "faiss"
    MEMORY = "memory"


# =============================================================================
# Part 2: Backend Factory
# =============================================================================
@dataclass
class BackendConfig:
    """Configuration for vector backends."""

    backend_type: BackendType
    dimension: int = 384

    # ChromaDB settings
    chroma_persist_dir: Optional[str] = None
    chroma_collection: str = "default"

    # Pinecone settings
    pinecone_api_key: Optional[str] = None
    pinecone_index: Optional[str] = None
    pinecone_environment: Optional[str] = None

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "default"

    # FAISS settings
    faiss_index_type: str = "flat"
    faiss_nlist: int = 100

    def validate(self) -> bool:
        """Validate configuration for the selected backend."""
        if self.dimension <= 0:
            return False

        if self.backend_type == BackendType.PINECONE:
            if not self.pinecone_api_key or not self.pinecone_index:
                return False

        return True


class BackendFactory:
    """Factory for creating vector database backends."""

    _backends: dict[BackendType, type] = {}

    @classmethod
    def register(cls, backend_type: BackendType, backend_class: type) -> None:
        """Register a backend implementation."""
        cls._backends[backend_type] = backend_class

    @classmethod
    def create(cls, config: BackendConfig) -> "VectorBackend":
        """Create a backend instance from config."""
        if config.backend_type not in cls._backends:
            raise ValueError(f"Backend {config.backend_type} not registered")

        backend_class = cls._backends[config.backend_type]

        # Create with appropriate arguments based on type
        if config.backend_type == BackendType.MEMORY:
            return backend_class(dimension=config.dimension)
        elif config.backend_type == BackendType.FAISS:
            return backend_class(
                dimension=config.dimension,
                index_type=config.faiss_index_type,
                nlist=config.faiss_nlist,
            )
        else:
            return backend_class(dimension=config.dimension)

    @classmethod
    def available_backends(cls) -> list[BackendType]:
        """List registered backends."""
        return list(cls._backends.keys())


# =============================================================================
# Part 3: In-Memory Backend
# =============================================================================
class InMemoryBackend:
    """Simple in-memory vector backend for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict] = {}

    def add(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict]
    ) -> None:
        """Add vectors to store."""
        for i, (vid, vec, meta) in enumerate(zip(ids, vectors, metadata)):
            if len(vec) != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch: {len(vec)} != {self._dimension}"
                )
            self._vectors[vid] = vec
            self._metadata[vid] = meta

    def search(
        self, vector: list[float], k: int, filter: Optional[dict] = None
    ) -> list[dict]:
        """Search for similar vectors using cosine similarity."""
        results = []

        for vid, stored_vec in self._vectors.items():
            meta = self._metadata.get(vid, {})

            # Apply filter
            if filter and not self._matches_filter(meta, filter):
                continue

            score = self._cosine_similarity(vector, stored_vec)
            results.append({"id": vid, "score": score, "metadata": meta})

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:k]

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        count = 0
        for vid in ids:
            if vid in self._vectors:
                del self._vectors[vid]
                del self._metadata[vid]
                count += 1
        return count

    def count(self) -> int:
        """Get vector count."""
        return len(self._vectors)

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    def _matches_filter(self, metadata: dict, filter: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True


# Register the backend
BackendFactory.register(BackendType.MEMORY, InMemoryBackend)


# =============================================================================
# Part 4: FAISS Backend Wrapper
# =============================================================================
class FAISSBackend:
    """FAISS vector backend wrapper."""

    def __init__(self, dimension: int, index_type: str = "flat", nlist: int = 100):
        self._dimension = dimension
        self._index_type = index_type
        self._nlist = nlist
        self._index = None
        self._id_map: dict[int, str] = {}  # Internal ID -> string ID
        self._reverse_map: dict[str, int] = {}  # String ID -> internal ID
        self._metadata: dict[str, dict] = {}
        self._deleted: set[str] = set()
        self._counter = 0

        self._create_index()

    def _create_index(self):
        """Create the appropriate FAISS index."""
        if not HAS_FAISS:
            self._index = None
            return

        if self._index_type == "flat":
            self._index = faiss.IndexFlatIP(self._dimension)
        elif self._index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self._dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self._dimension, self._nlist)
        elif self._index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(self._dimension, 32)
        else:
            self._index = faiss.IndexFlatIP(self._dimension)

    def add(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict]
    ) -> None:
        """Add vectors to FAISS index."""
        if self._index is None:
            # Mock implementation
            for vid, vec, meta in zip(ids, vectors, metadata):
                self._id_map[self._counter] = vid
                self._reverse_map[vid] = self._counter
                self._metadata[vid] = meta
                self._counter += 1
            return

        # Normalize vectors for cosine similarity
        vectors_np = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vectors_np)

        # Add to index
        start_id = self._counter
        self._index.add(vectors_np)

        # Map IDs
        for i, vid in enumerate(ids):
            internal_id = start_id + i
            self._id_map[internal_id] = vid
            self._reverse_map[vid] = internal_id
            self._metadata[vid] = metadata[i]

        self._counter = start_id + len(ids)

    def search(
        self, vector: list[float], k: int, filter: Optional[dict] = None
    ) -> list[dict]:
        """Search FAISS index with post-filtering."""
        if self._index is None:
            return []

        # Normalize query
        query = np.array([vector], dtype="float32")
        faiss.normalize_L2(query)

        # Over-fetch for filtering
        fetch_k = k * 3 if filter else k

        distances, indices = self._index.search(query, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue

            vid = self._id_map.get(idx)
            if not vid or vid in self._deleted:
                continue

            meta = self._metadata.get(vid, {})

            # Apply filter
            if filter and not self._matches_filter(meta, filter):
                continue

            results.append({"id": vid, "score": float(dist), "metadata": meta})

            if len(results) >= k:
                break

        return results

    def _matches_filter(self, metadata: dict, filter: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def delete(self, ids: list[str]) -> int:
        """Delete vectors (mark as deleted)."""
        count = 0
        for vid in ids:
            if vid in self._reverse_map and vid not in self._deleted:
                self._deleted.add(vid)
                count += 1
        return count

    def count(self) -> int:
        """Get active vector count."""
        return len(self._id_map) - len(self._deleted)

    def rebuild_index(self) -> None:
        """Rebuild index to remove deleted vectors."""
        # Would rebuild the index without deleted vectors
        pass


# Register FAISS backend
BackendFactory.register(BackendType.FAISS, FAISSBackend)


# =============================================================================
# Part 5: Embedding Provider
# =============================================================================
class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
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
            # Mock implementation
            h = hashlib.md5(text.encode()).hexdigest()
            return [
                int(h[i : i + 2], 16) / 255.0
                for i in range(0, min(len(h), self._dimension * 2), 2)
            ]

        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        if self._model is None:
            return [self.embed(t) for t in texts]

        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class CachedEmbeddingProvider(EmbeddingProvider):
    """Cached wrapper around an embedding provider."""

    def __init__(self, provider: EmbeddingProvider, max_cache_size: int = 10000):
        self._provider = provider
        self._cache: dict[str, list[float]] = {}
        self._max_cache_size = max_cache_size
        self._hits = 0
        self._misses = 0

    def _cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, text: str) -> list[float]:
        """Embed with caching."""
        key = self._cache_key(text)

        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        embedding = self._provider.embed(text)

        # Add to cache if not full
        if len(self._cache) < self._max_cache_size:
            self._cache[key] = embedding

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed with caching."""
        results = []
        to_embed = []
        to_embed_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                self._hits += 1
                results.append((i, self._cache[key]))
            else:
                self._misses += 1
                to_embed.append(text)
                to_embed_indices.append(i)

        # Embed uncached texts
        if to_embed:
            new_embeddings = self._provider.embed_batch(to_embed)

            for idx, (text, embedding) in enumerate(zip(to_embed, new_embeddings)):
                key = self._cache_key(text)
                if len(self._cache) < self._max_cache_size:
                    self._cache[key] = embedding
                results.append((to_embed_indices[idx], embedding))

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "cache_size": len(self._cache),
        }


# =============================================================================
# Part 6: Hybrid Search
# =============================================================================
@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    rerank: bool = True
    fusion_method: str = "rrf"


class HybridSearchEngine:
    """Hybrid search combining vector search and keyword search."""

    def __init__(
        self,
        vector_backend: VectorBackend,
        embedding_provider: EmbeddingProvider,
        config: HybridSearchConfig,
    ):
        self._backend = vector_backend
        self._embeddings = embedding_provider
        self._config = config
        self._documents: dict[str, str] = {}

    def add_document(self, doc_id: str, text: str, metadata: dict) -> None:
        """Add document for both vector and keyword search."""
        # Store text for keyword search
        self._documents[doc_id] = text.lower()

        # Add to vector backend
        embedding = self._embeddings.embed(text)
        self._backend.add([doc_id], [embedding], [metadata])

    def keyword_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Perform keyword search using BM25-like scoring."""
        query_terms = query.lower().split()
        results = []

        for doc_id, text in self._documents.items():
            score = 0.0

            for term in query_terms:
                if term in text:
                    # Simple TF scoring
                    tf = text.count(term)
                    score += math.log(1 + tf)

            if score > 0:
                results.append((doc_id, score))

        # Normalize scores
        if results:
            max_score = max(r[1] for r in results)
            if max_score > 0:
                results = [(doc_id, score / max_score) for doc_id, score in results]

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def vector_search(
        self, query: str, k: int, filter: Optional[dict] = None
    ) -> list[tuple[str, float]]:
        """Perform vector similarity search."""
        query_embedding = self._embeddings.embed(query)
        results = self._backend.search(query_embedding, k, filter)

        return [(r["id"], r["score"]) for r in results]

    def search(self, query: str, k: int, filter: Optional[dict] = None) -> list[dict]:
        """Perform hybrid search."""
        # Get results from both sources
        semantic_results = self.vector_search(query, k * 2, filter)
        keyword_results = self.keyword_search(query, k * 2)

        # Fuse results
        if self._config.fusion_method == "rrf":
            fused = self._fuse_rrf(semantic_results, keyword_results)
        else:
            fused = self._fuse_weighted(semantic_results, keyword_results)

        # Return top k
        return [
            {"id": doc_id, "score": score, "text": self._documents.get(doc_id, "")}
            for doc_id, score in fused[:k]
        ]

    def _fuse_rrf(
        self,
        semantic_results: list[tuple[str, float]],
        keyword_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Fuse results using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}

        # RRF from semantic results
        for rank, (doc_id, _) in enumerate(semantic_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # RRF from keyword results
        for rank, (doc_id, _) in enumerate(keyword_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Sort by fused score
        result = [(doc_id, score) for doc_id, score in scores.items()]
        result.sort(key=lambda x: x[1], reverse=True)

        return result

    def _fuse_weighted(
        self,
        semantic_results: list[tuple[str, float]],
        keyword_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fuse results using weighted combination."""
        scores: dict[str, float] = {}

        # Weighted semantic scores
        for doc_id, score in semantic_results:
            scores[doc_id] = self._config.semantic_weight * score

        # Add weighted keyword scores
        for doc_id, score in keyword_results:
            scores[doc_id] = scores.get(doc_id, 0) + self._config.keyword_weight * score

        result = [(doc_id, score) for doc_id, score in scores.items()]
        result.sort(key=lambda x: x[1], reverse=True)

        return result


# =============================================================================
# Part 7: Reranker
# =============================================================================
class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, documents: list[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        """Rerank documents for a query."""
        pass


class CrossEncoderReranker(Reranker):
    """Reranker using cross-encoder models."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

        if HAS_SENTENCE_TRANSFORMERS:
            self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, documents: list[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        """Rerank using cross-encoder."""
        if self._model is None or not documents:
            # Return as-is with mock scores
            return [(doc_id, 1.0 / (i + 1)) for i, (doc_id, _) in enumerate(documents)][
                :top_k
            ]

        # Create pairs for scoring
        pairs = [(query, text) for _, text in documents]
        scores = self._model.predict(pairs)

        # Combine with IDs and sort
        results = [(documents[i][0], float(scores[i])) for i in range(len(documents))]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


class MMRReranker(Reranker):
    """Maximal Marginal Relevance reranker for diversity."""

    def __init__(
        self, embedding_provider: EmbeddingProvider, lambda_param: float = 0.5
    ):
        self._embeddings = embedding_provider
        self._lambda = lambda_param

    def rerank(
        self, query: str, documents: list[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        """Rerank using MMR for diversity."""
        if not documents:
            return []

        # Get embeddings
        query_emb = self._embeddings.embed(query)
        doc_embs = self._embeddings.embed_batch([text for _, text in documents])

        # Calculate relevance scores
        relevance = []
        for emb in doc_embs:
            sim = self._cosine_similarity(query_emb, emb)
            relevance.append(sim)

        # MMR selection
        selected = []
        remaining = list(range(len(documents)))

        while len(selected) < top_k and remaining:
            best_idx = None
            best_score = float("-inf")

            for idx in remaining:
                # Relevance to query
                rel = relevance[idx]

                # Max similarity to already selected
                max_sim = 0.0
                for sel_idx in selected:
                    sim = self._cosine_similarity(doc_embs[idx], doc_embs[sel_idx])
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = self._lambda * rel - (1 - self._lambda) * max_sim

                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        # Return results
        return [(documents[idx][0], relevance[idx]) for idx in selected]

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)


# =============================================================================
# Part 8: Query Processing Pipeline
# =============================================================================
@dataclass
class QueryContext:
    """Context passed through the query pipeline."""

    original_query: str
    processed_query: str = ""
    query_embedding: Optional[list[float]] = None
    filters: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.processed_query:
            self.processed_query = self.original_query


class QueryProcessor(ABC):
    """Abstract base class for query processors."""

    @abstractmethod
    def process(self, context: QueryContext) -> QueryContext:
        """Process the query context."""
        pass


class QueryExpander(QueryProcessor):
    """Expand query with synonyms or related terms."""

    def __init__(self, synonym_map: Optional[dict[str, list[str]]] = None):
        self._synonyms = synonym_map or {}

    def process(self, context: QueryContext) -> QueryContext:
        """Expand query with synonyms."""
        words = context.processed_query.lower().split()
        expanded = []

        for word in words:
            expanded.append(word)
            if word in self._synonyms:
                expanded.extend(self._synonyms[word][:2])  # Add up to 2 synonyms

        context.processed_query = " ".join(expanded)
        return context

    def add_synonyms(self, word: str, synonyms: list[str]) -> None:
        """Add synonym mapping."""
        self._synonyms[word.lower()] = synonyms


class HypotheticalDocumentEmbedder(QueryProcessor):
    """HyDE: Generate a hypothetical document to improve search."""

    def __init__(self, llm_generator: Optional[Callable[[str], str]] = None):
        self._generator = llm_generator

    def process(self, context: QueryContext) -> QueryContext:
        """Generate hypothetical document from query."""
        if self._generator:
            # Generate hypothetical answer
            hypothetical = self._generator(context.original_query)
            context.processed_query = hypothetical
            context.metadata["hyde_generated"] = True

        return context


class QueryPipeline:
    """Pipeline for processing queries before search."""

    def __init__(self):
        self._processors: list[QueryProcessor] = []

    def add(self, processor: QueryProcessor) -> "QueryPipeline":
        """Add a processor to the pipeline."""
        self._processors.append(processor)
        return self

    def process(self, query: str, filters: Optional[dict] = None) -> QueryContext:
        """Process query through all processors."""
        context = QueryContext(
            original_query=query, processed_query=query, filters=filters
        )

        for processor in self._processors:
            context = processor.process(context)

        return context


# =============================================================================
# Part 9: Multi-Backend Search Manager
# =============================================================================
class MultiBackendSearchManager:
    """Manage searches across multiple vector backends."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self._embeddings = embedding_provider
        self._backends: dict[str, VectorBackend] = {}
        self._primary: Optional[str] = None

    def add_backend(
        self, name: str, backend: VectorBackend, primary: bool = False
    ) -> None:
        """Add a backend."""
        self._backends[name] = backend
        if primary or self._primary is None:
            self._primary = name

    def remove_backend(self, name: str) -> bool:
        """Remove a backend."""
        if name in self._backends:
            del self._backends[name]
            if self._primary == name:
                self._primary = next(iter(self._backends), None)
            return True
        return False

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
        backends: Optional[list[str]] = None,
    ) -> dict[str, bool]:
        """Add document to specified backends."""
        embedding = self._embeddings.embed(text)
        target_backends = backends or list(self._backends.keys())

        results = {}
        for name in target_backends:
            if name in self._backends:
                try:
                    self._backends[name].add([doc_id], [embedding], [metadata])
                    results[name] = True
                except Exception:
                    results[name] = False

        return results

    def search(
        self,
        query: str,
        k: int,
        backends: Optional[list[str]] = None,
        aggregate: bool = True,
    ) -> dict:
        """Search across backends."""
        query_embedding = self._embeddings.embed(query)
        target_backends = backends or list(self._backends.keys())

        all_results = {}
        for name in target_backends:
            if name in self._backends:
                results = self._backends[name].search(query_embedding, k)
                all_results[name] = results

        if not aggregate:
            return all_results

        # Aggregate results using RRF
        scores: dict[str, float] = {}
        result_data: dict[str, dict] = {}

        for backend_name, results in all_results.items():
            for rank, result in enumerate(results):
                doc_id = result["id"]
                rrf_score = 1.0 / (60 + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score
                result_data[doc_id] = result

        # Sort and return
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return {
            "results": [
                {**result_data[doc_id], "aggregated_score": scores[doc_id]}
                for doc_id in sorted_ids[:k]
            ]
        }

    def search_with_failover(self, query: str, k: int) -> list[dict]:
        """Search with automatic failover."""
        query_embedding = self._embeddings.embed(query)

        # Try primary first
        if self._primary and self._primary in self._backends:
            try:
                results = self._backends[self._primary].search(query_embedding, k)
                if results:
                    return results
            except Exception:
                pass

        # Failover to other backends
        for name, backend in self._backends.items():
            if name == self._primary:
                continue
            try:
                results = backend.search(query_embedding, k)
                if results:
                    return results
            except Exception:
                continue

        return []

    def health_check(self) -> dict[str, bool]:
        """Check health of all backends."""
        results = {}
        for name, backend in self._backends.items():
            try:
                backend.count()
                results[name] = True
            except Exception:
                results[name] = False
        return results


# =============================================================================
# Part 10: Complete Advanced Search System
# =============================================================================
class AdvancedSearchSystem:
    """Complete advanced search system with all features."""

    def __init__(
        self,
        backend_configs: list[BackendConfig],
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_hybrid: bool = True,
        enable_cache: bool = True,
    ):
        self._backend_configs = backend_configs
        self._embedding_model = embedding_model
        self._enable_hybrid = enable_hybrid
        self._enable_cache = enable_cache

        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._search_manager: Optional[MultiBackendSearchManager] = None
        self._hybrid_engine: Optional[HybridSearchEngine] = None
        self._query_pipeline: Optional[QueryPipeline] = None
        self._reranker: Optional[Reranker] = None

    def initialize(self) -> None:
        """Initialize all components."""
        # Create embedding provider
        base_provider = SentenceTransformerProvider(self._embedding_model)

        if self._enable_cache:
            self._embedding_provider = CachedEmbeddingProvider(base_provider)
        else:
            self._embedding_provider = base_provider

        # Create backends and search manager
        self._search_manager = MultiBackendSearchManager(self._embedding_provider)

        for i, config in enumerate(self._backend_configs):
            backend = BackendFactory.create(config)
            is_primary = i == 0
            self._search_manager.add_backend(
                f"backend_{i}", backend, primary=is_primary
            )

        # Create query pipeline
        self._query_pipeline = QueryPipeline()

        # Create reranker
        self._reranker = MMRReranker(self._embedding_provider)

        # Create hybrid engine if enabled
        if self._enable_hybrid and self._backend_configs:
            config = HybridSearchConfig()
            primary_backend = self._search_manager._backends.get(
                self._search_manager._primary
            )
            if primary_backend:
                self._hybrid_engine = HybridSearchEngine(
                    primary_backend, self._embedding_provider, config
                )

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
        chunk: bool = False,
        chunk_size: int = 500,
    ) -> list[str]:
        """Add a document to the system."""
        if self._search_manager is None:
            self.initialize()

        metadata = metadata or {}

        if not chunk:
            self._search_manager.add_document(doc_id, text, metadata)

            # Also add to hybrid engine if enabled
            if self._hybrid_engine:
                self._hybrid_engine.add_document(doc_id, text, metadata)

            return [doc_id]

        # Chunk the document
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            chunk_id = f"{doc_id}_chunk_{i // chunk_size}"
            chunk_meta = {
                **metadata,
                "parent_id": doc_id,
                "chunk_index": i // chunk_size,
            }
            chunks.append((chunk_id, chunk_text, chunk_meta))

        # Add all chunks
        chunk_ids = []
        for chunk_id, chunk_text, chunk_meta in chunks:
            self._search_manager.add_document(chunk_id, chunk_text, chunk_meta)
            if self._hybrid_engine:
                self._hybrid_engine.add_document(chunk_id, chunk_text, chunk_meta)
            chunk_ids.append(chunk_id)

        return chunk_ids

    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
        use_hybrid: Optional[bool] = None,
        rerank: bool = False,
        rerank_k: Optional[int] = None,
    ) -> list[dict]:
        """Perform search with all features."""
        if self._search_manager is None:
            self.initialize()

        # Process query
        context = self._query_pipeline.process(query, filter)
        processed_query = context.processed_query

        # Decide search method
        use_hybrid = use_hybrid if use_hybrid is not None else self._enable_hybrid

        if use_hybrid and self._hybrid_engine:
            results = self._hybrid_engine.search(
                processed_query, k * 2 if rerank else k, filter
            )
        else:
            search_results = self._search_manager.search(
                processed_query, k * 2 if rerank else k
            )
            results = search_results.get("results", [])

        # Rerank if requested
        if rerank and results and self._reranker:
            # Prepare documents for reranking
            docs = [(r.get("id", ""), r.get("text", "")) for r in results]
            rerank_k = rerank_k or k
            reranked = self._reranker.rerank(query, docs, rerank_k)

            # Map back to full results
            id_to_result = {r.get("id"): r for r in results}
            results = [
                {**id_to_result.get(doc_id, {}), "score": score}
                for doc_id, score in reranked
            ]

        return results[:k]

    def batch_search(
        self, queries: list[str], k: int = 10, parallel: bool = True
    ) -> list[list[dict]]:
        """Perform multiple searches."""
        # Simple sequential implementation
        return [self.search(query, k) for query in queries]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from all backends."""
        if self._search_manager is None:
            return False

        for backend in self._search_manager._backends.values():
            backend.delete([doc_id])

        return True

    def get_statistics(self) -> dict:
        """Get system statistics."""
        if self._search_manager is None:
            return {}

        stats = {"backends": {}, "health": self._search_manager.health_check()}

        for name, backend in self._search_manager._backends.items():
            stats["backends"][name] = {"count": backend.count()}

        if self._enable_cache and isinstance(
            self._embedding_provider, CachedEmbeddingProvider
        ):
            stats["embedding_cache"] = self._embedding_provider.cache_stats()

        return stats

    def optimize(self) -> None:
        """Optimize all backends."""
        if self._search_manager is None:
            return

        for backend in self._search_manager._backends.values():
            if hasattr(backend, "rebuild_index"):
                backend.rebuild_index()


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Configure backends
    configs = [BackendConfig(backend_type=BackendType.MEMORY, dimension=384)]

    # Create search system
    search = AdvancedSearchSystem(
        backend_configs=configs,
        embedding_model="all-MiniLM-L6-v2",
        enable_hybrid=True,
        enable_cache=True,
    )
    search.initialize()

    # Add documents
    documents = [
        (
            "doc1",
            "Python is a versatile programming language.",
            {"topic": "programming"},
        ),
        ("doc2", "Machine learning enables pattern recognition.", {"topic": "ai"}),
        (
            "doc3",
            "Vector databases store embeddings efficiently.",
            {"topic": "databases"},
        ),
    ]

    for doc_id, text, metadata in documents:
        search.add_document(doc_id, text, metadata)

    # Search with reranking
    results = search.search(
        "What programming language should I learn?", k=3, use_hybrid=True, rerank=True
    )

    print("Search results:")
    for result in results:
        print(f"  {result.get('id', 'unknown')}: {result.get('score', 0):.4f}")

    # Get statistics
    stats = search.get_statistics()
    print(f"\nSystem statistics: {stats}")
