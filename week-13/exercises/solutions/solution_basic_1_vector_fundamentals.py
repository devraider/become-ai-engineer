"""
Week 13 - Solution 1: Vector Fundamentals

Complete implementations for vector operations, similarity calculations,
and in-memory vector storage.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Iterator
from abc import ABC, abstractmethod
from enum import Enum
import math
import hashlib


# =============================================================================
# Part 1: Vector Representation
# =============================================================================
@dataclass
class Vector:
    """Represents a vector with metadata."""

    id: str
    values: list[float]
    metadata: dict = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return len(self.values)

    def normalize(self) -> "Vector":
        """Return a normalized (unit) vector."""
        magnitude = math.sqrt(sum(v * v for v in self.values))
        if magnitude == 0:
            return Vector(
                id=self.id, values=self.values.copy(), metadata=self.metadata.copy()
            )

        normalized_values = [v / magnitude for v in self.values]
        return Vector(
            id=self.id, values=normalized_values, metadata=self.metadata.copy()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"id": self.id, "values": self.values, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "Vector":
        """Create from dictionary."""
        return cls(
            id=data["id"], values=data["values"], metadata=data.get("metadata", {})
        )


# =============================================================================
# Part 2: Distance Metrics and Similarity Calculator
# =============================================================================
class DistanceMetric(Enum):
    """Supported distance/similarity metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class SimilarityCalculator:
    """Calculate similarity/distance between vectors."""

    def __init__(self, metric: DistanceMetric = DistanceMetric.COSINE):
        self._metric = metric

    def calculate(self, v1: list[float], v2: list[float]) -> float:
        """Calculate similarity/distance based on metric."""
        if self._metric == DistanceMetric.COSINE:
            return self._cosine_similarity(v1, v2)
        elif self._metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_distance(v1, v2)
        elif self._metric == DistanceMetric.DOT_PRODUCT:
            return self._dot_product(v1, v2)
        elif self._metric == DistanceMetric.MANHATTAN:
            return self._manhattan_distance(v1, v2)
        else:
            raise ValueError(f"Unknown metric: {self._metric}")

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity (0 to 1 for normalized vectors)."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    def _euclidean_distance(self, v1: list[float], v2: list[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def _dot_product(self, v1: list[float], v2: list[float]) -> float:
        """Calculate dot product."""
        return sum(a * b for a, b in zip(v1, v2))

    def _manhattan_distance(self, v1: list[float], v2: list[float]) -> float:
        """Calculate Manhattan (L1) distance."""
        return sum(abs(a - b) for a, b in zip(v1, v2))

    def similarity_to_distance(self, similarity: float) -> float:
        """Convert similarity score to distance."""
        if self._metric == DistanceMetric.COSINE:
            return 1.0 - similarity
        elif self._metric == DistanceMetric.DOT_PRODUCT:
            # Assuming normalized vectors, max dot product is 1
            return 1.0 - similarity
        else:
            # For distance metrics, return as-is
            return similarity

    def distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score."""
        if self._metric in (DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT):
            return 1.0 - distance
        else:
            # For distance metrics, use inverse
            return 1.0 / (1.0 + distance)


# =============================================================================
# Part 3: Search Result
# =============================================================================
@dataclass
class SearchResult:
    """Represents a search result."""

    id: str
    score: float
    vector: Optional[Vector] = None
    distance: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {"id": self.id, "score": self.score}
        if self.distance is not None:
            result["distance"] = self.distance
        if self.vector:
            result["metadata"] = self.vector.metadata
        return result


# =============================================================================
# Part 4: In-Memory Vector Store
# =============================================================================
class InMemoryVectorStore:
    """Simple in-memory vector store with search capability."""

    def __init__(self, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE):
        self._dimension = dimension
        self._vectors: dict[str, Vector] = {}
        self._calculator = SimilarityCalculator(metric)
        self._metric = metric

    def add(self, vector: Vector) -> None:
        """Add a vector to the store."""
        if vector.dimension != self._dimension:
            raise ValueError(
                f"Vector dimension {vector.dimension} doesn't match "
                f"store dimension {self._dimension}"
            )
        self._vectors[vector.id] = vector

    def add_batch(self, vectors: list[Vector]) -> None:
        """Add multiple vectors."""
        for vector in vectors:
            self.add(vector)

    def get(self, vector_id: str) -> Optional[Vector]:
        """Get a vector by ID."""
        return self._vectors.get(vector_id)

    def delete(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        if vector_id in self._vectors:
            del self._vectors[vector_id]
            return True
        return False

    def count(self) -> int:
        """Get number of vectors."""
        return len(self._vectors)

    def search(
        self, query: list[float], k: int = 10, filter: Optional[dict] = None
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if len(query) != self._dimension:
            raise ValueError(
                f"Query dimension {len(query)} doesn't match "
                f"store dimension {self._dimension}"
            )

        results = []

        for vector in self._vectors.values():
            # Apply filter if provided
            if filter and not self._matches_filter(vector.metadata, filter):
                continue

            # Calculate similarity/distance
            score = self._calculator.calculate(query, vector.values)

            # Convert to similarity if using distance metric
            if self._metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.MANHATTAN):
                similarity = self._calculator.distance_to_similarity(score)
                distance = score
            else:
                similarity = score
                distance = self._calculator.similarity_to_distance(score)

            results.append(
                SearchResult(
                    id=vector.id, score=similarity, vector=vector, distance=distance
                )
            )

        # Sort by score (descending for similarity)
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:k]

    def _matches_filter(self, metadata: dict, filter: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def clear(self) -> None:
        """Clear all vectors."""
        self._vectors.clear()


# =============================================================================
# Part 5: Embedding Model Interface
# =============================================================================
class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

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


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate deterministic mock embedding from text hash."""
        # Create deterministic embedding based on text hash
        h = hashlib.md5(text.encode()).hexdigest()

        # Generate values from hash
        values = []
        for i in range(self._dimension):
            # Use different parts of hash for different dimensions
            idx = (i * 2) % len(h)
            val = int(h[idx : idx + 2], 16) / 255.0
            values.append(val)

        # Normalize
        magnitude = math.sqrt(sum(v * v for v in values))
        if magnitude > 0:
            values = [v / magnitude for v in values]

        return values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


# =============================================================================
# Part 6: Vector Index Interface
# =============================================================================
class VectorIndex(ABC):
    """Abstract base class for vector indexes."""

    @abstractmethod
    def add(self, vectors: list[Vector]) -> None:
        """Add vectors to index."""
        pass

    @abstractmethod
    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        """Search for k nearest neighbors."""
        pass

    @abstractmethod
    def remove(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get number of indexed vectors."""
        pass


class FlatIndex(VectorIndex):
    """Flat (brute force) index."""

    def __init__(self, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE):
        self._dimension = dimension
        self._vectors: dict[str, list[float]] = {}
        self._calculator = SimilarityCalculator(metric)
        self._metric = metric

    def add(self, vectors: list[Vector]) -> None:
        """Add vectors to index."""
        for vector in vectors:
            if vector.dimension != self._dimension:
                raise ValueError(
                    f"Dimension mismatch: {vector.dimension} != {self._dimension}"
                )
            self._vectors[vector.id] = vector.values

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        """Search for k nearest neighbors."""
        scores = []

        for vid, values in self._vectors.items():
            score = self._calculator.calculate(query, values)

            # Convert distance metrics to similarity
            if self._metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.MANHATTAN):
                score = self._calculator.distance_to_similarity(score)

            scores.append((vid, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]

    def remove(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        for vid in ids:
            if vid in self._vectors:
                del self._vectors[vid]

    def count(self) -> int:
        """Get number of indexed vectors."""
        return len(self._vectors)


# =============================================================================
# Part 7: Batch Processor
# =============================================================================
class BatchProcessor:
    """Process items in batches."""

    def __init__(self, batch_size: int = 100):
        self._batch_size = batch_size

    def process(self, items: list, processor: Callable[[Any], Any]) -> Iterator:
        """Process items in batches, yielding results."""
        for batch in self.iterate_batches(items):
            for item in batch:
                yield processor(item)

    def iterate_batches(self, items: list) -> Iterator[list]:
        """Iterate over items in batches."""
        for i in range(0, len(items), self._batch_size):
            yield items[i : i + self._batch_size]

    def process_batch(
        self, items: list, batch_processor: Callable[[list], list]
    ) -> list:
        """Process entire batches at once."""
        results = []
        for batch in self.iterate_batches(items):
            batch_results = batch_processor(batch)
            results.extend(batch_results)
        return results


# =============================================================================
# Part 8: Similarity Matrix
# =============================================================================
class SimilarityMatrix:
    """Compute pairwise similarity matrix."""

    def __init__(
        self, vectors: list[list[float]], metric: DistanceMetric = DistanceMetric.COSINE
    ):
        self._vectors = vectors
        self._calculator = SimilarityCalculator(metric)
        self._matrix: Optional[list[list[float]]] = None

    def compute(self) -> list[list[float]]:
        """Compute the full similarity matrix."""
        n = len(self._vectors)
        self._matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                sim = self._calculator.calculate(self._vectors[i], self._vectors[j])
                self._matrix[i][j] = sim
                self._matrix[j][i] = sim  # Symmetric

        return self._matrix

    def get(self, i: int, j: int) -> float:
        """Get similarity between vectors i and j."""
        if self._matrix is None:
            self.compute()
        return self._matrix[i][j]

    def find_most_similar(self, k: int = 5) -> list[tuple[int, int, float]]:
        """Find k most similar pairs."""
        if self._matrix is None:
            self.compute()

        pairs = []
        n = len(self._vectors)

        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, self._matrix[i][j]))

        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs[:k]

    def find_outliers(self, threshold: float = 0.3) -> list[int]:
        """Find vectors with low average similarity."""
        if self._matrix is None:
            self.compute()

        n = len(self._vectors)
        outliers = []

        for i in range(n):
            avg_sim = sum(self._matrix[i]) / n
            if avg_sim < threshold:
                outliers.append(i)

        return outliers


# =============================================================================
# Part 9: Vector Statistics
# =============================================================================
class VectorStats:
    """Compute statistics over a collection of vectors."""

    @staticmethod
    def compute(vectors: list[Vector]) -> dict:
        """Compute comprehensive statistics."""
        if not vectors:
            return {"count": 0, "dimension": 0}

        dimension = vectors[0].dimension
        count = len(vectors)

        # Compute mean vector
        mean = [0.0] * dimension
        for vector in vectors:
            for i, v in enumerate(vector.values):
                mean[i] += v
        mean = [m / count for m in mean]

        # Compute std per dimension
        std = [0.0] * dimension
        for vector in vectors:
            for i, v in enumerate(vector.values):
                std[i] += (v - mean[i]) ** 2
        std = [math.sqrt(s / count) for s in std]

        # Compute magnitudes
        magnitudes = []
        for vector in vectors:
            mag = math.sqrt(sum(v * v for v in vector.values))
            magnitudes.append(mag)

        return {
            "count": count,
            "dimension": dimension,
            "mean": mean,
            "std": std,
            "mean_magnitude": sum(magnitudes) / count,
            "min_magnitude": min(magnitudes),
            "max_magnitude": max(magnitudes),
        }

    @staticmethod
    def compute_centroid(vectors: list[Vector]) -> list[float]:
        """Compute centroid of vectors."""
        if not vectors:
            return []

        dimension = vectors[0].dimension
        centroid = [0.0] * dimension

        for vector in vectors:
            for i, v in enumerate(vector.values):
                centroid[i] += v

        count = len(vectors)
        return [c / count for c in centroid]


# =============================================================================
# Part 10: Simple Vector Database
# =============================================================================
class SimpleVectorDB:
    """Simple vector database with collection support."""

    def __init__(self):
        self._collections: dict[str, InMemoryVectorStore] = {}

    def create_collection(
        self, name: str, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE
    ) -> None:
        """Create a new collection."""
        if name in self._collections:
            raise ValueError(f"Collection {name} already exists")

        self._collections[name] = InMemoryVectorStore(
            dimension=dimension, metric=metric
        )

    def get_collection(self, name: str) -> InMemoryVectorStore:
        """Get a collection by name."""
        if name not in self._collections:
            raise ValueError(f"Collection {name} not found")
        return self._collections[name]

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if name in self._collections:
            del self._collections[name]
            return True
        return False

    def list_collections(self) -> list[str]:
        """List all collection names."""
        return list(self._collections.keys())

    def insert(
        self, collection: str, id: str, values: list[float], metadata: dict
    ) -> None:
        """Insert a vector into a collection."""
        store = self.get_collection(collection)
        vector = Vector(id=id, values=values, metadata=metadata)
        store.add(vector)

    def search(
        self,
        collection: str,
        query: list[float],
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search a collection."""
        store = self.get_collection(collection)
        return store.search(query, k, filter)

    def delete(self, collection: str, id: str) -> bool:
        """Delete a vector from a collection."""
        store = self.get_collection(collection)
        return store.delete(id)

    def count(self, collection: str) -> int:
        """Get vector count in collection."""
        store = self.get_collection(collection)
        return store.count()


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Create embedding model
    model = MockEmbeddingModel(dimension=128)

    # Create vector store
    store = InMemoryVectorStore(dimension=128)

    # Add some documents
    documents = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "Vector databases store embeddings",
    ]

    for i, doc in enumerate(documents):
        embedding = model.embed(doc)
        vector = Vector(id=f"doc_{i}", values=embedding, metadata={"text": doc})
        store.add(vector)

    print(f"Added {store.count()} documents")

    # Search
    query = model.embed("What is Python?")
    results = store.search(query, k=2)

    print("\nSearch results:")
    for result in results:
        print(f"  {result.id}: score={result.score:.4f}")
        print(f"    Text: {result.vector.metadata.get('text', 'N/A')}")
