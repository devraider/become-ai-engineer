"""
Week 13 - Exercise 1 (Basic): Vector Database Fundamentals

Learn core concepts of vector operations, similarity calculations,
and basic vector storage.

Complete the TODOs to implement each component.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum


# =============================================================================
# Part 1: Vector Representation
# =============================================================================
@dataclass
class Vector:
    """
    Represents a vector with its data and metadata.

    Attributes:
        id: Unique identifier for the vector
        values: The actual vector values
        metadata: Optional metadata dictionary
        text: Optional original text (if embedding)

    TODO: Implement the following methods:
    - dimension: Return the dimensionality of the vector
    - normalize: Return a normalized copy of this vector
    - to_numpy: Convert values to numpy array
    - from_list: Class method to create from list
    """

    id: str
    values: list[float]
    metadata: dict = field(default_factory=dict)
    text: Optional[str] = None

    def dimension(self) -> int:
        """Return the dimensionality of the vector."""
        # TODO: Implement this method
        pass

    def normalize(self) -> "Vector":
        """Return a normalized (unit length) copy of this vector."""
        # TODO: Implement this method
        # Hint: Use numpy for efficient computation
        pass

    def to_numpy(self) -> np.ndarray:
        """Convert values to numpy array."""
        # TODO: Implement this method
        pass

    @classmethod
    def from_list(cls, id: str, values: list[float], **kwargs) -> "Vector":
        """Create a Vector from a list of values."""
        # TODO: Implement this class method
        pass


# =============================================================================
# Part 2: Distance Metrics
# =============================================================================
class DistanceMetric(str, Enum):
    """Supported distance/similarity metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class SimilarityCalculator:
    """
    Calculator for various similarity/distance metrics.

    TODO: Implement all similarity calculation methods.
    """

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity = (a · b) / (||a|| * ||b||)
        Range: -1 to 1 (1 = identical direction)

        TODO: Implement cosine similarity
        """
        pass

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Euclidean (L2) distance between two vectors.

        Euclidean distance = sqrt(sum((a_i - b_i)^2))
        Range: 0 to infinity (0 = identical)

        TODO: Implement Euclidean distance
        """
        pass

    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate dot product between two vectors.

        Dot product = sum(a_i * b_i)
        Range: -infinity to infinity

        TODO: Implement dot product
        """
        pass

    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Manhattan (L1) distance between two vectors.

        Manhattan distance = sum(|a_i - b_i|)
        Range: 0 to infinity (0 = identical)

        TODO: Implement Manhattan distance
        """
        pass

    @classmethod
    def calculate(cls, a: np.ndarray, b: np.ndarray, metric: DistanceMetric) -> float:
        """
        Calculate similarity/distance using the specified metric.

        TODO: Implement metric dispatch
        """
        pass


# =============================================================================
# Part 3: Search Result
# =============================================================================
@dataclass
class SearchResult:
    """
    Represents a search result with the matched vector and score.

    Attributes:
        vector: The matched vector
        score: Similarity/distance score
        rank: Position in results (1-indexed)

    TODO: Implement comparison methods for sorting results.
    """

    vector: Vector
    score: float
    rank: int = 0

    def __lt__(self, other: "SearchResult") -> bool:
        """Compare results for sorting (higher score = better)."""
        # TODO: Implement comparison
        pass

    def __eq__(self, other: "SearchResult") -> bool:
        """Check equality based on vector ID."""
        # TODO: Implement equality check
        pass


# =============================================================================
# Part 4: In-Memory Vector Store
# =============================================================================
class InMemoryVectorStore:
    """
    Simple in-memory vector store for learning purposes.

    This store holds vectors in memory and performs brute-force
    similarity search. Not suitable for large-scale use.

    TODO: Implement all methods for vector storage and retrieval.
    """

    def __init__(self, metric: DistanceMetric = DistanceMetric.COSINE):
        """Initialize the vector store."""
        self._vectors: dict[str, Vector] = {}
        self._metric = metric
        self._calculator = SimilarityCalculator()

    def add(self, vector: Vector) -> None:
        """
        Add a vector to the store.

        TODO: Implement vector addition
        """
        pass

    def add_batch(self, vectors: list[Vector]) -> None:
        """
        Add multiple vectors to the store.

        TODO: Implement batch addition
        """
        pass

    def get(self, vector_id: str) -> Optional[Vector]:
        """
        Retrieve a vector by ID.

        TODO: Implement vector retrieval
        """
        pass

    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector by ID.

        Returns True if deleted, False if not found.

        TODO: Implement vector deletion
        """
        pass

    def search(
        self, query: np.ndarray, k: int = 10, filter_fn: Optional[callable] = None
    ) -> list[SearchResult]:
        """
        Search for the k most similar vectors.

        Args:
            query: Query vector as numpy array
            k: Number of results to return
            filter_fn: Optional function to filter vectors (takes Vector, returns bool)

        Returns:
            List of SearchResult objects sorted by similarity

        TODO: Implement brute-force similarity search
        """
        pass

    def count(self) -> int:
        """Return the number of vectors in the store."""
        # TODO: Implement count
        pass

    def clear(self) -> None:
        """Remove all vectors from the store."""
        # TODO: Implement clear
        pass


# =============================================================================
# Part 5: Embedding Interface
# =============================================================================
class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.

    TODO: Review this interface - implementations will be in exercises.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class MockEmbeddingModel(EmbeddingModel):
    """
    Mock embedding model for testing (random embeddings).

    TODO: Implement the mock embedding model.
    """

    def __init__(self, dimension: int = 384, seed: Optional[int] = None):
        self._dimension = dimension
        self._rng = np.random.RandomState(seed)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """Generate a deterministic random embedding for text."""
        # TODO: Implement caching for consistent embeddings
        pass

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        # TODO: Implement batch embedding
        pass

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        # TODO: Implement dimension property
        pass


# =============================================================================
# Part 6: Vector Index Interface
# =============================================================================
class VectorIndex(ABC):
    """
    Abstract base class for vector indexes.

    Defines the interface for different indexing strategies.
    """

    @abstractmethod
    def build(self, vectors: list[Vector]) -> None:
        """Build the index from a list of vectors."""
        pass

    @abstractmethod
    def add(self, vector: Vector) -> None:
        """Add a single vector to the index."""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Search for k nearest neighbors."""
        pass

    @abstractmethod
    def remove(self, vector_id: str) -> bool:
        """Remove a vector from the index."""
        pass


class FlatIndex(VectorIndex):
    """
    Flat (brute-force) index implementation.

    Performs exact nearest neighbor search by comparing
    the query against all vectors.

    TODO: Implement the flat index.
    """

    def __init__(self, metric: DistanceMetric = DistanceMetric.COSINE):
        self._vectors: dict[str, np.ndarray] = {}
        self._metric = metric

    def build(self, vectors: list[Vector]) -> None:
        """Build index from vectors."""
        # TODO: Implement index building
        pass

    def add(self, vector: Vector) -> None:
        """Add a vector to the index."""
        # TODO: Implement vector addition
        pass

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """
        Search for k nearest neighbors.

        Returns list of (vector_id, score) tuples.

        TODO: Implement brute-force search
        """
        pass

    def remove(self, vector_id: str) -> bool:
        """Remove a vector from the index."""
        # TODO: Implement vector removal
        pass


# =============================================================================
# Part 7: Batch Operations
# =============================================================================
class BatchProcessor:
    """
    Processor for batch vector operations.

    Helps with efficient processing of large vector sets.

    TODO: Implement batch processing methods.
    """

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

    def process_in_batches(self, items: list, processor: callable) -> list:
        """
        Process items in batches.

        Args:
            items: List of items to process
            processor: Function to process a batch (takes list, returns list)

        Returns:
            Flattened list of all results

        TODO: Implement batch processing
        """
        pass

    def create_batches(self, items: list) -> list[list]:
        """
        Split items into batches.

        TODO: Implement batch creation
        """
        pass


# =============================================================================
# Part 8: Similarity Matrix
# =============================================================================
class SimilarityMatrix:
    """
    Compute and store pairwise similarity matrix.

    Useful for clustering and visualization.

    TODO: Implement similarity matrix computation.
    """

    def __init__(self, metric: DistanceMetric = DistanceMetric.COSINE):
        self._metric = metric
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[str] = []

    def compute(self, vectors: list[Vector]) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Returns an NxN matrix where entry [i,j] is the
        similarity between vectors[i] and vectors[j].

        TODO: Implement similarity matrix computation
        """
        pass

    def get_most_similar(self, vector_id: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Get the k most similar vectors to a given vector.

        TODO: Implement using the precomputed matrix
        """
        pass

    def get_clusters(self, threshold: float) -> list[list[str]]:
        """
        Get clusters of similar vectors based on threshold.

        Simple threshold-based clustering.

        TODO: Implement basic clustering
        """
        pass


# =============================================================================
# Part 9: Vector Statistics
# =============================================================================
class VectorStats:
    """
    Compute statistics about a collection of vectors.

    TODO: Implement statistical analysis methods.
    """

    @staticmethod
    def compute_centroid(vectors: list[Vector]) -> np.ndarray:
        """
        Compute the centroid (mean) of vectors.

        TODO: Implement centroid computation
        """
        pass

    @staticmethod
    def compute_variance(vectors: list[Vector]) -> float:
        """
        Compute the variance of vectors from their centroid.

        TODO: Implement variance computation
        """
        pass

    @staticmethod
    def compute_pairwise_distances(
        vectors: list[Vector], metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    ) -> dict:
        """
        Compute statistics about pairwise distances.

        Returns dict with min, max, mean, std of distances.

        TODO: Implement pairwise distance statistics
        """
        pass

    @staticmethod
    def find_outliers(vectors: list[Vector], threshold: float = 2.0) -> list[str]:
        """
        Find vectors that are outliers (far from centroid).

        Uses threshold * std as cutoff.

        TODO: Implement outlier detection
        """
        pass


# =============================================================================
# Part 10: Simple Vector Database
# =============================================================================
class SimpleVectorDB:
    """
    A simple vector database combining storage, indexing, and search.

    This is a learning implementation - not for production use.

    TODO: Implement the complete simple vector database.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        self._model = embedding_model
        self._store = InMemoryVectorStore(metric)
        self._index = FlatIndex(metric)

    def add_text(self, text: str, id: str, metadata: Optional[dict] = None) -> Vector:
        """
        Add a text document to the database.

        Generates embedding and stores the vector.

        TODO: Implement text addition
        """
        pass

    def add_texts(
        self, texts: list[str], ids: list[str], metadatas: Optional[list[dict]] = None
    ) -> list[Vector]:
        """
        Add multiple text documents.

        TODO: Implement batch text addition
        """
        pass

    def search_text(
        self, query: str, k: int = 10, filter_fn: Optional[callable] = None
    ) -> list[SearchResult]:
        """
        Search for similar documents using a text query.

        TODO: Implement text search
        """
        pass

    def search_vector(
        self, query: np.ndarray, k: int = 10, filter_fn: Optional[callable] = None
    ) -> list[SearchResult]:
        """
        Search using a vector query directly.

        TODO: Implement vector search
        """
        pass

    def get(self, vector_id: str) -> Optional[Vector]:
        """Get a vector by ID."""
        # TODO: Implement retrieval
        pass

    def delete(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        # TODO: Implement deletion
        pass

    def count(self) -> int:
        """Get the number of vectors."""
        # TODO: Implement count
        pass


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Create mock embedding model
    model = MockEmbeddingModel(dimension=384, seed=42)

    # Create simple vector database
    db = SimpleVectorDB(model)

    # Add some documents
    docs = [
        ("doc1", "Python is a programming language"),
        ("doc2", "Machine learning uses algorithms"),
        ("doc3", "Vector databases store embeddings"),
    ]

    for doc_id, text in docs:
        db.add_text(text, doc_id, metadata={"source": "example"})

    # Search
    results = db.search_text("What is Python?", k=2)

    print("Search results:")
    for result in results:
        print(f"  {result.vector.id}: {result.score:.4f}")
