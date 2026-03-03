"""
Tests for Week 13 - Exercise 1: Vector Fundamentals

Run with: pytest week-13/exercises/tests/test_exercise_basic_1_vector_fundamentals.py -v
"""

import pytest
import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_basic_1_vector_fundamentals import (
    Vector,
    DistanceMetric,
    SimilarityCalculator,
    SearchResult,
    InMemoryVectorStore,
    EmbeddingModel,
    MockEmbeddingModel,
    VectorIndex,
    FlatIndex,
    BatchProcessor,
    SimilarityMatrix,
    VectorStats,
    SimpleVectorDB,
)


# =============================================================================
# Part 1: Vector Tests
# =============================================================================
class TestVector:
    """Tests for Vector dataclass."""

    def test_vector_creation(self):
        """Test basic vector creation."""
        v = Vector(id="v1", values=[1.0, 2.0, 3.0])
        assert v.id == "v1"
        assert v.values == [1.0, 2.0, 3.0]
        assert v.metadata == {}

    def test_vector_with_metadata(self):
        """Test vector with metadata."""
        v = Vector(id="v1", values=[1.0, 2.0], metadata={"label": "test"})
        assert v.metadata["label"] == "test"

    def test_vector_dimension(self):
        """Test dimension property."""
        v = Vector(id="v1", values=[1.0, 2.0, 3.0, 4.0])
        assert v.dimension == 4

    def test_vector_normalize(self):
        """Test vector normalization."""
        v = Vector(id="v1", values=[3.0, 4.0])
        normalized = v.normalize()

        assert normalized.id == "v1"
        assert len(normalized.values) == 2
        # Normalized vector should have magnitude 1
        magnitude = math.sqrt(sum(x**2 for x in normalized.values))
        assert abs(magnitude - 1.0) < 1e-6

    def test_vector_to_dict(self):
        """Test conversion to dictionary."""
        v = Vector(id="v1", values=[1.0, 2.0], metadata={"key": "value"})
        d = v.to_dict()

        assert d["id"] == "v1"
        assert d["values"] == [1.0, 2.0]
        assert d["metadata"]["key"] == "value"

    def test_vector_from_dict(self):
        """Test creation from dictionary."""
        d = {"id": "v1", "values": [1.0, 2.0], "metadata": {"key": "value"}}
        v = Vector.from_dict(d)

        assert v.id == "v1"
        assert v.values == [1.0, 2.0]
        assert v.metadata["key"] == "value"


# =============================================================================
# Part 2: Similarity Calculator Tests
# =============================================================================
class TestSimilarityCalculator:
    """Tests for SimilarityCalculator."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity for identical vectors."""
        calc = SimilarityCalculator(DistanceMetric.COSINE)
        v1 = [1.0, 2.0, 3.0]

        sim = calc.calculate(v1, v1)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        calc = SimilarityCalculator(DistanceMetric.COSINE)
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]

        sim = calc.calculate(v1, v2)
        assert abs(sim) < 1e-6

    def test_euclidean_distance_identical(self):
        """Test Euclidean distance for identical vectors."""
        calc = SimilarityCalculator(DistanceMetric.EUCLIDEAN)
        v1 = [1.0, 2.0, 3.0]

        dist = calc.calculate(v1, v1)
        assert abs(dist) < 1e-6

    def test_euclidean_distance_known(self):
        """Test Euclidean distance for known values."""
        calc = SimilarityCalculator(DistanceMetric.EUCLIDEAN)
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]

        dist = calc.calculate(v1, v2)
        assert abs(dist - 5.0) < 1e-6

    def test_dot_product(self):
        """Test dot product calculation."""
        calc = SimilarityCalculator(DistanceMetric.DOT_PRODUCT)
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]

        dp = calc.calculate(v1, v2)
        assert abs(dp - 32.0) < 1e-6  # 1*4 + 2*5 + 3*6 = 32

    def test_manhattan_distance(self):
        """Test Manhattan distance."""
        calc = SimilarityCalculator(DistanceMetric.MANHATTAN)
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 6.0, 3.0]

        dist = calc.calculate(v1, v2)
        assert abs(dist - 7.0) < 1e-6  # |1-4| + |2-6| + |3-3| = 7

    def test_similarity_to_distance(self):
        """Test similarity to distance conversion."""
        calc = SimilarityCalculator(DistanceMetric.COSINE)
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]

        # Cosine similarity = 0, so distance should be 1
        dist = calc.similarity_to_distance(calc.calculate(v1, v2))
        assert abs(dist - 1.0) < 1e-6


# =============================================================================
# Part 3: In-Memory Vector Store Tests
# =============================================================================
class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    def test_add_vector(self):
        """Test adding a vector."""
        store = InMemoryVectorStore(dimension=3)
        v = Vector(id="v1", values=[1.0, 2.0, 3.0])

        store.add(v)
        assert store.count() == 1

    def test_get_vector(self):
        """Test getting a vector by ID."""
        store = InMemoryVectorStore(dimension=3)
        v = Vector(id="v1", values=[1.0, 2.0, 3.0], metadata={"key": "value"})
        store.add(v)

        retrieved = store.get("v1")
        assert retrieved is not None
        assert retrieved.id == "v1"
        assert retrieved.metadata["key"] == "value"

    def test_get_nonexistent(self):
        """Test getting a nonexistent vector."""
        store = InMemoryVectorStore(dimension=3)
        assert store.get("nonexistent") is None

    def test_delete_vector(self):
        """Test deleting a vector."""
        store = InMemoryVectorStore(dimension=3)
        v = Vector(id="v1", values=[1.0, 2.0, 3.0])
        store.add(v)

        assert store.delete("v1") is True
        assert store.count() == 0
        assert store.get("v1") is None

    def test_search(self):
        """Test similarity search."""
        store = InMemoryVectorStore(dimension=3)
        store.add(Vector(id="v1", values=[1.0, 0.0, 0.0]))
        store.add(Vector(id="v2", values=[0.0, 1.0, 0.0]))
        store.add(Vector(id="v3", values=[0.9, 0.1, 0.0]))

        results = store.search([1.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        # v1 should be most similar (identical)
        assert results[0].id == "v1"
        assert results[0].score > 0.99

    def test_search_with_filter(self):
        """Test search with metadata filter."""
        store = InMemoryVectorStore(dimension=3)
        store.add(Vector(id="v1", values=[1.0, 0.0, 0.0], metadata={"category": "A"}))
        store.add(Vector(id="v2", values=[0.9, 0.1, 0.0], metadata={"category": "B"}))
        store.add(Vector(id="v3", values=[0.8, 0.2, 0.0], metadata={"category": "A"}))

        results = store.search([1.0, 0.0, 0.0], k=2, filter={"category": "A"})

        assert len(results) == 2
        assert all(r.id in ["v1", "v3"] for r in results)


# =============================================================================
# Part 4: Mock Embedding Model Tests
# =============================================================================
class TestMockEmbeddingModel:
    """Tests for MockEmbeddingModel."""

    def test_embed_returns_correct_dimension(self):
        """Test embedding has correct dimension."""
        model = MockEmbeddingModel(dimension=384)
        embedding = model.embed("test text")

        assert len(embedding) == 384

    def test_embed_deterministic(self):
        """Test same text produces same embedding."""
        model = MockEmbeddingModel(dimension=128)

        emb1 = model.embed("hello world")
        emb2 = model.embed("hello world")

        assert emb1 == emb2

    def test_embed_batch(self):
        """Test batch embedding."""
        model = MockEmbeddingModel(dimension=64)
        texts = ["text1", "text2", "text3"]

        embeddings = model.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 64 for e in embeddings)

    def test_different_texts_different_embeddings(self):
        """Test different texts produce different embeddings."""
        model = MockEmbeddingModel(dimension=64)

        emb1 = model.embed("hello")
        emb2 = model.embed("world")

        assert emb1 != emb2


# =============================================================================
# Part 5: Flat Index Tests
# =============================================================================
class TestFlatIndex:
    """Tests for FlatIndex."""

    def test_add_vectors(self):
        """Test adding vectors to index."""
        index = FlatIndex(dimension=3)
        vectors = [
            Vector(id="v1", values=[1.0, 0.0, 0.0]),
            Vector(id="v2", values=[0.0, 1.0, 0.0]),
        ]

        index.add(vectors)
        assert index.count() == 2

    def test_search(self):
        """Test search on flat index."""
        index = FlatIndex(dimension=3)
        index.add(
            [
                Vector(id="v1", values=[1.0, 0.0, 0.0]),
                Vector(id="v2", values=[0.0, 1.0, 0.0]),
                Vector(id="v3", values=[0.707, 0.707, 0.0]),
            ]
        )

        results = index.search([1.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        assert results[0][0] == "v1"  # Most similar

    def test_remove(self):
        """Test removing vectors from index."""
        index = FlatIndex(dimension=3)
        index.add(
            [
                Vector(id="v1", values=[1.0, 0.0, 0.0]),
                Vector(id="v2", values=[0.0, 1.0, 0.0]),
            ]
        )

        index.remove(["v1"])
        assert index.count() == 1


# =============================================================================
# Part 6: Batch Processor Tests
# =============================================================================
class TestBatchProcessor:
    """Tests for BatchProcessor."""

    def test_process_single_batch(self):
        """Test processing a single batch."""
        processor = BatchProcessor(batch_size=10)
        items = [1, 2, 3, 4, 5]

        results = list(processor.process(items, lambda x: x * 2))
        assert results == [2, 4, 6, 8, 10]

    def test_process_multiple_batches(self):
        """Test processing multiple batches."""
        processor = BatchProcessor(batch_size=3)
        items = list(range(10))

        results = list(processor.process(items, lambda x: x))
        assert results == items

    def test_batch_count(self):
        """Test batch counting."""
        processor = BatchProcessor(batch_size=4)
        items = list(range(10))

        count = 0
        for _ in processor.iterate_batches(items):
            count += 1

        assert count == 3  # 10 items / 4 per batch = 3 batches


# =============================================================================
# Part 7: Vector Statistics Tests
# =============================================================================
class TestVectorStats:
    """Tests for VectorStats."""

    def test_compute_stats(self):
        """Test computing statistics."""
        vectors = [
            Vector(id="v1", values=[1.0, 2.0, 3.0]),
            Vector(id="v2", values=[4.0, 5.0, 6.0]),
            Vector(id="v3", values=[7.0, 8.0, 9.0]),
        ]

        stats = VectorStats.compute(vectors)

        assert stats["count"] == 3
        assert stats["dimension"] == 3

    def test_mean_vector(self):
        """Test mean vector calculation."""
        vectors = [
            Vector(id="v1", values=[1.0, 2.0]),
            Vector(id="v2", values=[3.0, 4.0]),
        ]

        stats = VectorStats.compute(vectors)
        mean = stats["mean"]

        assert abs(mean[0] - 2.0) < 1e-6
        assert abs(mean[1] - 3.0) < 1e-6


# =============================================================================
# Part 8: Similarity Matrix Tests
# =============================================================================
class TestSimilarityMatrix:
    """Tests for SimilarityMatrix."""

    def test_compute_matrix(self):
        """Test computing similarity matrix."""
        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]

        matrix = SimilarityMatrix(vectors, DistanceMetric.COSINE)
        m = matrix.compute()

        # Diagonal should be 1 (self-similarity)
        assert abs(m[0][0] - 1.0) < 1e-6
        assert abs(m[1][1] - 1.0) < 1e-6

        # v1 and v2 are orthogonal
        assert abs(m[0][1]) < 1e-6

    def test_find_most_similar(self):
        """Test finding most similar pairs."""
        vectors = [
            [1.0, 0.0],
            [0.9, 0.1],  # Most similar to v0
            [0.0, 1.0],
        ]

        matrix = SimilarityMatrix(vectors, DistanceMetric.COSINE)
        pairs = matrix.find_most_similar(k=1)

        assert len(pairs) == 1
        assert (pairs[0][0] == 0 and pairs[0][1] == 1) or (
            pairs[0][0] == 1 and pairs[0][1] == 0
        )


# =============================================================================
# Part 9: Simple Vector DB Tests
# =============================================================================
class TestSimpleVectorDB:
    """Tests for SimpleVectorDB."""

    def test_create_collection(self):
        """Test creating a collection."""
        db = SimpleVectorDB()
        db.create_collection("test", dimension=128)

        assert "test" in db.list_collections()

    def test_insert_and_search(self):
        """Test inserting and searching."""
        db = SimpleVectorDB()
        db.create_collection("test", dimension=3)

        db.insert("test", "doc1", [1.0, 0.0, 0.0], {"label": "A"})
        db.insert("test", "doc2", [0.0, 1.0, 0.0], {"label": "B"})

        results = db.search("test", [1.0, 0.0, 0.0], k=1)

        assert len(results) == 1
        assert results[0].id == "doc1"

    def test_delete_from_collection(self):
        """Test deleting from collection."""
        db = SimpleVectorDB()
        db.create_collection("test", dimension=3)
        db.insert("test", "doc1", [1.0, 0.0, 0.0], {})

        db.delete("test", "doc1")

        results = db.search("test", [1.0, 0.0, 0.0], k=1)
        assert len(results) == 0

    def test_delete_collection(self):
        """Test deleting a collection."""
        db = SimpleVectorDB()
        db.create_collection("test", dimension=3)
        db.delete_collection("test")

        assert "test" not in db.list_collections()


# =============================================================================
# Integration Tests
# =============================================================================
class TestIntegration:
    """Integration tests for vector operations."""

    def test_end_to_end_search(self):
        """Test complete search workflow."""
        # Create embedding model and store
        model = MockEmbeddingModel(dimension=64)
        store = InMemoryVectorStore(dimension=64)

        # Add some documents
        docs = ["Python programming", "Machine learning", "Vector databases"]
        for i, doc in enumerate(docs):
            embedding = model.embed(doc)
            vector = Vector(id=f"doc{i}", values=embedding, metadata={"text": doc})
            store.add(vector)

        # Search
        query_embedding = model.embed("Python code")
        results = store.search(query_embedding, k=2)

        assert len(results) == 2
        # Should have valid scores
        assert all(0 <= r.score <= 1 for r in results)

    def test_batch_processing_with_index(self):
        """Test batch processing with index."""
        index = FlatIndex(dimension=3)
        processor = BatchProcessor(batch_size=2)

        # Create vectors in batches
        vectors = [
            Vector(id=f"v{i}", values=[float(i), float(i + 1), float(i + 2)])
            for i in range(5)
        ]

        for batch in processor.iterate_batches(vectors):
            index.add(batch)

        assert index.count() == 5
