"""
Tests for Week 8 - Exercise Intermediate 2: Embeddings & Similarity
"""

import pytest
import math

from exercise_intermediate_2_embeddings import (
    cosine_similarity,
    euclidean_distance,
    TFIDFEmbedder,
    EmbeddingCache,
    find_most_similar,
    embed_documents_batch,
    cluster_by_similarity,
    reduce_dimensions,
    normalize_embedding,
    normalize_embeddings,
    embedding_comparison_report,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)


# =============================================================================
# TASK 1: Cosine Similarity
# =============================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v1 = [1, 0, 0]
        result = cosine_similarity(v1, v1)
        assert result is not None
        assert abs(result - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        result = cosine_similarity(v1, v2)
        assert abs(result - 0.0) < 0.001

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        v1 = [1, 0, 0]
        v2 = [-1, 0, 0]
        result = cosine_similarity(v1, v2)
        assert abs(result - (-1.0)) < 0.001


# =============================================================================
# TASK 2: Euclidean Distance
# =============================================================================


class TestEuclideanDistance:
    def test_identical_vectors(self):
        """Identical vectors should have distance 0."""
        v = [1, 2, 3]
        result = euclidean_distance(v, v)
        assert result is not None
        assert abs(result) < 0.001

    def test_known_distance(self):
        """Test with known values (3-4-5 triangle)."""
        v1 = [0, 0]
        v2 = [3, 4]
        result = euclidean_distance(v1, v2)
        assert abs(result - 5.0) < 0.001


# =============================================================================
# TASK 3: TF-IDF Embedder
# =============================================================================


class TestTFIDFEmbedder:
    def test_fit_creates_vocabulary(self):
        """Fit should create vocabulary."""
        embedder = TFIDFEmbedder()
        embedder.fit(["hello world", "hello there"])
        assert embedder.fitted == True
        assert len(embedder.vocabulary) > 0

    def test_transform_returns_vectors(self):
        """Transform should return vectors."""
        embedder = TFIDFEmbedder()
        vectors = embedder.fit_transform(["hello world", "hello there"])
        assert vectors is not None
        assert len(vectors) == 2
        assert len(vectors[0]) > 0

    def test_similar_docs_have_similar_vectors(self):
        """Similar documents should have similar vectors."""
        embedder = TFIDFEmbedder()
        vectors = embedder.fit_transform(
            ["the quick brown fox", "the quick brown dog", "hello world goodbye"]
        )
        if vectors:
            # First two should be more similar than first and third
            sim_12 = cosine_similarity(vectors[0], vectors[1])
            sim_13 = cosine_similarity(vectors[0], vectors[2])
            if sim_12 is not None and sim_13 is not None:
                assert sim_12 > sim_13


# =============================================================================
# TASK 4: Embedding Cache
# =============================================================================


class TestEmbeddingCache:
    def test_put_and_get(self):
        """Should store and retrieve embeddings."""
        cache = EmbeddingCache()
        cache.put("hello", [1.0, 2.0, 3.0])
        result = cache.get("hello")
        assert result is not None
        assert result == [1.0, 2.0, 3.0]

    def test_miss_returns_none(self):
        """Should return None for uncached text."""
        cache = EmbeddingCache()
        result = cache.get("not_cached")
        assert result is None


# =============================================================================
# TASK 5: Similarity Search
# =============================================================================


class TestFindMostSimilar:
    def test_returns_top_k(self):
        """Should return top k results."""
        query = [1.0, 0.0]
        doc_embs = [[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]]
        docs = ["doc1", "doc2", "doc3"]

        result = find_most_similar(query, doc_embs, docs, top_k=2)
        assert result is not None
        assert len(result) == 2

    def test_returns_sorted_by_similarity(self):
        """Results should be sorted by similarity."""
        query = [1.0, 0.0]
        doc_embs = [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]
        docs = ["doc1", "doc2", "doc3"]

        result = find_most_similar(query, doc_embs, docs, top_k=3)
        if result:
            scores = [score for _, score in result]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# TASK 6: Batch Embedding
# =============================================================================


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed"
)
class TestEmbedDocumentsBatch:
    def test_returns_embeddings(self):
        """Should return embeddings for all documents."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        docs = ["hello world", "how are you", "goodbye"]
        result = embed_documents_batch(model, docs, batch_size=2)

        assert result is not None
        assert len(result) == 3


# =============================================================================
# TASK 7: Semantic Clustering
# =============================================================================


class TestClusterBySimilarity:
    def test_groups_similar_documents(self):
        """Should group similar documents together."""
        embeddings = [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]]
        docs = ["a", "b", "c", "d"]

        result = cluster_by_similarity(embeddings, docs, threshold=0.7)

        if result:
            # Should create 2 clusters
            assert len(result) >= 1
            # Total docs should match
            total = sum(len(c) for c in result)
            assert total == 4


# =============================================================================
# TASK 8: Dimensionality Reduction
# =============================================================================


class TestReduceDimensions:
    def test_reduces_to_target(self):
        """Should reduce to target dimensions."""
        embeddings = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        result = reduce_dimensions(embeddings, target_dim=3)

        assert result is not None
        assert len(result[0]) == 3
        assert result[0] == [1, 2, 3]


# =============================================================================
# TASK 9: Embedding Normalization
# =============================================================================


class TestNormalizeEmbedding:
    def test_normalizes_to_unit_length(self):
        """Should normalize to unit length."""
        emb = [3, 4]
        result = normalize_embedding(emb)

        assert result is not None
        # Check unit length
        magnitude = math.sqrt(sum(x**2 for x in result))
        assert abs(magnitude - 1.0) < 0.001

    def test_known_values(self):
        """Test with known 3-4-5 triangle."""
        result = normalize_embedding([3, 4])
        if result:
            assert abs(result[0] - 0.6) < 0.001
            assert abs(result[1] - 0.8) < 0.001


class TestNormalizeEmbeddings:
    def test_normalizes_all(self):
        """Should normalize all embeddings."""
        embeddings = [[3, 4], [5, 12]]
        result = normalize_embeddings(embeddings)

        if result:
            for emb in result:
                magnitude = math.sqrt(sum(x**2 for x in emb))
                assert abs(magnitude - 1.0) < 0.001


# =============================================================================
# TASK 10: Embedding Comparison Report
# =============================================================================


class TestEmbeddingComparisonReport:
    def test_returns_report_dict(self):
        """Should return a report dictionary."""
        embeddings = [[1, 0], [0.8, 0.2], [0, 1]]
        docs = ["a", "b", "c"]

        result = embedding_comparison_report(embeddings, docs, top_pairs=2)

        assert result is not None
        assert "total_documents" in result
        assert "most_similar_pairs" in result

    def test_report_contents(self):
        """Should include expected fields."""
        embeddings = [[1, 0], [0.5, 0.5], [0, 1]]
        docs = ["a", "b", "c"]

        result = embedding_comparison_report(embeddings, docs, top_pairs=2)

        if result:
            assert result["total_documents"] == 3
            assert result["embedding_dim"] == 2


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
