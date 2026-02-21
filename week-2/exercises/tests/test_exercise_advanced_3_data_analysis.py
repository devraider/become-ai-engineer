"""
Week 2 - Tests for Exercise 3: Advanced Data Analysis
=====================================================

Run tests:
    python -m pytest tests/test_exercise_advanced_3_data_analysis.py -v
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_advanced_3_data_analysis import (
    compute_embedding_statistics,
    batch_cosine_similarity,
    top_k_similar,
    create_document_embeddings_df,
    analyze_text_by_embedding_clusters,
    aggregate_embeddings_by_group,
    compute_pairwise_distances,
    normalize_embeddings,
    sliding_window_embeddings,
)


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 8)


@pytest.fixture
def sample_query():
    """Sample query embedding."""
    np.random.seed(123)
    return np.random.randn(8)


# =============================================================================
# TESTS FOR compute_embedding_statistics
# =============================================================================


class TestComputeEmbeddingStatistics:
    """Tests for Task 1."""

    def test_returns_dict(self, sample_embeddings):
        """Test that function returns a dictionary."""
        result = compute_embedding_statistics(sample_embeddings)
        assert result is not None, "Function returned None"
        assert isinstance(result, dict), "Should return a dict"

    def test_has_required_keys(self, sample_embeddings):
        """Test that all required keys are present."""
        result = compute_embedding_statistics(sample_embeddings)
        if result:
            required_keys = ["mean", "std", "norms", "min_norm_idx", "max_norm_idx"]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

    def test_mean_shape(self, sample_embeddings):
        """Test mean embedding shape."""
        result = compute_embedding_statistics(sample_embeddings)
        if result:
            assert result["mean"].shape == (
                8,
            ), "Mean should have shape (embedding_dim,)"


# =============================================================================
# TESTS FOR batch_cosine_similarity
# =============================================================================


class TestBatchCosineSimilarity:
    """Tests for Task 2."""

    def test_returns_array(self, sample_query, sample_embeddings):
        """Test that function returns numpy array."""
        result = batch_cosine_similarity(sample_query, sample_embeddings)
        assert result is not None, "Function returned None"
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self, sample_query, sample_embeddings):
        """Test output shape."""
        result = batch_cosine_similarity(sample_query, sample_embeddings)
        if result is not None:
            assert result.shape == (10,), "Should return (n_samples,) array"

    def test_values_in_range(self, sample_query, sample_embeddings):
        """Test that similarities are in [-1, 1]."""
        result = batch_cosine_similarity(sample_query, sample_embeddings)
        if result is not None:
            assert np.all(result >= -1.01) and np.all(result <= 1.01)


# =============================================================================
# TESTS FOR top_k_similar
# =============================================================================


class TestTopKSimilar:
    """Tests for Task 3."""

    def test_returns_tuple(self, sample_query, sample_embeddings):
        """Test that function returns a tuple."""
        result = top_k_similar(sample_query, sample_embeddings, k=3)
        assert result is not None, "Function returned None"
        assert len(result) == 2, "Should return (indices, similarities)"

    def test_correct_k(self, sample_query, sample_embeddings):
        """Test that k results are returned."""
        result = top_k_similar(sample_query, sample_embeddings, k=5)
        if result:
            indices, sims = result
            assert len(indices) == 5
            assert len(sims) == 5

    def test_sorted_descending(self, sample_query, sample_embeddings):
        """Test results are sorted by similarity descending."""
        result = top_k_similar(sample_query, sample_embeddings, k=5)
        if result:
            _, sims = result
            assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))


# =============================================================================
# TESTS FOR create_document_embeddings_df
# =============================================================================


class TestCreateDocumentEmbeddingsDf:
    """Tests for Task 4."""

    def test_returns_dataframe(self, sample_embeddings):
        """Test that function returns DataFrame."""
        texts = [f"doc_{i}" for i in range(10)]
        result = create_document_embeddings_df(texts, sample_embeddings)
        assert result is not None, "Function returned None"
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_embeddings):
        """Test required columns are present."""
        texts = [f"doc_{i}" for i in range(10)]
        result = create_document_embeddings_df(texts, sample_embeddings)
        if result is not None:
            assert "text" in result.columns
            assert "embedding" in result.columns
            assert "norm" in result.columns


# =============================================================================
# TESTS FOR aggregate_embeddings_by_group
# =============================================================================


class TestAggregateEmbeddingsByGroup:
    """Tests for Task 6."""

    def test_returns_dataframe(self):
        """Test function returns DataFrame."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "embedding": [[1, 2], [3, 4], [5, 6], [7, 8]],
            }
        )
        result = aggregate_embeddings_by_group(df, "category")
        assert result is not None, "Function returned None"
        assert isinstance(result, pd.DataFrame)

    def test_correct_groups(self):
        """Test correct number of groups."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C"],
                "embedding": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            }
        )
        result = aggregate_embeddings_by_group(df, "category")
        if result is not None:
            assert len(result) == 3


# =============================================================================
# TESTS FOR compute_pairwise_distances
# =============================================================================


class TestComputePairwiseDistances:
    """Tests for Task 7."""

    def test_returns_array(self, sample_embeddings):
        """Test function returns array."""
        result = compute_pairwise_distances(sample_embeddings)
        assert result is not None, "Function returned None"
        assert isinstance(result, np.ndarray)

    def test_symmetric(self, sample_embeddings):
        """Test distance matrix is symmetric."""
        result = compute_pairwise_distances(sample_embeddings)
        if result is not None:
            np.testing.assert_array_almost_equal(result, result.T)

    def test_diagonal_zero(self, sample_embeddings):
        """Test diagonal is zero."""
        result = compute_pairwise_distances(sample_embeddings)
        if result is not None:
            np.testing.assert_array_almost_equal(np.diag(result), np.zeros(10))


# =============================================================================
# TESTS FOR normalize_embeddings
# =============================================================================


class TestNormalizeEmbeddings:
    """Tests for Task 8."""

    def test_returns_array(self, sample_embeddings):
        """Test function returns array."""
        result = normalize_embeddings(sample_embeddings)
        assert result is not None, "Function returned None"
        assert isinstance(result, np.ndarray)

    def test_unit_norms(self, sample_embeddings):
        """Test all vectors have unit norm."""
        result = normalize_embeddings(sample_embeddings)
        if result is not None:
            norms = np.linalg.norm(result, axis=1)
            np.testing.assert_array_almost_equal(norms, np.ones(10))


# =============================================================================
# TESTS FOR sliding_window_embeddings
# =============================================================================


class TestSlidingWindowEmbeddings:
    """Tests for Task 9."""

    def test_returns_list(self):
        """Test function returns list."""
        tokens = ["a", "b", "c", "d", "e"]
        embeddings = np.random.randn(5, 3)
        result = sliding_window_embeddings(tokens, embeddings, window_size=2)
        assert result is not None, "Function returned None"
        assert isinstance(result, list)

    def test_correct_window_count(self):
        """Test correct number of windows."""
        tokens = ["a", "b", "c", "d", "e"]
        embeddings = np.random.randn(5, 3)
        result = sliding_window_embeddings(tokens, embeddings, window_size=2)
        if result:
            # 5 tokens with window 2 = 4 windows
            assert len(result) == 4
