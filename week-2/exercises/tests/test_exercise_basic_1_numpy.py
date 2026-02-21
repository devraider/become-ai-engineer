"""
Week 2 - Tests for Exercise 1: NumPy Embedding Operations
==========================================================

Run all tests:
    python -m pytest exercises/tests/ -v

Run only exercise 1 tests:
    python -m pytest exercises/tests/test_exercise_1.py -v
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import exercises
sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_basic_1_numpy import (
    create_embedding_matrix,
    get_word_embedding,
    calculate_similarity,
    find_most_similar,
    softmax,
)


class TestCreateEmbeddingMatrix:
    """Tests for Task 1: create_embedding_matrix"""

    def test_shape_correct(self):
        """Test that embedding matrix has correct shape."""
        matrix = create_embedding_matrix(1000, 384)
        assert matrix is not None, "Function returned None - did you forget to return?"
        assert matrix.shape == (
            1000,
            384,
        ), f"Expected shape (1000, 384), got {matrix.shape}"

    def test_contains_random_values(self):
        """Test that matrix contains random values (not all zeros)."""
        matrix = create_embedding_matrix(100, 50)
        assert not np.allclose(
            matrix, 0
        ), "Matrix should contain random values, not zeros"

    def test_different_sizes(self):
        """Test with different sizes."""
        small = create_embedding_matrix(10, 5)
        large = create_embedding_matrix(5000, 768)
        assert small.shape == (10, 5)
        assert large.shape == (5000, 768)

    def test_returns_numpy_array(self):
        """Test that function returns a numpy array."""
        matrix = create_embedding_matrix(10, 5)
        assert isinstance(
            matrix, np.ndarray
        ), f"Expected numpy array, got {type(matrix)}"


class TestGetWordEmbedding:
    """Tests for Task 2: get_word_embedding"""

    def test_shape_correct(self):
        """Test that word embedding has correct shape."""
        matrix = create_embedding_matrix(1000, 384)
        emb = get_word_embedding(matrix, 42)
        assert emb is not None, "Function returned None"
        assert emb.shape == (384,), f"Expected shape (384,), got {emb.shape}"

    def test_correct_values(self):
        """Test that we get the correct row."""
        matrix = np.arange(20).reshape(4, 5).astype(float)
        emb = get_word_embedding(matrix, 2)
        expected = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        assert np.allclose(emb, expected), f"Expected {expected}, got {emb}"

    def test_first_word(self):
        """Test getting the first word (index 0)."""
        matrix = np.arange(15).reshape(3, 5).astype(float)
        emb = get_word_embedding(matrix, 0)
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert np.allclose(emb, expected)

    def test_last_word(self):
        """Test getting the last word."""
        matrix = np.arange(15).reshape(3, 5).astype(float)
        emb = get_word_embedding(matrix, 2)
        expected = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        assert np.allclose(emb, expected)


class TestCalculateSimilarity:
    """Tests for Task 3: calculate_similarity"""

    def test_range_valid(self):
        """Test that similarity is between -1 and 1."""
        matrix = create_embedding_matrix(100, 50)
        sim = calculate_similarity(matrix, 10, 20)
        assert sim is not None, "Function returned None"
        assert -1 <= sim <= 1, f"Similarity should be between -1 and 1, got {sim}"

    def test_same_word_perfect_similarity(self):
        """Test that a word is perfectly similar to itself."""
        matrix = create_embedding_matrix(100, 50)
        sim = calculate_similarity(matrix, 10, 10)
        assert np.isclose(
            sim, 1.0
        ), f"Word should be perfectly similar to itself, got {sim}"

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0."""
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        sim = calculate_similarity(matrix, 0, 1)
        assert np.isclose(
            sim, 0.0
        ), f"Orthogonal vectors should have 0 similarity, got {sim}"

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors is -1."""
        matrix = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
        sim = calculate_similarity(matrix, 0, 1)
        assert np.isclose(
            sim, -1.0
        ), f"Opposite vectors should have -1 similarity, got {sim}"


class TestFindMostSimilar:
    """Tests for Task 4: find_most_similar"""

    def test_returns_correct_count(self):
        """Test that we get the right number of similar words."""
        matrix = create_embedding_matrix(100, 50)
        similar = find_most_similar(matrix, 10, top_k=5)
        assert similar is not None, "Function returned None"
        assert len(similar) == 5, f"Expected 5 similar words, got {len(similar)}"

    def test_excludes_target(self):
        """Test that target word is not in results."""
        matrix = create_embedding_matrix(100, 50)
        similar = find_most_similar(matrix, 10, top_k=5)
        assert 10 not in similar, "Results should not include the target word itself"

    def test_different_top_k(self):
        """Test with different top_k value."""
        matrix = create_embedding_matrix(100, 50)
        similar = find_most_similar(matrix, 50, top_k=3)
        assert len(similar) == 3

    def test_top_k_10(self):
        """Test with top_k=10."""
        matrix = create_embedding_matrix(100, 50)
        similar = find_most_similar(matrix, 25, top_k=10)
        assert len(similar) == 10
        assert 25 not in similar

    def test_returns_array(self):
        """Test that function returns numpy array."""
        matrix = create_embedding_matrix(100, 50)
        similar = find_most_similar(matrix, 10, top_k=5)
        assert isinstance(
            similar, np.ndarray
        ), f"Expected numpy array, got {type(similar)}"


class TestSoftmax:
    """Tests for Task 5: softmax"""

    def test_sums_to_one(self):
        """Test that softmax output sums to 1."""
        x = np.array([2.0, 1.0, 0.1])
        probs = softmax(x)
        assert probs is not None, "Function returned None"
        assert np.isclose(
            probs.sum(), 1.0
        ), f"Softmax should sum to 1, got {probs.sum()}"

    def test_all_positive(self):
        """Test that all probabilities are positive."""
        x = np.array([-1.0, -2.0, -3.0])
        probs = softmax(x)
        assert np.all(probs > 0), "All softmax outputs should be positive"

    def test_preserves_order(self):
        """Test that larger input gives larger probability."""
        x = np.array([3.0, 1.0, 2.0])
        probs = softmax(x)
        assert (
            probs[0] > probs[2] > probs[1]
        ), "Softmax should preserve relative ordering"

    def test_numerical_stability(self):
        """Test that softmax handles large numbers without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        probs = softmax(x)
        assert not np.any(np.isnan(probs)), "Softmax should handle large numbers"
        assert not np.any(np.isinf(probs)), "Softmax should not produce infinity"
        assert np.isclose(probs.sum(), 1.0)

    def test_single_element(self):
        """Test softmax with single element."""
        x = np.array([5.0])
        probs = softmax(x)
        assert np.isclose(probs[0], 1.0), "Single element softmax should be 1.0"

    def test_equal_inputs(self):
        """Test that equal inputs give equal probabilities."""
        x = np.array([1.0, 1.0, 1.0])
        probs = softmax(x)
        assert np.allclose(
            probs, [1 / 3, 1 / 3, 1 / 3]
        ), "Equal inputs should give equal probabilities"
