"""
Week 2 - Exercise 1: NumPy Embedding Operations
================================================

Complete the TODO sections below and run the tests to verify your solution.

Run this file to check your progress:
    python exercise_1_numpy.py

Run tests:
    python -m pytest tests/test_exercise_1.py -v
"""

import numpy as np


# =============================================================================
# HELPER FUNCTION (provided)
# =============================================================================


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def create_embedding_matrix(vocab_size: int, embedding_dim: int) -> np.ndarray:
    """
    Task 1: Create a random embedding matrix.

    Args:
        vocab_size: Number of words in vocabulary (rows)
        embedding_dim: Dimension of each embedding (columns)

    Returns:
        A numpy array of shape (vocab_size, embedding_dim) with random values
        drawn from a standard normal distribution.

    Example:
        >>> matrix = create_embedding_matrix(1000, 384)
        >>> matrix.shape
        (1000, 384)
    """
    # TODO: Create and return the embedding matrix
    # Hint: Use np.random.randn()
    pass


def get_word_embedding(embeddings: np.ndarray, word_index: int) -> np.ndarray:
    """
    Task 2: Get the embedding vector for a specific word.

    Args:
        embeddings: The embedding matrix of shape (vocab_size, embedding_dim)
        word_index: The index of the word to retrieve

    Returns:
        A 1D numpy array containing the embedding for the word

    Example:
        >>> emb = get_word_embedding(matrix, 42)
        >>> emb.shape
        (384,)
    """
    # TODO: Return the embedding at the given index
    pass


def calculate_similarity(embeddings: np.ndarray, word_a: int, word_b: int) -> float:
    """
    Task 3: Calculate cosine similarity between two word embeddings.

    Args:
        embeddings: The embedding matrix
        word_a: Index of first word
        word_b: Index of second word

    Returns:
        Cosine similarity score between -1 and 1
    """
    # TODO: Get both embeddings and return their cosine similarity
    # Hint: Use the cosine_similarity() helper function provided above
    pass


def find_most_similar(
    embeddings: np.ndarray, target_word: int, top_k: int = 5
) -> np.ndarray:
    """
    Task 4: Find the k most similar words to the target word.

    Args:
        embeddings: The embedding matrix
        target_word: Index of the word to find similar words for
        top_k: Number of similar words to return

    Returns:
        Array of indices of the top_k most similar words (excluding the target itself)

    Example:
        >>> similar = find_most_similar(matrix, 100, top_k=5)
        >>> len(similar)
        5
        >>> 100 not in similar  # Should not include itself
        True
    """
    # TODO: Implement this function
    # Steps:
    # 1. Get the target embedding
    # 2. Calculate similarity with ALL other embeddings
    # 3. Sort by similarity (descending)
    # 4. Return top_k indices, excluding the target word itself
    #
    # Hint: np.argsort() sorts in ascending order
    pass


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Task 5: Implement the softmax function.

    The softmax function converts a vector of scores into probabilities.
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    For numerical stability, subtract the max value before exponentiating:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Args:
        x: Input array of scores

    Returns:
        Array of probabilities that sum to 1

    Example:
        >>> probs = softmax(np.array([2.0, 1.0, 0.1]))
        >>> np.isclose(probs.sum(), 1.0)
        True
    """
    # TODO: Implement softmax with numerical stability
    pass


# =============================================================================
# QUICK CHECK - Run this file directly to test your solutions
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 2 - Exercise 1: NumPy Embedding Operations")
    print("=" * 60)

    all_passed = True

    # Task 1
    print("\n📝 Task 1: Create embedding matrix...")
    try:
        matrix = create_embedding_matrix(1000, 384)
        if matrix is not None and matrix.shape == (1000, 384):
            print("   ✅ PASSED - Shape is correct (1000, 384)")
        else:
            print(
                f"   ❌ FAILED - Expected shape (1000, 384), got {matrix.shape if matrix is not None else None}"
            )
            all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Task 2
    print("\n📝 Task 2: Get word embedding...")
    try:
        if matrix is not None:
            emb = get_word_embedding(matrix, 42)
            if emb is not None and emb.shape == (384,):
                print("   ✅ PASSED - Got embedding for word 42")
            else:
                print(
                    f"   ❌ FAILED - Expected shape (384,), got {emb.shape if emb is not None else None}"
                )
                all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Task 3
    print("\n📝 Task 3: Calculate similarity...")
    try:
        if matrix is not None:
            sim = calculate_similarity(matrix, 10, 20)
            if sim is not None and -1 <= sim <= 1:
                print(f"   ✅ PASSED - Similarity between word 10 and 20: {sim:.4f}")
            else:
                print(f"   ❌ FAILED - Invalid similarity value: {sim}")
                all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Task 4
    print("\n📝 Task 4: Find most similar words...")
    try:
        if matrix is not None:
            similar = find_most_similar(matrix, 100, top_k=5)
            if similar is not None and len(similar) == 5 and 100 not in similar:
                print(f"   ✅ PASSED - Top 5 similar to word 100: {similar}")
            else:
                print(f"   ❌ FAILED - Got: {similar}")
                all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Task 5
    print("\n📝 Task 5: Softmax implementation...")
    try:
        logits = np.array([2.0, 1.0, 0.1])
        probs = softmax(logits)
        if probs is not None and np.isclose(probs.sum(), 1.0):
            print(f"   ✅ PASSED - Softmax([2.0, 1.0, 0.1]) = {probs}")
        else:
            print(f"   ❌ FAILED - Probabilities don't sum to 1: {probs}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TASKS PASSED! Great job!")
        print("\nRun full tests: python -m pytest tests/test_exercise_1.py -v")
    else:
        print("❌ Some tasks need work. Keep trying!")
    print("=" * 60)
