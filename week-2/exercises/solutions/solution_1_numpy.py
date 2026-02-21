"""
Week 2 - Exercise 1: SOLUTIONS
==============================

This file contains the complete solutions for Exercise 1.
Try to solve the exercises yourself first before looking at these!
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def create_embedding_matrix(vocab_size: int, embedding_dim: int) -> np.ndarray:
    """
    Task 1: Create a random embedding matrix.

    Solution: Use np.random.randn() which samples from standard normal distribution.
    This is commonly used for weight initialization in neural networks.
    """
    return np.random.randn(vocab_size, embedding_dim)


def get_word_embedding(embeddings: np.ndarray, word_index: int) -> np.ndarray:
    """
    Task 2: Get the embedding vector for a specific word.

    Solution: Simple array indexing. In NumPy, indexing a 2D array with a single
    index returns the entire row.
    """
    return embeddings[word_index]


def calculate_similarity(embeddings: np.ndarray, word_a: int, word_b: int) -> float:
    """
    Task 3: Calculate cosine similarity between two word embeddings.

    Solution: Get both embeddings and use the cosine similarity formula.
    """
    emb_a = embeddings[word_a]
    emb_b = embeddings[word_b]
    return cosine_similarity(emb_a, emb_b)


def find_most_similar(
    embeddings: np.ndarray, target_word: int, top_k: int = 5
) -> np.ndarray:
    """
    Task 4: Find the k most similar words to the target word.

    Solution:
    1. Get target embedding
    2. Calculate similarity with all words
    3. Sort and get top k (excluding target itself)
    """
    target = embeddings[target_word]

    # Calculate similarity with all words
    similarities = np.array([cosine_similarity(target, emb) for emb in embeddings])

    # Get indices sorted by similarity (descending)
    # argsort gives ascending order, so we reverse with [::-1]
    sorted_indices = np.argsort(similarities)[::-1]

    # Filter out the target word and take top k
    result = []
    for idx in sorted_indices:
        if idx != target_word:
            result.append(idx)
        if len(result) == top_k:
            break

    return np.array(result)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Task 5: Implement the softmax function.

    Solution:
    - Subtract max for numerical stability (prevents overflow)
    - Exponentiate
    - Divide by sum

    Mathematical formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


# =============================================================================
# ALTERNATIVE/OPTIMIZED SOLUTIONS
# =============================================================================


def find_most_similar_optimized(
    embeddings: np.ndarray, target_word: int, top_k: int = 5
) -> np.ndarray:
    """
    Optimized version using vectorized operations.

    This is much faster for large embedding matrices because it:
    1. Uses matrix multiplication instead of loop
    2. Pre-normalizes embeddings
    """
    # Normalize all embeddings (so dot product = cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Target embedding (normalized)
    target = normalized[target_word]

    # Compute all similarities at once
    similarities = normalized @ target

    # Set target's similarity to -inf so it won't be selected
    similarities[target_word] = -np.inf

    # Get top k indices
    # argpartition is O(n) vs argsort which is O(n log n)
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]

    # Sort the top k by similarity
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    return top_indices


def softmax_2d(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax that works on 2D arrays (batch processing).

    This is what you'd use in practice for batch predictions.
    """
    # Subtract max along the specified axis
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("Exercise 1 Solutions Demo")
    print("=" * 50)

    # Task 1
    matrix = create_embedding_matrix(1000, 384)
    print(f"1. Created embedding matrix: {matrix.shape}")

    # Task 2
    word_emb = get_word_embedding(matrix, 42)
    print(f"2. Word 42 embedding: {word_emb[:5]}... (truncated)")

    # Task 3
    sim = calculate_similarity(matrix, 10, 20)
    print(f"3. Similarity(word 10, word 20): {sim:.4f}")

    # Task 4
    similar = find_most_similar(matrix, 100, top_k=5)
    print(f"4. Top 5 similar to word 100: {similar}")

    # Task 5
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print(f"5. Softmax([2.0, 1.0, 0.1]) = {probs}")
    print(f"   Sum = {probs.sum():.6f}")

    # Optimized version comparison
    print("\n--- Optimized version ---")
    similar_opt = find_most_similar_optimized(matrix, 100, top_k=5)
    print(f"4. (Optimized) Top 5 similar to word 100: {similar_opt}")
