"""
Week 2 - Exercise 3: Advanced Data Analysis for AI
===================================================

Combine NumPy and Pandas skills for real-world AI data analysis tasks.
Focus on embedding analysis, batch processing, and data pipeline patterns.

Run this file:
    python exercise_advanced_3_data_analysis.py

Run tests:
    python -m pytest tests/test_exercise_advanced_3_data_analysis.py -v
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Task 1: Compute statistics for a batch of embeddings.

    Real-world use: Analyzing embedding distributions for quality checks,
    detecting outliers, and understanding model behavior.

    Args:
        embeddings: 2D array of shape (n_samples, embedding_dim)

    Returns:
        Dictionary with keys:
        - 'mean': mean embedding vector
        - 'std': standard deviation per dimension
        - 'norms': L2 norm of each embedding
        - 'min_norm_idx': index of embedding with smallest norm
        - 'max_norm_idx': index of embedding with largest norm

    Example:
        >>> emb = np.array([[1, 2], [3, 4], [5, 6]])
        >>> stats = compute_embedding_statistics(emb)
        >>> stats['mean'].shape
        (2,)
    """
    # TODO: Implement
    pass


def batch_cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Task 2: Compute cosine similarity between a query and all embeddings.

    Real-world use: Semantic search, finding similar documents/sentences,
    retrieval-augmented generation (RAG).

    Args:
        query: 1D array of shape (embedding_dim,)
        embeddings: 2D array of shape (n_samples, embedding_dim)

    Returns:
        1D array of similarities (n_samples,)

    Hints:
        - Normalize query and embeddings first
        - Use matrix multiplication for efficiency
        - similarity = (query · embedding) / (||query|| * ||embedding||)
    """
    # TODO: Implement
    pass


def top_k_similar(
    query: np.ndarray, embeddings: np.ndarray, k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 3: Find top-k most similar embeddings to a query.

    Real-world use: RAG systems, recommendation engines, semantic search.

    Args:
        query: 1D array (embedding_dim,)
        embeddings: 2D array (n_samples, embedding_dim)
        k: Number of top results to return

    Returns:
        Tuple of (indices, similarities) for top-k results, sorted descending

    Hints:
        - Use batch_cosine_similarity from Task 2
        - np.argsort() to get sorted indices
    """
    # TODO: Implement
    pass


def create_document_embeddings_df(
    texts: List[str], embeddings: np.ndarray, metadata: Dict[str, List] = None
) -> pd.DataFrame:
    """
    Task 4: Create a structured DataFrame for document embeddings.

    Real-world use: Managing embedding databases, building search indices,
    organizing document collections.

    Args:
        texts: List of text strings
        embeddings: 2D array (n_texts, embedding_dim)
        metadata: Optional dict of additional columns

    Returns:
        DataFrame with columns:
        - 'text': the original text
        - 'embedding': list representation of embedding
        - 'norm': L2 norm of embedding
        - Plus any metadata columns

    Example:
        >>> texts = ["hello", "world"]
        >>> emb = np.random.randn(2, 3)
        >>> df = create_document_embeddings_df(texts, emb)
        >>> 'text' in df.columns and 'embedding' in df.columns
        True
    """
    # TODO: Implement
    pass


def analyze_text_by_embedding_clusters(
    df: pd.DataFrame, n_clusters: int = 3
) -> pd.DataFrame:
    """
    Task 5: Cluster texts by their embeddings and analyze.

    Real-world use: Topic discovery, document organization, finding patterns.

    Args:
        df: DataFrame with 'text' and 'embedding' columns
        n_clusters: Number of clusters

    Returns:
        Original DataFrame with added 'cluster' column

    Hints:
        - Extract embeddings as numpy array
        - Use sklearn.cluster.KMeans
        - Assign cluster labels back to DataFrame
    """
    # TODO: Implement
    pass


def aggregate_embeddings_by_group(
    df: pd.DataFrame, group_col: str, embedding_col: str = "embedding"
) -> pd.DataFrame:
    """
    Task 6: Compute mean embedding for each group.

    Real-world use: Computing class centroids, aggregating user interests,
    building prototype representations.

    Args:
        df: DataFrame with embeddings
        group_col: Column to group by
        embedding_col: Column containing embeddings

    Returns:
        DataFrame with group_col and 'mean_embedding' columns

    Example:
        >>> df = pd.DataFrame({
        ...     'category': ['A', 'A', 'B'],
        ...     'embedding': [[1,2], [3,4], [5,6]]
        ... })
        >>> result = aggregate_embeddings_by_group(df, 'category')
        >>> len(result)
        2
    """
    # TODO: Implement
    pass


def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """
    Task 7: Compute pairwise Euclidean distances between all embeddings.

    Real-world use: Diversity analysis, redundancy detection, clustering prep.

    Args:
        embeddings: 2D array (n_samples, embedding_dim)

    Returns:
        2D symmetric distance matrix (n_samples, n_samples)

    Hints:
        - Use broadcasting: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        - Or use scipy.spatial.distance.cdist
    """
    # TODO: Implement
    pass


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Task 8: L2 normalize embeddings (unit vectors).

    Real-world use: Preparing embeddings for cosine similarity,
    standardizing for ML models.

    Args:
        embeddings: 2D array (n_samples, embedding_dim)

    Returns:
        Normalized embeddings where each row has L2 norm = 1
    """
    # TODO: Implement
    pass


def sliding_window_embeddings(
    tokens: List[str], embeddings: np.ndarray, window_size: int = 3
) -> List[Tuple[List[str], np.ndarray]]:
    """
    Task 9: Create sliding window chunks with averaged embeddings.

    Real-world use: Document chunking for RAG, context window preparation.

    Args:
        tokens: List of token strings
        embeddings: 2D array (n_tokens, embedding_dim)
        window_size: Number of tokens per window

    Returns:
        List of (window_tokens, mean_embedding) tuples
    """
    # TODO: Implement
    pass


def export_embeddings_for_visualization(
    embeddings: np.ndarray, labels: List[str], output_path: str
) -> None:
    """
    Task 10: Export embeddings in format for visualization tools.

    Real-world use: Preparing data for TensorBoard projector,
    UMAP/t-SNE visualization.

    Args:
        embeddings: 2D array (n_samples, embedding_dim)
        labels: List of labels for each embedding
        output_path: Path to save TSV file

    Output format (TSV):
        label\tdim1\tdim2\t...
    """
    # TODO: Implement
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 2 Exercise 3: Advanced Data Analysis")
    print("=" * 60)

    # Sample data
    np.random.seed(42)
    embeddings = np.random.randn(100, 64)
    texts = [f"document_{i}" for i in range(100)]

    # Test your implementations
    print("\n1. Testing compute_embedding_statistics...")
    stats = compute_embedding_statistics(embeddings)
    if stats:
        print(f"   Mean shape: {stats['mean'].shape}")

    print("\n2. Testing batch_cosine_similarity...")
    query = np.random.randn(64)
    sims = batch_cosine_similarity(query, embeddings)
    if sims is not None:
        print(f"   Similarities shape: {sims.shape}")

    print("\n3. Testing top_k_similar...")
    result = top_k_similar(query, embeddings, k=5)
    if result:
        indices, scores = result
        print(f"   Top 5 indices: {indices}")

    print("\nComplete all TODOs and run tests to verify!")
