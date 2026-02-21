"""
Week 2 - Exercise 3: Advanced Data Analysis - SOLUTIONS
=======================================================

Reference solutions for the advanced data analysis exercise.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """Task 1: Compute statistics for a batch of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1)

    return {
        "mean": embeddings.mean(axis=0),
        "std": embeddings.std(axis=0),
        "norms": norms,
        "min_norm_idx": int(norms.argmin()),
        "max_norm_idx": int(norms.argmax()),
    }


def batch_cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Task 2: Compute cosine similarity between query and all embeddings."""
    # Normalize query
    query_norm = query / np.linalg.norm(query)

    # Normalize embeddings
    embeddings_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / embeddings_norms

    # Dot product gives cosine similarity
    similarities = embeddings_normalized @ query_norm

    return similarities


def top_k_similar(
    query: np.ndarray, embeddings: np.ndarray, k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Task 3: Find top-k most similar embeddings."""
    similarities = batch_cosine_similarity(query, embeddings)

    # Get indices sorted by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]

    top_indices = sorted_indices[:k]
    top_similarities = similarities[top_indices]

    return top_indices, top_similarities


def create_document_embeddings_df(
    texts: List[str], embeddings: np.ndarray, metadata: Dict[str, List] = None
) -> pd.DataFrame:
    """Task 4: Create DataFrame for document embeddings."""
    df = pd.DataFrame(
        {
            "text": texts,
            "embedding": [emb.tolist() for emb in embeddings],
            "norm": np.linalg.norm(embeddings, axis=1),
        }
    )

    if metadata:
        for key, values in metadata.items():
            df[key] = values

    return df


def analyze_text_by_embedding_clusters(
    df: pd.DataFrame, n_clusters: int = 3
) -> pd.DataFrame:
    """Task 5: Cluster texts by their embeddings."""
    from sklearn.cluster import KMeans

    # Extract embeddings as numpy array
    embeddings = np.array(df["embedding"].tolist())

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    df = df.copy()
    df["cluster"] = clusters

    return df


def aggregate_embeddings_by_group(
    df: pd.DataFrame, group_col: str, embedding_col: str = "embedding"
) -> pd.DataFrame:
    """Task 6: Compute mean embedding for each group."""

    def mean_embedding(embeddings):
        arr = np.array(list(embeddings))
        return arr.mean(axis=0).tolist()

    result = df.groupby(group_col)[embedding_col].apply(mean_embedding).reset_index()
    result.columns = [group_col, "mean_embedding"]

    return result


def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Task 7: Compute pairwise Euclidean distances."""
    # Using broadcasting: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    norms_sq = np.sum(embeddings**2, axis=1)
    dot_products = embeddings @ embeddings.T

    distances_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * dot_products
    distances_sq = np.maximum(distances_sq, 0)  # Numerical stability

    return np.sqrt(distances_sq)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Task 8: L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def sliding_window_embeddings(
    tokens: List[str], embeddings: np.ndarray, window_size: int = 3
) -> List[Tuple[List[str], np.ndarray]]:
    """Task 9: Create sliding window chunks with averaged embeddings."""
    results = []

    for i in range(len(tokens) - window_size + 1):
        window_tokens = tokens[i : i + window_size]
        window_embeddings = embeddings[i : i + window_size]
        mean_embedding = window_embeddings.mean(axis=0)
        results.append((window_tokens, mean_embedding))

    return results


def export_embeddings_for_visualization(
    embeddings: np.ndarray, labels: List[str], output_path: str
) -> None:
    """Task 10: Export embeddings for visualization."""
    with open(output_path, "w") as f:
        # Write header
        dim_cols = "\t".join([f"dim{i}" for i in range(embeddings.shape[1])])
        f.write(f"label\t{dim_cols}\n")

        # Write data
        for label, emb in zip(labels, embeddings):
            emb_str = "\t".join([str(x) for x in emb])
            f.write(f"{label}\t{emb_str}\n")


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 2 Exercise 3: Solutions Demo")
    print("=" * 60)

    np.random.seed(42)
    embeddings = np.random.randn(100, 64)
    texts = [f"document_{i}" for i in range(100)]

    # 1. Statistics
    stats = compute_embedding_statistics(embeddings)
    print(f"\n1. Mean shape: {stats['mean'].shape}")
    print(f"   Min/Max norm indices: {stats['min_norm_idx']}, {stats['max_norm_idx']}")

    # 2. Batch similarity
    query = np.random.randn(64)
    sims = batch_cosine_similarity(query, embeddings)
    print(f"\n2. Similarity range: [{sims.min():.3f}, {sims.max():.3f}]")

    # 3. Top-k
    indices, scores = top_k_similar(query, embeddings, k=5)
    print(f"\n3. Top 5 indices: {indices}")
    print(f"   Top 5 scores: {scores.round(3)}")

    # 4. DataFrame
    df = create_document_embeddings_df(texts[:10], embeddings[:10])
    print(f"\n4. DataFrame columns: {list(df.columns)}")

    # 5. Clustering
    df_clustered = analyze_text_by_embedding_clusters(df, n_clusters=3)
    print(
        f"\n5. Cluster distribution: {df_clustered['cluster'].value_counts().to_dict()}"
    )

    # 6. Aggregation
    df_clustered["group"] = ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"]
    agg = aggregate_embeddings_by_group(df_clustered, "group")
    print(f"\n6. Groups aggregated: {list(agg['group'])}")

    # 7. Pairwise distances
    small_emb = embeddings[:5]
    distances = compute_pairwise_distances(small_emb)
    print(f"\n7. Distance matrix shape: {distances.shape}")

    # 8. Normalize
    normalized = normalize_embeddings(small_emb)
    norms = np.linalg.norm(normalized, axis=1)
    print(f"\n8. Normalized norms: {norms.round(3)}")

    # 9. Sliding window
    tokens = ["hello", "world", "this", "is", "test"]
    token_emb = np.random.randn(5, 8)
    windows = sliding_window_embeddings(tokens, token_emb, window_size=3)
    print(f"\n9. Number of windows: {len(windows)}")

    print("\n" + "=" * 60)
    print("All solutions working!")
