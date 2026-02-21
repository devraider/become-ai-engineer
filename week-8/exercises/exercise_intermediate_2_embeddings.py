"""
Week 8 - Exercise Intermediate 2: Embeddings & Similarity
==========================================================

Learn to create and use embeddings for semantic search.

Instructions:
- Complete each TODO with your implementation
- Run tests with: pytest tests/test_exercise_intermediate_2_embeddings.py -v
- Check solutions in solutions/solution_intermediate_2_embeddings.py
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import math

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    np = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# =============================================================================
# TASK 1: Cosine Similarity (Pure Python)
# =============================================================================


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Formula: cos(θ) = (A · B) / (||A|| × ||B||)

    Args:
        vec1: First vector
        vec2: Second vector (same length as vec1)

    Returns:
        Cosine similarity score between -1 and 1

    Example:
        >>> v1 = [1, 0, 0]
        >>> v2 = [1, 0, 0]
        >>> cosine_similarity(v1, v2)  # 1.0 (identical)

        >>> v3 = [0, 1, 0]
        >>> cosine_similarity(v1, v3)  # 0.0 (orthogonal)
    """
    # TODO: Implement this function
    # 1. Calculate dot product
    # 2. Calculate magnitudes
    # 3. Return cosine similarity
    pass


# =============================================================================
# TASK 2: Euclidean Distance
# =============================================================================


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Formula: d = sqrt(sum((a_i - b_i)^2))

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance (0 = identical, larger = more different)

    Example:
        >>> v1 = [0, 0]
        >>> v2 = [3, 4]
        >>> euclidean_distance(v1, v2)  # 5.0
    """
    # TODO: Implement this function
    pass


# =============================================================================
# TASK 3: Simple TF-IDF Embeddings
# =============================================================================


class TFIDFEmbedder:
    """Simple TF-IDF based embeddings (no external dependencies)."""

    def __init__(self):
        """Initialize the embedder."""
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False

    def fit(self, documents: List[str]) -> "TFIDFEmbedder":
        """
        Fit the embedder on a corpus of documents.

        Args:
            documents: List of documents to learn vocabulary from

        Returns:
            self (for chaining)

        Example:
            >>> embedder = TFIDFEmbedder()
            >>> embedder.fit(["hello world", "hello there"])
        """
        # TODO: Implement this method
        # 1. Build vocabulary from all documents
        # 2. Calculate IDF for each term
        # 3. Set self.fitted = True
        pass

    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transform documents to TF-IDF vectors.

        Args:
            documents: List of documents to transform

        Returns:
            List of TF-IDF vectors

        Example:
            >>> embedder = TFIDFEmbedder().fit(["hello world", "hello there"])
            >>> vectors = embedder.transform(["hello world"])
        """
        # TODO: Implement this method
        # 1. For each document:
        #    - Calculate TF for each term
        #    - Multiply by IDF
        #    - Create vector
        pass

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Fit and transform in one step."""
        return self.fit(documents).transform(documents)


# =============================================================================
# TASK 4: Embedding Cache
# =============================================================================


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""

    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the cache.

        Args:
            cache_file: Optional path to persist cache (JSON format)
        """
        self.cache: Dict[str, List[float]] = {}
        self.cache_file = cache_file

        if cache_file and os.path.exists(cache_file):
            self._load_cache()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None if not found
        """
        # TODO: Implement this method
        pass

    def put(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding.

        Args:
            text: Text key
            embedding: Embedding vector
        """
        # TODO: Implement this method
        pass

    def _load_cache(self) -> None:
        """Load cache from file."""
        # TODO: Implement this method
        pass

    def save(self) -> None:
        """Save cache to file."""
        # TODO: Implement this method
        pass


# =============================================================================
# TASK 5: Similarity Search
# =============================================================================


def find_most_similar(
    query_embedding: List[float],
    document_embeddings: List[List[float]],
    documents: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Find the most similar documents to a query.

    Args:
        query_embedding: Query vector
        document_embeddings: List of document vectors
        documents: List of document texts (same order as embeddings)
        top_k: Number of results to return

    Returns:
        List of (document, similarity_score) tuples, sorted by similarity

    Example:
        >>> query = [1.0, 0.0, 0.0]
        >>> docs_emb = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
        >>> docs = ["doc1", "doc2", "doc3"]
        >>> results = find_most_similar(query, docs_emb, docs, top_k=2)
        >>> # Returns [("doc1", 1.0), ("doc3", 0.707...)]
    """
    # TODO: Implement this function
    # 1. Calculate similarity between query and each document
    # 2. Sort by similarity
    # 3. Return top_k results
    pass


# =============================================================================
# TASK 6: Batch Embedding with Progress
# =============================================================================


def embed_documents_batch(
    embedder, documents: List[str], batch_size: int = 32, show_progress: bool = True
) -> List[List[float]]:
    """
    Embed documents in batches for efficiency.

    Args:
        embedder: An embedder with encode() method
        documents: List of documents to embed
        batch_size: Number of documents per batch
        show_progress: Whether to print progress

    Returns:
        List of embeddings

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> embeddings = embed_documents_batch(model, docs, batch_size=32)
    """
    # TODO: Implement this function
    # 1. Split documents into batches
    # 2. Embed each batch
    # 3. Combine results
    # 4. Print progress if requested
    pass


# =============================================================================
# TASK 7: Semantic Clustering
# =============================================================================


def cluster_by_similarity(
    embeddings: List[List[float]], documents: List[str], threshold: float = 0.7
) -> List[List[str]]:
    """
    Cluster documents by semantic similarity.

    Simple greedy clustering: documents are added to existing cluster
    if similarity to centroid exceeds threshold.

    Args:
        embeddings: Document embeddings
        documents: Document texts
        threshold: Similarity threshold for clustering

    Returns:
        List of clusters (each cluster is a list of documents)

    Example:
        >>> embs = [[1,0], [0.9,0.1], [0,1], [0.1,0.9]]
        >>> docs = ["a", "b", "c", "d"]
        >>> clusters = cluster_by_similarity(embs, docs, 0.8)
        >>> # Might return [["a", "b"], ["c", "d"]]
    """
    # TODO: Implement this function
    # 1. Start with first document as first cluster
    # 2. For each document, find closest cluster centroid
    # 3. If similarity > threshold, add to cluster
    # 4. Otherwise, create new cluster
    pass


# =============================================================================
# TASK 8: Embedding Dimensionality Reduction (Simple PCA)
# =============================================================================


def reduce_dimensions(
    embeddings: List[List[float]], target_dim: int = 50
) -> List[List[float]]:
    """
    Reduce embedding dimensionality using simple truncation.

    Note: Real PCA is more sophisticated - this is a simplified version.

    Args:
        embeddings: Original embeddings
        target_dim: Target number of dimensions

    Returns:
        Reduced embeddings

    Example:
        >>> embs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        >>> reduced = reduce_dimensions(embs, target_dim=3)
        >>> # Returns [[1, 2, 3], [6, 7, 8]]
    """
    # TODO: Implement this function
    # Truncate each embedding to target_dim dimensions
    pass


# =============================================================================
# TASK 9: Embedding Normalization
# =============================================================================


def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalize embedding to unit length.

    After normalization, cosine similarity equals dot product.

    Args:
        embedding: Raw embedding vector

    Returns:
        Normalized embedding (unit length)

    Example:
        >>> emb = [3, 4]
        >>> norm = normalize_embedding(emb)
        >>> # Returns [0.6, 0.8]
    """
    # TODO: Implement this function
    # Divide each element by the vector magnitude
    pass


def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """Normalize multiple embeddings."""
    # TODO: Implement this function
    pass


# =============================================================================
# TASK 10: Embedding Comparison Report
# =============================================================================


def embedding_comparison_report(
    embeddings: List[List[float]], documents: List[str], top_pairs: int = 5
) -> Dict[str, Any]:
    """
    Generate a report comparing all document embeddings.

    Args:
        embeddings: Document embeddings
        documents: Document texts
        top_pairs: Number of most/least similar pairs to report

    Returns:
        Report dictionary with:
        - total_documents: Number of documents
        - embedding_dim: Embedding dimension
        - most_similar_pairs: List of (doc1, doc2, similarity)
        - least_similar_pairs: List of (doc1, doc2, similarity)
        - avg_similarity: Average pairwise similarity

    Example:
        >>> embs = [[1,0], [0.8,0.2], [0,1]]
        >>> docs = ["a", "b", "c"]
        >>> report = embedding_comparison_report(embs, docs, top_pairs=2)
        >>> print(report['most_similar_pairs'][0])
    """
    # TODO: Implement this function
    # 1. Calculate all pairwise similarities
    # 2. Find most/least similar pairs
    # 3. Calculate average similarity
    # 4. Return report dict
    pass


# =============================================================================
# MAIN - Test your implementations
# =============================================================================

if __name__ == "__main__":
    print("Week 8 - Exercise Intermediate 2: Embeddings & Similarity")
    print("=" * 60)

    # Test cosine similarity
    print("\n1. Cosine Similarity:")
    v1 = [1, 0, 0]
    v2 = [1, 0, 0]
    v3 = [0, 1, 0]
    print(f"Same vector: {cosine_similarity(v1, v2)}")
    print(f"Orthogonal: {cosine_similarity(v1, v3)}")

    # Test TF-IDF
    print("\n2. TF-IDF Embeddings:")
    corpus = ["the quick brown fox", "the lazy dog", "the quick dog"]
    tfidf = TFIDFEmbedder()
    vectors = tfidf.fit_transform(corpus)
    if vectors:
        print(f"Vector dimensions: {len(vectors[0])}")

    # Test similarity search
    print("\n3. Similarity Search:")
    if vectors:
        query_vec = vectors[0]  # "the quick brown fox"
        results = find_most_similar(query_vec, vectors, corpus, top_k=2)
        if results:
            print(f"Most similar to '{corpus[0]}':")
            for doc, score in results:
                print(f"  {doc}: {score:.3f}")
