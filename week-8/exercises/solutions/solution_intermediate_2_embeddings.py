"""
Week 8 - Exercise Intermediate 2: Embeddings & Similarity - SOLUTIONS
=====================================================================
"""

import os
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    np = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# =============================================================================
# TASK 1: Cosine Similarity
# =============================================================================


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Magnitudes
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


# =============================================================================
# TASK 2: Euclidean Distance
# =============================================================================


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    squared_diff = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
    return math.sqrt(squared_diff)


# =============================================================================
# TASK 3: Simple TF-IDF Embeddings
# =============================================================================


class TFIDFEmbedder:
    """Simple TF-IDF based embeddings."""

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def fit(self, documents: List[str]) -> "TFIDFEmbedder":
        """Fit the embedder on a corpus of documents."""
        # Build vocabulary
        all_terms = set()
        doc_term_counts = []

        for doc in documents:
            terms = set(self._tokenize(doc))
            all_terms.update(terms)
            doc_term_counts.append(terms)

        # Create vocabulary mapping
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}

        # Calculate IDF
        n_docs = len(documents)
        for term in self.vocabulary:
            doc_freq = sum(1 for doc_terms in doc_term_counts if term in doc_terms)
            self.idf[term] = math.log(n_docs / (doc_freq + 1)) + 1

        self.fitted = True
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents to TF-IDF vectors."""
        if not self.fitted:
            raise ValueError("Embedder not fitted. Call fit() first.")

        vectors = []
        vocab_size = len(self.vocabulary)

        for doc in documents:
            vector = [0.0] * vocab_size
            tokens = self._tokenize(doc)

            # Calculate term frequencies
            term_counts = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            # Build TF-IDF vector
            for term, count in term_counts.items():
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    tf = count / len(tokens) if tokens else 0
                    vector[idx] = tf * self.idf.get(term, 1)

            vectors.append(vector)

        return vectors

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Fit and transform in one step."""
        return self.fit(documents).transform(documents)


# =============================================================================
# TASK 4: Embedding Cache
# =============================================================================


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""

    def __init__(self, cache_file: Optional[str] = None):
        self.cache: Dict[str, List[float]] = {}
        self.cache_file = cache_file

        if cache_file and os.path.exists(cache_file):
            self._load_cache()

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        return self.cache.get(text)

    def put(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        self.cache[text] = embedding

    def _load_cache(self) -> None:
        """Load cache from file."""
        if self.cache_file:
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache = {}

    def save(self) -> None:
        """Save cache to file."""
        if self.cache_file:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)


# =============================================================================
# TASK 5: Similarity Search
# =============================================================================


def find_most_similar(
    query_embedding: List[float],
    document_embeddings: List[List[float]],
    documents: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Find the most similar documents to a query."""
    similarities = []

    for doc, emb in zip(documents, document_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((doc, sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


# =============================================================================
# TASK 6: Batch Embedding with Progress
# =============================================================================


def embed_documents_batch(
    embedder, documents: List[str], batch_size: int = 32, show_progress: bool = True
) -> List[List[float]]:
    """Embed documents in batches for efficiency."""
    all_embeddings = []
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_num = i // batch_size + 1

        if show_progress:
            print(f"Processing batch {batch_num}/{total_batches}")

        # Embed batch
        if hasattr(embedder, "encode"):
            embeddings = embedder.encode(batch)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
        else:
            embeddings = [[0.0] * 384 for _ in batch]  # Fallback

        all_embeddings.extend(embeddings)

    return all_embeddings


# =============================================================================
# TASK 7: Semantic Clustering
# =============================================================================


def cluster_by_similarity(
    embeddings: List[List[float]], documents: List[str], threshold: float = 0.7
) -> List[List[str]]:
    """Cluster documents by semantic similarity."""
    if not embeddings:
        return []

    clusters: List[List[str]] = []
    cluster_centroids: List[List[float]] = []

    for doc, emb in zip(documents, embeddings):
        best_cluster_idx = -1
        best_similarity = -1

        # Find best matching cluster
        for i, centroid in enumerate(cluster_centroids):
            sim = cosine_similarity(emb, centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_cluster_idx = i

        if best_similarity >= threshold:
            # Add to existing cluster
            clusters[best_cluster_idx].append(doc)
            # Update centroid (simple averaging)
            n = len(clusters[best_cluster_idx])
            new_centroid = [
                (c * (n - 1) + e) / n
                for c, e in zip(cluster_centroids[best_cluster_idx], emb)
            ]
            cluster_centroids[best_cluster_idx] = new_centroid
        else:
            # Create new cluster
            clusters.append([doc])
            cluster_centroids.append(emb[:])

    return clusters


# =============================================================================
# TASK 8: Embedding Dimensionality Reduction
# =============================================================================


def reduce_dimensions(
    embeddings: List[List[float]], target_dim: int = 50
) -> List[List[float]]:
    """Reduce embedding dimensionality using simple truncation."""
    return [emb[:target_dim] for emb in embeddings]


# =============================================================================
# TASK 9: Embedding Normalization
# =============================================================================


def normalize_embedding(embedding: List[float]) -> List[float]:
    """Normalize embedding to unit length."""
    magnitude = math.sqrt(sum(x**2 for x in embedding))

    if magnitude == 0:
        return embedding[:]

    return [x / magnitude for x in embedding]


def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """Normalize multiple embeddings."""
    return [normalize_embedding(emb) for emb in embeddings]


# =============================================================================
# TASK 10: Embedding Comparison Report
# =============================================================================


def embedding_comparison_report(
    embeddings: List[List[float]], documents: List[str], top_pairs: int = 5
) -> Dict[str, Any]:
    """Generate a report comparing all document embeddings."""
    n = len(documents)

    # Calculate all pairwise similarities
    pairs = []
    total_sim = 0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            pairs.append((documents[i], documents[j], sim))
            total_sim += sim
            count += 1

    # Sort by similarity
    pairs.sort(key=lambda x: x[2], reverse=True)

    avg_sim = total_sim / count if count > 0 else 0

    return {
        "total_documents": n,
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "most_similar_pairs": pairs[:top_pairs],
        "least_similar_pairs": (
            pairs[-top_pairs:][::-1] if len(pairs) >= top_pairs else pairs[::-1]
        ),
        "avg_similarity": avg_sim,
        "total_pairs": count,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Week 8 - Exercise Intermediate 2: Embeddings - SOLUTIONS")
    print("=" * 60)

    # Test cosine similarity
    print("\n1. Cosine Similarity:")
    print(f"Same: {cosine_similarity([1,0,0], [1,0,0])}")
    print(f"Orthogonal: {cosine_similarity([1,0,0], [0,1,0])}")

    # Test TF-IDF
    print("\n2. TF-IDF:")
    corpus = ["the quick brown fox", "the lazy dog", "the quick dog"]
    tfidf = TFIDFEmbedder()
    vectors = tfidf.fit_transform(corpus)
    print(f"Vocabulary size: {len(tfidf.vocabulary)}")
    print(f"Vector dim: {len(vectors[0])}")

    # Test similarity search
    print("\n3. Similarity Search:")
    results = find_most_similar(vectors[0], vectors, corpus, top_k=3)
    for doc, score in results:
        print(f"  {doc}: {score:.3f}")

    # Test normalization
    print("\n4. Normalization:")
    norm = normalize_embedding([3, 4])
    print(f"[3, 4] normalized: {norm}")
    mag = math.sqrt(sum(x**2 for x in norm))
    print(f"Magnitude: {mag:.3f}")
