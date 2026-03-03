"""
Week 11 - Exercise 1 (Basic): Embeddings and Similarity
=======================================================

Learn to work with transformer embeddings for semantic similarity tasks.

Topics:
- Mean pooling implementation
- Sentence similarity computation
- Semantic search
- Attention analysis
"""

from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.util import find_spec

# Check for optional dependencies
TORCH_AVAILABLE = find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = find_spec("transformers") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

if TRANSFORMERS_AVAILABLE:
    from transformers import AutoModel, AutoTokenizer


# =============================================================================
# TASK 1: Mean Pooling
# =============================================================================
class MeanPooler:
    """
    Extract embeddings using mean pooling over token embeddings.

    Mean pooling averages all token embeddings, weighted by attention mask,
    to create a single sentence embedding.

    TODO:
    1. Implement __init__ to load model and tokenizer
    2. Implement _mean_pooling to average token embeddings
    3. Implement encode to get sentence embeddings

    Example:
        pooler = MeanPooler("bert-base-uncased")
        embedding = pooler.encode("Hello world")
        print(embedding.shape)  # (768,)
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Load model and tokenizer."""
        # TODO: Load tokenizer with AutoTokenizer.from_pretrained
        # TODO: Load model with AutoModel.from_pretrained
        # TODO: Set model to eval mode
        self.tokenizer = None
        self.model = None
        raise NotImplementedError("Implement model loading")

    def _mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to token embeddings.

        Args:
            model_output: Output from transformer model
            attention_mask: Attention mask from tokenizer

        Returns:
            Mean pooled embeddings

        Hint:
        - Get token embeddings from model_output[0] (last_hidden_state)
        - Expand attention mask to match embedding dimensions
        - Sum embeddings * mask, then divide by sum of mask
        """
        # TODO: Get token embeddings (last_hidden_state)
        # TODO: Expand attention mask for broadcasting
        # TODO: Apply mask and compute mean
        raise NotImplementedError("Implement mean pooling")

    def encode(self, text: str) -> "torch.Tensor":
        """
        Encode text to embedding using mean pooling.

        Args:
            text: Input text to encode

        Returns:
            Sentence embedding tensor
        """
        # TODO: Tokenize text
        # TODO: Get model output (with torch.no_grad())
        # TODO: Apply mean pooling
        # TODO: Return squeezed embedding
        raise NotImplementedError("Implement encoding")


# =============================================================================
# TASK 2: CLS Token Extractor
# =============================================================================
class CLSExtractor:
    """
    Extract embeddings using the [CLS] token.

    The [CLS] token is the first token and is designed to capture
    sentence-level information in BERT-style models.

    TODO:
    1. Implement __init__ to load model and tokenizer
    2. Implement encode to extract [CLS] embedding

    Example:
        extractor = CLSExtractor("bert-base-uncased")
        embedding = extractor.encode("Hello world")
        print(embedding.shape)  # (768,)
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Load model and tokenizer."""
        # TODO: Load tokenizer
        # TODO: Load model
        # TODO: Set to eval mode
        self.tokenizer = None
        self.model = None
        raise NotImplementedError("Implement model loading")

    def encode(self, text: str) -> "torch.Tensor":
        """
        Extract [CLS] token embedding.

        Args:
            text: Input text

        Returns:
            CLS embedding tensor

        Hint:
        - The CLS token is at index 0 of the sequence
        - Use model_output.last_hidden_state[:, 0, :]
        """
        # TODO: Tokenize text
        # TODO: Get model output
        # TODO: Extract CLS token (first token)
        # TODO: Return squeezed embedding
        raise NotImplementedError("Implement CLS extraction")


# =============================================================================
# TASK 3: Cosine Similarity Calculator
# =============================================================================
class SimilarityCalculator:
    """
    Calculate cosine similarity between embeddings.

    TODO:
    1. Implement cosine_similarity for two vectors
    2. Implement pairwise_similarity for a batch
    3. Implement similarity_matrix for all pairs

    Example:
        calc = SimilarityCalculator()
        sim = calc.cosine_similarity(emb1, emb2)
        print(sim)  # 0.85
    """

    def cosine_similarity(self, emb1: "torch.Tensor", emb2: "torch.Tensor") -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity score (-1 to 1)

        Formula: cos_sim = (a · b) / (||a|| * ||b||)
        """
        # TODO: Normalize embeddings
        # TODO: Compute dot product
        # TODO: Return as float
        raise NotImplementedError("Implement cosine similarity")

    def pairwise_similarity(
        self, query: "torch.Tensor", candidates: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Calculate similarity between query and multiple candidates.

        Args:
            query: Query embedding (D,)
            candidates: Candidate embeddings (N, D)

        Returns:
            Similarity scores (N,)
        """
        # TODO: Normalize query and candidates
        # TODO: Compute matrix multiplication
        # TODO: Return similarities
        raise NotImplementedError("Implement pairwise similarity")

    def similarity_matrix(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        """
        Calculate pairwise similarity matrix.

        Args:
            embeddings: Embeddings (N, D)

        Returns:
            Similarity matrix (N, N)
        """
        # TODO: Normalize all embeddings
        # TODO: Compute embeddings @ embeddings.T
        raise NotImplementedError("Implement similarity matrix")


# =============================================================================
# TASK 4: Semantic Search Engine
# =============================================================================
@dataclass
class SearchResult:
    """A search result with text, score, and index."""

    text: str
    score: float
    index: int


class SemanticSearchEngine:
    """
    Semantic search using embedding similarity.

    TODO:
    1. Implement __init__ to set up embedder
    2. Implement add_documents to index documents
    3. Implement search to find similar documents
    4. Implement batch_search for multiple queries

    Example:
        engine = SemanticSearchEngine()
        engine.add_documents(["AI is amazing", "Dogs are cute"])
        results = engine.search("Machine learning", top_k=1)
        print(results[0].text)  # "AI is amazing"
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize search engine with embedder."""
        # TODO: Create MeanPooler instance
        # TODO: Initialize empty lists for documents and embeddings
        self.embedder = None
        self.documents: list[str] = []
        self.embeddings: Optional["torch.Tensor"] = None
        raise NotImplementedError("Implement initialization")

    def add_documents(self, documents: list[str]) -> None:
        """
        Add documents to the search index.

        Args:
            documents: List of documents to add
        """
        # TODO: Store documents
        # TODO: Encode all documents
        # TODO: Stack embeddings into tensor
        raise NotImplementedError("Implement document indexing")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # TODO: Encode query
        # TODO: Calculate similarities with all documents
        # TODO: Get top-k indices
        # TODO: Return SearchResult objects
        raise NotImplementedError("Implement search")

    def batch_search(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[SearchResult]]:
        """
        Search for multiple queries at once.

        Args:
            queries: List of search queries
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        # TODO: Encode all queries
        # TODO: Calculate similarity matrix
        # TODO: Get top-k for each query
        raise NotImplementedError("Implement batch search")


# =============================================================================
# TASK 5: Attention Analyzer
# =============================================================================
class AttentionAnalyzer:
    """
    Analyze attention patterns from transformer models.

    TODO:
    1. Implement __init__ to load model with attention output
    2. Implement get_attention_weights to extract attention
    3. Implement get_token_importance based on attention
    4. Implement visualize_attention to create heatmap data

    Example:
        analyzer = AttentionAnalyzer()
        weights = analyzer.get_attention_weights("The cat sat")
        importance = analyzer.get_token_importance("The cat sat")
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Load model configured to output attention weights."""
        # TODO: Load tokenizer
        # TODO: Load model with output_attentions=True
        self.tokenizer = None
        self.model = None
        raise NotImplementedError("Implement model loading")

    def get_attention_weights(self, text: str) -> "torch.Tensor":
        """
        Extract attention weights for input text.

        Args:
            text: Input text

        Returns:
            Attention weights tensor (layers, heads, seq_len, seq_len)

        Hint:
        - Use model output.attentions
        - Stack all layer attentions
        """
        # TODO: Tokenize text
        # TODO: Get model output with attention
        # TODO: Stack attention from all layers
        # TODO: Squeeze batch dimension
        raise NotImplementedError("Implement attention extraction")

    def get_token_importance(self, text: str, layer: int = -1) -> dict[str, float]:
        """
        Calculate token importance based on attention.

        Args:
            text: Input text
            layer: Which layer to use (-1 for last)

        Returns:
            Dictionary mapping tokens to importance scores

        Hint:
        - Sum attention received by each token across all heads
        - Normalize to sum to 1
        """
        # TODO: Get attention weights
        # TODO: Select layer
        # TODO: Average across heads
        # TODO: Sum attention received by each token
        # TODO: Map to token strings
        raise NotImplementedError("Implement token importance")

    def visualize_attention(self, text: str, layer: int = -1, head: int = 0) -> dict:
        """
        Get attention data for visualization.

        Args:
            text: Input text
            layer: Layer index
            head: Head index

        Returns:
            Dict with 'tokens' and 'attention_matrix'
        """
        # TODO: Tokenize to get tokens
        # TODO: Get attention weights
        # TODO: Extract specific layer and head
        # TODO: Return visualization data
        raise NotImplementedError("Implement attention visualization")


# =============================================================================
# TASK 6: Sentence Transformer Wrapper
# =============================================================================
class SentenceTransformerWrapper:
    """
    Wrapper for Sentence Transformers models.

    Sentence Transformers are specifically trained for semantic similarity.

    TODO:
    1. Implement __init__ to load sentence-transformers model
    2. Implement encode for single/batch encoding
    3. Implement similarity for sentence pairs

    Example:
        st = SentenceTransformerWrapper()
        emb = st.encode("Hello world")
        sim = st.similarity("Hello", "Hi there")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load Sentence Transformer model."""
        # TODO: Import and load SentenceTransformer
        # Hint: from sentence_transformers import SentenceTransformer
        self.model = None
        raise NotImplementedError("Implement SentenceTransformer loading")

    def encode(self, texts: str | list[str], normalize: bool = True) -> "torch.Tensor":
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize embeddings

        Returns:
            Embeddings tensor
        """
        # TODO: Use model.encode()
        # TODO: Convert to tensor if needed
        # TODO: Optionally normalize
        raise NotImplementedError("Implement encoding")

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score
        """
        # TODO: Encode both texts
        # TODO: Calculate cosine similarity
        raise NotImplementedError("Implement similarity")


# =============================================================================
# TASK 7: Embedding Cache
# =============================================================================
class EmbeddingCache:
    """
    Cache embeddings to avoid recomputation.

    TODO:
    1. Implement __init__ with embedder and cache storage
    2. Implement get_or_compute to check cache first
    3. Implement batch_encode with caching
    4. Implement save/load for persistence

    Example:
        cache = EmbeddingCache(embedder)
        emb = cache.get_or_compute("Hello")  # Computes and caches
        emb = cache.get_or_compute("Hello")  # Returns cached
    """

    def __init__(self, embedder):
        """Initialize cache with an embedder."""
        # TODO: Store embedder
        # TODO: Initialize empty cache dict
        self.embedder = embedder
        self.cache: dict[str, "torch.Tensor"] = {}

    def get_or_compute(self, text: str) -> "torch.Tensor":
        """
        Get embedding from cache or compute it.

        Args:
            text: Text to encode

        Returns:
            Embedding tensor
        """
        # TODO: Check if text in cache
        # TODO: If not, compute and store
        # TODO: Return embedding
        raise NotImplementedError("Implement cached encoding")

    def batch_encode(self, texts: list[str]) -> "torch.Tensor":
        """
        Encode batch with caching.

        Args:
            texts: List of texts

        Returns:
            Stacked embeddings
        """
        # TODO: Separate cached and uncached texts
        # TODO: Compute only uncached
        # TODO: Update cache
        # TODO: Return all embeddings in order
        raise NotImplementedError("Implement batch caching")

    def save(self, path: str) -> None:
        """Save cache to file."""
        # TODO: Use torch.save for cache dict
        raise NotImplementedError("Implement cache saving")

    def load(self, path: str) -> None:
        """Load cache from file."""
        # TODO: Use torch.load and update cache
        raise NotImplementedError("Implement cache loading")

    @property
    def size(self) -> int:
        """Number of cached embeddings."""
        return len(self.cache)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


# =============================================================================
# TASK 8: Embedding Dimensionality Reducer
# =============================================================================
class DimensionalityReducer:
    """
    Reduce embedding dimensions for visualization or efficiency.

    TODO:
    1. Implement PCA reduction
    2. Implement random projection
    3. Implement truncation (keep first N dims)

    Example:
        reducer = DimensionalityReducer(target_dim=128)
        reduced = reducer.pca_reduce(embeddings)
    """

    def __init__(self, target_dim: int = 128):
        """Set target dimensionality."""
        self.target_dim = target_dim
        self.pca_components = None

    def fit_pca(self, embeddings: "torch.Tensor") -> None:
        """
        Fit PCA on embeddings.

        Args:
            embeddings: Training embeddings (N, D)
        """
        # TODO: Center embeddings
        # TODO: Compute SVD
        # TODO: Store top components
        raise NotImplementedError("Implement PCA fitting")

    def pca_reduce(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        """
        Reduce dimensions using fitted PCA.

        Args:
            embeddings: Embeddings to reduce

        Returns:
            Reduced embeddings
        """
        # TODO: Check PCA is fitted
        # TODO: Center and project
        raise NotImplementedError("Implement PCA reduction")

    def random_projection(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        """
        Reduce using random projection (Johnson-Lindenstrauss).

        Args:
            embeddings: Embeddings to reduce

        Returns:
            Reduced embeddings
        """
        # TODO: Create random projection matrix
        # TODO: Apply projection
        # TODO: Scale appropriately
        raise NotImplementedError("Implement random projection")

    def truncate(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        """
        Simply truncate to first N dimensions.

        Args:
            embeddings: Embeddings to truncate

        Returns:
            Truncated embeddings
        """
        # TODO: Slice to target_dim
        raise NotImplementedError("Implement truncation")


# =============================================================================
# TASK 9: Embedding Pooling Strategies
# =============================================================================
class PoolingStrategy(ABC):
    """Abstract base class for pooling strategies."""

    @abstractmethod
    def pool(
        self, token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """Pool token embeddings into sentence embedding."""
        pass


class MeanPoolingStrategy(PoolingStrategy):
    """Mean pooling strategy."""

    def pool(
        self, token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Apply mean pooling.

        TODO: Implement weighted mean using attention mask
        """
        raise NotImplementedError("Implement mean pooling")


class MaxPoolingStrategy(PoolingStrategy):
    """Max pooling strategy."""

    def pool(
        self, token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Apply max pooling.

        TODO: Take maximum value for each dimension across tokens
        Hint: Set padded tokens to very negative value before max
        """
        raise NotImplementedError("Implement max pooling")


class WeightedMeanPoolingStrategy(PoolingStrategy):
    """Weighted mean pooling with position weights."""

    def __init__(self, weight_start: float = 1.0, weight_end: float = 0.5):
        """Set weight range (linearly interpolated by position)."""
        self.weight_start = weight_start
        self.weight_end = weight_end

    def pool(
        self, token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Apply weighted mean pooling with position weights.

        TODO:
        1. Create position weights (linear from start to end)
        2. Combine with attention mask
        3. Apply weighted mean
        """
        raise NotImplementedError("Implement weighted mean pooling")


# =============================================================================
# TASK 10: Multi-Model Embedder
# =============================================================================
class MultiModelEmbedder:
    """
    Combine embeddings from multiple models.

    TODO:
    1. Implement __init__ to load multiple models
    2. Implement encode to get combined embeddings
    3. Implement weighted_encode with model weights

    Example:
        embedder = MultiModelEmbedder(["bert-base", "roberta-base"])
        combined = embedder.encode("Hello")  # Concatenated embeddings
    """

    def __init__(self, model_names: list[str]):
        """Load multiple embedding models."""
        # TODO: Create MeanPooler for each model
        self.embedders: list = []
        raise NotImplementedError("Implement multi-model loading")

    def encode(self, text: str, method: str = "concat") -> "torch.Tensor":
        """
        Encode using all models.

        Args:
            text: Input text
            method: 'concat' or 'mean' to combine embeddings

        Returns:
            Combined embedding
        """
        # TODO: Get embedding from each model
        # TODO: Combine based on method
        raise NotImplementedError("Implement multi-model encoding")

    def weighted_encode(self, text: str, weights: list[float]) -> "torch.Tensor":
        """
        Encode with weighted combination.

        Args:
            text: Input text
            weights: Weight for each model

        Returns:
            Weighted average embedding
        """
        # TODO: Get embeddings
        # TODO: Apply weights and sum
        raise NotImplementedError("Implement weighted encoding")


if __name__ == "__main__":
    print("Week 11 - Exercise 1: Embeddings and Similarity")
    print("=" * 50)
    print("\nThis exercise covers:")
    print("1. Mean pooling for sentence embeddings")
    print("2. CLS token extraction")
    print("3. Cosine similarity calculation")
    print("4. Semantic search engine")
    print("5. Attention analysis")
    print("6. Sentence Transformers")
    print("7. Embedding caching")
    print("8. Dimensionality reduction")
    print("9. Pooling strategies")
    print("10. Multi-model embeddings")
    print("\nImplement each class following the TODOs!")
