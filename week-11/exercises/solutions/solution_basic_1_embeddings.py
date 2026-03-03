"""
Solutions for Week 11 - Exercise 1 (Basic): Embeddings and Similarity
=====================================================================
"""

from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.util import find_spec

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HAS_SENTENCE_TRANSFORMERS = find_spec("sentence_transformers") is not None

if HAS_SENTENCE_TRANSFORMERS:
    from sentence_transformers import SentenceTransformer


# =============================================================================
# TASK 1: Mean Pooling
# =============================================================================
class MeanPooler:
    """Extract embeddings using mean pooling over token embeddings."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Load model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to token embeddings."""
        token_embeddings = model_output[0]  # last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to embedding using mean pooling."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        return embedding.squeeze()


# =============================================================================
# TASK 2: CLS Token Extractor
# =============================================================================
class CLSExtractor:
    """Extract embeddings using the [CLS] token."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Load model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        """Extract [CLS] token embedding."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze()


# =============================================================================
# TASK 3: Cosine Similarity Calculator
# =============================================================================
class SimilarityCalculator:
    """Calculate cosine similarity between embeddings."""

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings."""
        emb1_norm = F.normalize(emb1.unsqueeze(0), p=2, dim=1)
        emb2_norm = F.normalize(emb2.unsqueeze(0), p=2, dim=1)
        return torch.mm(emb1_norm, emb2_norm.T).item()

    def pairwise_similarity(
        self, query: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """Calculate similarity between query and multiple candidates."""
        query_norm = F.normalize(query.unsqueeze(0), p=2, dim=1)
        candidates_norm = F.normalize(candidates, p=2, dim=1)
        return torch.mm(query_norm, candidates_norm.T).squeeze()

    def similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise similarity matrix."""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        return torch.mm(embeddings_norm, embeddings_norm.T)


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
    """Semantic search using embedding similarity."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize search engine with embedder."""
        self.embedder = MeanPooler(model_name)
        self.documents: list[str] = []
        self.embeddings: Optional[torch.Tensor] = None

    def add_documents(self, documents: list[str]) -> None:
        """Add documents to the search index."""
        self.documents = documents
        embeddings_list = [self.embedder.encode(doc) for doc in documents]
        self.embeddings = torch.stack(embeddings_list)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for similar documents."""
        query_embedding = self.embedder.encode(query)
        calculator = SimilarityCalculator()
        similarities = calculator.pairwise_similarity(query_embedding, self.embeddings)

        top_k = min(top_k, len(self.documents))
        scores, indices = torch.topk(similarities, top_k)

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append(
                SearchResult(text=self.documents[idx], score=score, index=idx)
            )
        return results

    def batch_search(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[SearchResult]]:
        """Search for multiple queries at once."""
        return [self.search(query, top_k) for query in queries]


# =============================================================================
# TASK 5: Attention Analyzer
# =============================================================================
class AttentionAnalyzer:
    """Analyze attention patterns from transformer models."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Load model configured to output attention weights."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()

    def get_attention_weights(self, text: str) -> torch.Tensor:
        """Extract attention weights for input text."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        attentions = torch.stack(outputs.attentions)
        return attentions.squeeze(1)  # Remove batch dim

    def get_token_importance(self, text: str, layer: int = -1) -> dict[str, float]:
        """Calculate token importance based on attention."""
        inputs = self.tokenizer(text, return_tensors="pt")
        attention_weights = self.get_attention_weights(text)

        layer_attention = attention_weights[layer]  # (heads, seq, seq)
        avg_attention = layer_attention.mean(dim=0)  # (seq, seq)
        importance = avg_attention.sum(dim=0)  # Sum attention received
        importance = importance / importance.sum()  # Normalize

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return {token: imp.item() for token, imp in zip(tokens, importance)}

    def visualize_attention(self, text: str, layer: int = -1, head: int = 0) -> dict:
        """Get attention data for visualization."""
        inputs = self.tokenizer(text, return_tensors="pt")
        attention_weights = self.get_attention_weights(text)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attention_matrix = attention_weights[layer, head].tolist()

        return {"tokens": tokens, "attention_matrix": attention_matrix}


# =============================================================================
# TASK 6: Sentence Transformer Wrapper
# =============================================================================
class SentenceTransformerWrapper:
    """Wrapper for Sentence Transformers models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load Sentence Transformer model."""
        if HAS_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(model_name)
        else:
            # Fallback to regular transformer
            self.model = None
            self._fallback = MeanPooler("bert-base-uncased")

    def encode(self, texts: str | list[str], normalize: bool = True) -> torch.Tensor:
        """Encode text(s) to embeddings."""
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings
        else:
            # Fallback
            if isinstance(texts, str):
                texts = [texts]
            embeddings = torch.stack([self._fallback.encode(t) for t in texts])
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


# =============================================================================
# TASK 7: Embedding Cache
# =============================================================================
class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, embedder):
        """Initialize cache with an embedder."""
        self.embedder = embedder
        self.cache: dict[str, torch.Tensor] = {}

    def get_or_compute(self, text: str) -> torch.Tensor:
        """Get embedding from cache or compute it."""
        if text not in self.cache:
            self.cache[text] = self.embedder.encode(text)
        return self.cache[text]

    def batch_encode(self, texts: list[str]) -> torch.Tensor:
        """Encode batch with caching."""
        embeddings = []
        for text in texts:
            embeddings.append(self.get_or_compute(text))
        return torch.stack(embeddings)

    def save(self, path: str) -> None:
        """Save cache to file."""
        torch.save(self.cache, path)

    def load(self, path: str) -> None:
        """Load cache from file."""
        self.cache = torch.load(path)

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
    """Reduce embedding dimensions for visualization or efficiency."""

    def __init__(self, target_dim: int = 128):
        """Set target dimensionality."""
        self.target_dim = target_dim
        self.pca_components = None
        self.mean = None

    def fit_pca(self, embeddings: torch.Tensor) -> None:
        """Fit PCA on embeddings."""
        self.mean = embeddings.mean(dim=0)
        centered = embeddings - self.mean
        _, _, V = torch.svd(centered)
        self.pca_components = V[:, : self.target_dim]

    def pca_reduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reduce dimensions using fitted PCA."""
        if self.pca_components is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        centered = embeddings - self.mean
        return torch.mm(centered, self.pca_components)

    def random_projection(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reduce using random projection (Johnson-Lindenstrauss)."""
        d = embeddings.shape[-1]
        projection = torch.randn(d, self.target_dim) / (self.target_dim**0.5)
        return torch.mm(embeddings, projection)

    def truncate(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Simply truncate to first N dimensions."""
        return embeddings[..., : self.target_dim]


# =============================================================================
# TASK 9: Embedding Pooling Strategies
# =============================================================================
class PoolingStrategy(ABC):
    """Abstract base class for pooling strategies."""

    @abstractmethod
    def pool(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings into sentence embedding."""
        pass


class MeanPoolingStrategy(PoolingStrategy):
    """Mean pooling strategy."""

    def pool(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling."""
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class MaxPoolingStrategy(PoolingStrategy):
    """Max pooling strategy."""

    def pool(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply max pooling."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        token_embeddings = token_embeddings.clone()
        token_embeddings[mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, dim=1)[0]


class WeightedMeanPoolingStrategy(PoolingStrategy):
    """Weighted mean pooling with position weights."""

    def __init__(self, weight_start: float = 1.0, weight_end: float = 0.5):
        """Set weight range (linearly interpolated by position)."""
        self.weight_start = weight_start
        self.weight_end = weight_end

    def pool(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply weighted mean pooling with position weights."""
        batch_size, seq_len, dim = token_embeddings.shape

        # Create position weights
        positions = torch.linspace(0, 1, seq_len)
        weights = self.weight_start + (self.weight_end - self.weight_start) * positions
        weights = weights.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, dim)

        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        weighted = token_embeddings * weights * mask_expanded

        sum_embeddings = torch.sum(weighted, dim=1)
        sum_weights = torch.sum(weights * mask_expanded, dim=1)

        return sum_embeddings / torch.clamp(sum_weights, min=1e-9)


# =============================================================================
# TASK 10: Multi-Model Embedder
# =============================================================================
class MultiModelEmbedder:
    """Combine embeddings from multiple models."""

    def __init__(self, model_names: list[str]):
        """Load multiple embedding models."""
        self.embedders = [MeanPooler(name) for name in model_names]

    def encode(self, text: str, method: str = "concat") -> torch.Tensor:
        """Encode using all models."""
        embeddings = [embedder.encode(text) for embedder in self.embedders]

        if method == "concat":
            return torch.cat(embeddings)
        elif method == "mean":
            stacked = torch.stack(embeddings)
            return stacked.mean(dim=0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def weighted_encode(self, text: str, weights: list[float]) -> torch.Tensor:
        """Encode with weighted combination."""
        if len(weights) != len(self.embedders):
            raise ValueError("Weights must match number of embedders")

        embeddings = [embedder.encode(text) for embedder in self.embedders]
        # Normalize embeddings to same scale
        embeddings = [F.normalize(e.unsqueeze(0), dim=1).squeeze() for e in embeddings]

        weighted_sum = sum(w * e for w, e in zip(weights, embeddings))
        return weighted_sum


if __name__ == "__main__":
    print("Week 11 - Exercise 1 Solutions: Embeddings and Similarity")
    print("=" * 60)

    # Test MeanPooler
    print("\n1. MeanPooler:")
    pooler = MeanPooler()
    emb = pooler.encode("Hello world")
    print(f"   Embedding shape: {emb.shape}")

    # Test SimilarityCalculator
    print("\n2. SimilarityCalculator:")
    calc = SimilarityCalculator()
    emb1 = pooler.encode("Machine learning is great")
    emb2 = pooler.encode("AI is amazing")
    emb3 = pooler.encode("The weather is nice")
    print(f"   ML vs AI: {calc.cosine_similarity(emb1, emb2):.4f}")
    print(f"   ML vs Weather: {calc.cosine_similarity(emb1, emb3):.4f}")

    # Test SemanticSearchEngine
    print("\n3. SemanticSearchEngine:")
    engine = SemanticSearchEngine()
    engine.add_documents(
        [
            "Python is a programming language",
            "Machine learning uses neural networks",
            "The weather is sunny today",
        ]
    )
    results = engine.search("AI programming", top_k=2)
    for r in results:
        print(f"   [{r.score:.4f}] {r.text}")

    print("\n✅ All solutions implemented!")
