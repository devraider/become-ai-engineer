"""
Tests for Week 11 - Exercise 1: Embeddings and Similarity
"""

import pytest
from importlib.util import find_spec
from unittest.mock import Mock, patch, MagicMock

# Check for optional dependencies
HAS_TORCH = find_spec("torch") is not None

if HAS_TORCH:
    import torch

# Import exercise classes
from exercise_basic_1_embeddings import (
    MeanPooler,
    CLSExtractor,
    SimilarityCalculator,
    SemanticSearchEngine,
    SearchResult,
    AttentionAnalyzer,
    SentenceTransformerWrapper,
    EmbeddingCache,
    DimensionalityReducer,
    PoolingStrategy,
    MeanPoolingStrategy,
    MaxPoolingStrategy,
    WeightedMeanPoolingStrategy,
    MultiModelEmbedder,
)


class TestMeanPooler:
    """Tests for MeanPooler class."""

    def test_init_creates_model_and_tokenizer(self):
        """Test that __init__ loads model and tokenizer."""
        pooler = MeanPooler("bert-base-uncased")
        assert pooler.model is not None
        assert pooler.tokenizer is not None

    def test_mean_pooling_returns_correct_shape(self):
        """Test mean pooling produces correct embedding shape."""
        pooler = MeanPooler("bert-base-uncased")
        embedding = pooler.encode("Hello world")
        # BERT base has 768 dimensions
        assert embedding.shape == (768,) or embedding.shape[-1] == 768

    def test_encode_handles_long_text(self):
        """Test encoding handles text longer than max length."""
        pooler = MeanPooler("bert-base-uncased")
        long_text = "word " * 1000  # Very long text
        embedding = pooler.encode(long_text)
        assert embedding is not None


class TestCLSExtractor:
    """Tests for CLSExtractor class."""

    def test_init_loads_model(self):
        """Test initialization loads model."""
        extractor = CLSExtractor("bert-base-uncased")
        assert extractor.model is not None

    def test_encode_returns_correct_shape(self):
        """Test CLS extraction returns correct shape."""
        extractor = CLSExtractor("bert-base-uncased")
        embedding = extractor.encode("Test sentence")
        assert embedding.shape[-1] == 768


class TestSimilarityCalculator:
    """Tests for SimilarityCalculator class."""

    def test_cosine_similarity_identical_vectors(self):
        """Test similarity of identical vectors is 1."""
        calc = SimilarityCalculator()
        vec = torch.randn(768)
        sim = calc.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0."""
        calc = SimilarityCalculator()
        vec1 = torch.tensor([1.0, 0.0])
        vec2 = torch.tensor([0.0, 1.0])
        sim = calc.cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.0001

    def test_pairwise_similarity_shape(self):
        """Test pairwise similarity returns correct shape."""
        calc = SimilarityCalculator()
        query = torch.randn(768)
        candidates = torch.randn(5, 768)
        sims = calc.pairwise_similarity(query, candidates)
        assert sims.shape == (5,)

    def test_similarity_matrix_symmetric(self):
        """Test similarity matrix is symmetric."""
        calc = SimilarityCalculator()
        embeddings = torch.randn(4, 768)
        matrix = calc.similarity_matrix(embeddings)
        assert matrix.shape == (4, 4)
        # Check symmetry
        assert torch.allclose(matrix, matrix.T, atol=1e-5)


class TestSemanticSearchEngine:
    """Tests for SemanticSearchEngine class."""

    def test_add_documents_stores_embeddings(self):
        """Test adding documents creates embeddings."""
        engine = SemanticSearchEngine()
        docs = ["Document one", "Document two"]
        engine.add_documents(docs)
        assert len(engine.documents) == 2
        assert engine.embeddings is not None

    def test_search_returns_results(self):
        """Test search returns SearchResult objects."""
        engine = SemanticSearchEngine()
        engine.add_documents(["AI is cool", "Weather is nice"])
        results = engine.search("machine learning", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_search_top_k_respected(self):
        """Test top_k limits results."""
        engine = SemanticSearchEngine()
        engine.add_documents(["Doc 1", "Doc 2", "Doc 3", "Doc 4"])
        results = engine.search("test", top_k=2)
        assert len(results) == 2


class TestAttentionAnalyzer:
    """Tests for AttentionAnalyzer class."""

    def test_get_attention_weights_shape(self):
        """Test attention weights have correct shape."""
        analyzer = AttentionAnalyzer("bert-base-uncased")
        weights = analyzer.get_attention_weights("The cat sat")
        # Should be (layers, heads, seq_len, seq_len)
        assert len(weights.shape) == 4

    def test_get_token_importance_returns_dict(self):
        """Test token importance returns token->score mapping."""
        analyzer = AttentionAnalyzer()
        importance = analyzer.get_token_importance("The cat sat")
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_visualize_attention_returns_data(self):
        """Test visualization data is returned."""
        analyzer = AttentionAnalyzer()
        data = analyzer.visualize_attention("Hello world")
        assert "tokens" in data
        assert "attention_matrix" in data


class TestSentenceTransformerWrapper:
    """Tests for SentenceTransformerWrapper class."""

    def test_encode_single_text(self):
        """Test encoding single text."""
        st = SentenceTransformerWrapper()
        emb = st.encode("Hello world")
        assert emb is not None

    def test_encode_batch(self):
        """Test encoding batch of texts."""
        st = SentenceTransformerWrapper()
        embs = st.encode(["Text 1", "Text 2"])
        assert len(embs) == 2

    def test_similarity_range(self):
        """Test similarity is in valid range."""
        st = SentenceTransformerWrapper()
        sim = st.similarity("Hello", "Hi there")
        assert -1.0 <= sim <= 1.0


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    def test_caching_works(self):
        """Test that caching avoids recomputation."""
        mock_embedder = Mock()
        mock_embedder.encode = Mock(return_value=Mock())

        cache = EmbeddingCache(mock_embedder)
        cache.get_or_compute("test")
        cache.get_or_compute("test")

        # Should only encode once
        assert mock_embedder.encode.call_count == 1

    def test_cache_size(self):
        """Test cache size tracking."""
        mock_embedder = Mock()
        mock_embedder.encode = Mock(return_value=Mock())

        cache = EmbeddingCache(mock_embedder)
        cache.get_or_compute("text1")
        cache.get_or_compute("text2")

        assert cache.size == 2

    def test_clear_cache(self):
        """Test cache clearing."""
        mock_embedder = Mock()
        mock_embedder.encode = Mock(return_value=Mock())

        cache = EmbeddingCache(mock_embedder)
        cache.get_or_compute("text")
        cache.clear()

        assert cache.size == 0


class TestDimensionalityReducer:
    """Tests for DimensionalityReducer class."""

    def test_truncate_reduces_dimensions(self):
        """Test truncation reduces to target dim."""
        reducer = DimensionalityReducer(target_dim=128)
        embeddings = torch.randn(10, 768)
        reduced = reducer.truncate(embeddings)
        assert reduced.shape == (10, 128)

    def test_pca_fit_and_reduce(self):
        """Test PCA fitting and reduction."""
        reducer = DimensionalityReducer(target_dim=64)
        embeddings = torch.randn(100, 768)
        reducer.fit_pca(embeddings)
        reduced = reducer.pca_reduce(embeddings)
        assert reduced.shape == (100, 64)


class TestPoolingStrategies:
    """Tests for pooling strategy classes."""

    def test_mean_pooling_strategy(self):
        """Test MeanPoolingStrategy."""
        strategy = MeanPoolingStrategy()
        tokens = torch.randn(1, 10, 768)
        mask = torch.ones(1, 10)
        pooled = strategy.pool(tokens, mask)
        assert pooled.shape == (1, 768)

    def test_max_pooling_strategy(self):
        """Test MaxPoolingStrategy."""
        strategy = MaxPoolingStrategy()
        tokens = torch.randn(1, 10, 768)
        mask = torch.ones(1, 10)
        pooled = strategy.pool(tokens, mask)
        assert pooled.shape == (1, 768)

    def test_weighted_mean_pooling(self):
        """Test WeightedMeanPoolingStrategy."""
        strategy = WeightedMeanPoolingStrategy(1.0, 0.5)
        tokens = torch.randn(1, 10, 768)
        mask = torch.ones(1, 10)
        pooled = strategy.pool(tokens, mask)
        assert pooled.shape == (1, 768)


class TestMultiModelEmbedder:
    """Tests for MultiModelEmbedder class."""

    def test_init_loads_multiple_models(self):
        """Test initialization loads all models."""
        embedder = MultiModelEmbedder(["bert-base-uncased", "distilbert-base-uncased"])
        assert len(embedder.embedders) == 2

    def test_concat_method(self):
        """Test concatenation combines embeddings."""
        embedder = MultiModelEmbedder(["bert-base-uncased", "distilbert-base-uncased"])
        combined = embedder.encode("test", method="concat")
        # Combined should be larger than single
        assert combined.shape[-1] > 768

    def test_mean_method(self):
        """Test mean combines embeddings."""
        embedder = MultiModelEmbedder(["bert-base-uncased", "distilbert-base-uncased"])
        combined = embedder.encode("test", method="mean")
        # Mean should match single model dimension
        assert combined.shape[-1] == 768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
