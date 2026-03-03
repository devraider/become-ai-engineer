"""
Tests for Week 13 - Exercise 3: Advanced Vector Search

Run with: pytest week-13/exercises/tests/test_exercise_advanced_3_vector_search.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_advanced_3_vector_search import (
    BackendType,
    BackendConfig,
    BackendFactory,
    InMemoryBackend,
    FAISSBackend,
    EmbeddingProvider,
    SentenceTransformerProvider,
    CachedEmbeddingProvider,
    HybridSearchConfig,
    HybridSearchEngine,
    Reranker,
    CrossEncoderReranker,
    MMRReranker,
    QueryContext,
    QueryProcessor,
    QueryExpander,
    HypotheticalDocumentEmbedder,
    QueryPipeline,
    MultiBackendSearchManager,
    AdvancedSearchSystem,
)


# =============================================================================
# Part 1: Backend Config Tests
# =============================================================================
class TestBackendConfig:
    """Tests for BackendConfig."""

    def test_memory_backend_config(self):
        """Test memory backend configuration."""
        config = BackendConfig(backend_type=BackendType.MEMORY, dimension=384)

        assert config.backend_type == BackendType.MEMORY
        assert config.dimension == 384

    def test_faiss_backend_config(self):
        """Test FAISS backend configuration."""
        config = BackendConfig(
            backend_type=BackendType.FAISS,
            dimension=768,
            faiss_index_type="ivf",
            faiss_nlist=100,
        )

        assert config.backend_type == BackendType.FAISS
        assert config.faiss_index_type == "ivf"

    def test_validate_memory_config(self):
        """Test memory config validation."""
        config = BackendConfig(backend_type=BackendType.MEMORY, dimension=128)

        assert config.validate() is True

    def test_validate_pinecone_config_missing_key(self):
        """Test Pinecone config validation with missing API key."""
        config = BackendConfig(
            backend_type=BackendType.PINECONE,
            dimension=384,
            # Missing api_key
        )

        # Should fail validation
        assert config.validate() is False


# =============================================================================
# Part 2: Backend Factory Tests
# =============================================================================
class TestBackendFactory:
    """Tests for BackendFactory."""

    def test_register_backend(self):
        """Test backend registration."""
        BackendFactory.register(BackendType.MEMORY, InMemoryBackend)

        assert BackendType.MEMORY in BackendFactory.available_backends()

    def test_create_memory_backend(self):
        """Test creating memory backend."""
        BackendFactory.register(BackendType.MEMORY, InMemoryBackend)

        config = BackendConfig(backend_type=BackendType.MEMORY, dimension=128)

        backend = BackendFactory.create(config)

        assert backend is not None


# =============================================================================
# Part 3: In-Memory Backend Tests
# =============================================================================
class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    def test_add_vectors(self):
        """Test adding vectors."""
        backend = InMemoryBackend(dimension=3)

        backend.add(
            ids=["v1", "v2"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadata=[{"key": "a"}, {"key": "b"}],
        )

        assert backend.count() == 2

    def test_search(self):
        """Test vector search."""
        backend = InMemoryBackend(dimension=3)
        backend.add(
            ids=["v1", "v2", "v3"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
            metadata=[{}, {}, {}],
        )

        results = backend.search([1.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        assert results[0]["id"] == "v1"  # Most similar

    def test_search_with_filter(self):
        """Test search with metadata filter."""
        backend = InMemoryBackend(dimension=3)
        backend.add(
            ids=["v1", "v2", "v3"],
            vectors=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
            metadata=[{"cat": "A"}, {"cat": "B"}, {"cat": "A"}],
        )

        results = backend.search([1.0, 0.0, 0.0], k=2, filter={"cat": "A"})

        assert len(results) == 2
        assert all(r["metadata"]["cat"] == "A" for r in results)

    def test_delete(self):
        """Test vector deletion."""
        backend = InMemoryBackend(dimension=3)
        backend.add(
            ids=["v1", "v2"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadata=[{}, {}],
        )

        count = backend.delete(["v1"])

        assert count == 1
        assert backend.count() == 1

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        backend = InMemoryBackend(dimension=2)

        # Identical vectors should have similarity 1
        sim = backend._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

        # Orthogonal vectors should have similarity 0
        sim = backend._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6


# =============================================================================
# Part 4: Cached Embedding Provider Tests
# =============================================================================
class TestCachedEmbeddingProvider:
    """Tests for CachedEmbeddingProvider."""

    class MockProvider:
        """Mock embedding provider for testing."""

        def __init__(self, dimension=64):
            self._dimension = dimension
            self.embed_count = 0

        def embed(self, text):
            self.embed_count += 1
            return [hash(text) % 100 / 100.0] * self._dimension

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

        @property
        def dimension(self):
            return self._dimension

    def test_caching(self):
        """Test that caching prevents duplicate embeddings."""
        mock = self.MockProvider()
        cached = CachedEmbeddingProvider(mock, max_cache_size=100)

        # First call should compute
        cached.embed("test")
        assert mock.embed_count == 1

        # Second call should use cache
        cached.embed("test")
        assert mock.embed_count == 1

    def test_different_texts_compute(self):
        """Test different texts are computed."""
        mock = self.MockProvider()
        cached = CachedEmbeddingProvider(mock, max_cache_size=100)

        cached.embed("text1")
        cached.embed("text2")

        assert mock.embed_count == 2

    def test_batch_caching(self):
        """Test batch embedding with caching."""
        mock = self.MockProvider()
        cached = CachedEmbeddingProvider(mock, max_cache_size=100)

        # Pre-cache some texts
        cached.embed("text1")
        cached.embed("text2")

        # Batch with mix of cached and new
        cached.embed_batch(["text1", "text3", "text2"])

        # Only text3 should be newly computed
        assert mock.embed_count == 3  # text1, text2, text3


# =============================================================================
# Part 5: Hybrid Search Tests
# =============================================================================
class TestHybridSearchConfig:
    """Tests for HybridSearchConfig."""

    def test_default_weights(self):
        """Test default weight configuration."""
        config = HybridSearchConfig()

        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.semantic_weight + config.keyword_weight == 1.0

    def test_custom_weights(self):
        """Test custom weight configuration."""
        config = HybridSearchConfig(semantic_weight=0.5, keyword_weight=0.5)

        assert config.semantic_weight == 0.5
        assert config.keyword_weight == 0.5


class TestHybridSearchEngine:
    """Tests for HybridSearchEngine."""

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create mock embedding provider."""

        class MockProvider:
            def embed(self, text):
                return [hash(text) % 100 / 100.0] * 64

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

            @property
            def dimension(self):
                return 64

        return MockProvider()

    def test_keyword_search_basic(self, mock_embedding_provider):
        """Test basic keyword search."""
        backend = InMemoryBackend(dimension=64)
        config = HybridSearchConfig()
        engine = HybridSearchEngine(backend, mock_embedding_provider, config)

        # Add documents
        engine.add_document("doc1", "Python programming language", {})
        engine.add_document("doc2", "Machine learning algorithms", {})

        # Keyword search
        results = engine.keyword_search("Python", k=2)

        # doc1 should be found (contains "Python")
        assert any(r[0] == "doc1" for r in results)


# =============================================================================
# Part 6: Reranker Tests
# =============================================================================
class TestMMRReranker:
    """Tests for MMRReranker."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock embedding provider."""

        class MockProvider:
            def embed(self, text):
                # Simple hash-based embedding for testing
                h = hash(text)
                return [(h >> i) & 1 for i in range(64)]

            @property
            def dimension(self):
                return 64

        return MockProvider()

    def test_mmr_basic(self, mock_provider):
        """Test basic MMR reranking."""
        reranker = MMRReranker(mock_provider, lambda_param=0.5)

        documents = [
            ("doc1", "Python programming"),
            ("doc2", "Python is great"),  # Similar to doc1
            ("doc3", "Machine learning"),  # Different topic
        ]

        results = reranker.rerank("Python coding", documents, top_k=3)

        # Should return reranked results
        assert len(results) == 3

    def test_mmr_diversity(self, mock_provider):
        """Test MMR promotes diversity."""
        reranker = MMRReranker(mock_provider, lambda_param=0.5)

        # Create similar documents
        documents = [
            ("doc1", "Python basics"),
            ("doc2", "Python fundamentals"),
            ("doc3", "Machine learning"),
            ("doc4", "Deep learning"),
        ]

        results = reranker.rerank("Python", documents, top_k=3)

        # MMR should promote diversity, so we shouldn't just get
        # the most similar documents
        ids = [r[0] for r in results]
        assert len(set(ids)) == 3  # All unique


# =============================================================================
# Part 7: Query Pipeline Tests
# =============================================================================
class TestQueryExpander:
    """Tests for QueryExpander."""

    def test_synonym_expansion(self):
        """Test query expansion with synonyms."""
        expander = QueryExpander(
            synonym_map={
                "fast": ["quick", "rapid", "speedy"],
                "big": ["large", "huge", "enormous"],
            }
        )

        context = QueryContext(original_query="fast algorithm")
        result = expander.process(context)

        # Expanded query should contain synonyms
        assert "quick" in result.processed_query or "fast" in result.processed_query

    def test_no_synonyms(self):
        """Test when no synonyms match."""
        expander = QueryExpander(synonym_map={"fast": ["quick"]})

        context = QueryContext(original_query="slow algorithm")
        result = expander.process(context)

        # Should preserve original query
        assert "slow" in result.processed_query


class TestQueryPipeline:
    """Tests for QueryPipeline."""

    def test_pipeline_execution(self):
        """Test pipeline executes processors in order."""

        class CountingProcessor:
            def __init__(self, id):
                self.id = id
                self.order = None

            def process(self, context):
                context.metadata[f"processor_{self.id}"] = True
                return context

        p1 = CountingProcessor("1")
        p2 = CountingProcessor("2")

        pipeline = QueryPipeline()
        pipeline.add(p1).add(p2)

        result = pipeline.process("test query")

        assert result.metadata.get("processor_1") is True
        assert result.metadata.get("processor_2") is True

    def test_empty_pipeline(self):
        """Test empty pipeline returns valid context."""
        pipeline = QueryPipeline()
        result = pipeline.process("test query")

        assert result.original_query == "test query"


# =============================================================================
# Part 8: Multi-Backend Search Manager Tests
# =============================================================================
class TestMultiBackendSearchManager:
    """Tests for MultiBackendSearchManager."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock embedding provider."""

        class MockProvider:
            def embed(self, text):
                return [hash(text) % 100 / 100.0] * 64

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

            @property
            def dimension(self):
                return 64

        return MockProvider()

    def test_add_backend(self, mock_provider):
        """Test adding backends."""
        manager = MultiBackendSearchManager(mock_provider)
        backend = InMemoryBackend(dimension=64)

        manager.add_backend("primary", backend, primary=True)

        assert "primary" in manager._backends
        assert manager._primary == "primary"

    def test_remove_backend(self, mock_provider):
        """Test removing backends."""
        manager = MultiBackendSearchManager(mock_provider)
        backend = InMemoryBackend(dimension=64)

        manager.add_backend("temp", backend)
        result = manager.remove_backend("temp")

        assert result is True
        assert "temp" not in manager._backends

    def test_add_document_to_backends(self, mock_provider):
        """Test adding document across backends."""
        manager = MultiBackendSearchManager(mock_provider)

        backend1 = InMemoryBackend(dimension=64)
        backend2 = InMemoryBackend(dimension=64)

        manager.add_backend("b1", backend1)
        manager.add_backend("b2", backend2)

        results = manager.add_document(
            "doc1", "Test content", {"key": "value"}, backends=["b1", "b2"]
        )

        assert results.get("b1") is True
        assert results.get("b2") is True


# =============================================================================
# Part 9: Advanced Search System Tests
# =============================================================================
class TestAdvancedSearchSystem:
    """Tests for AdvancedSearchSystem."""

    def test_initialization(self):
        """Test system initialization."""
        configs = [BackendConfig(backend_type=BackendType.MEMORY, dimension=64)]

        system = AdvancedSearchSystem(
            backend_configs=configs,
            embedding_model="all-MiniLM-L6-v2",
            enable_hybrid=False,
            enable_cache=True,
        )

        assert system._enable_cache is True

    def test_search_interface(self):
        """Test search method signature."""
        configs = [BackendConfig(backend_type=BackendType.MEMORY, dimension=64)]
        system = AdvancedSearchSystem(backend_configs=configs)

        # Verify callable exists
        assert callable(system.search)
        assert callable(system.add_document)


# =============================================================================
# Integration Tests
# =============================================================================
class TestIntegration:
    """Integration tests for advanced vector search."""

    def test_end_to_end_memory_backend(self):
        """Test complete flow with memory backend."""

        # Create components
        class MockProvider:
            def embed(self, text):
                # Create deterministic embeddings
                h = hash(text) & 0xFFFFFFFF
                return [((h >> i) & 0xFF) / 255.0 for i in range(64)]

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

            @property
            def dimension(self):
                return 64

        provider = MockProvider()
        backend = InMemoryBackend(dimension=64)

        # Add documents
        docs = [
            ("doc1", "Python programming language"),
            ("doc2", "Machine learning algorithms"),
            ("doc3", "Vector database systems"),
        ]

        for doc_id, text in docs:
            embedding = provider.embed(text)
            backend.add([doc_id], [embedding], [{"text": text}])

        # Search
        query_embedding = provider.embed("Python coding")
        results = backend.search(query_embedding, k=2)

        assert len(results) == 2
        # Results should have scores
        assert all("score" in r or "distance" in r for r in results)

    def test_hybrid_search_integration(self):
        """Test hybrid search integration."""

        class MockProvider:
            def embed(self, text):
                return [hash(text) % 100 / 100.0] * 64

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

            @property
            def dimension(self):
                return 64

        provider = MockProvider()
        backend = InMemoryBackend(dimension=64)
        config = HybridSearchConfig(semantic_weight=0.6, keyword_weight=0.4)

        engine = HybridSearchEngine(backend, provider, config)

        # Add documents
        engine.add_document(
            "doc1", "Python programming is fun", {"topic": "programming"}
        )
        engine.add_document("doc2", "Machine learning with Python", {"topic": "ai"})
        engine.add_document("doc3", "Database systems", {"topic": "databases"})

        # Hybrid search
        results = engine.search("Python", k=2)

        # Should get results combining semantic and keyword scores
        assert len(results) <= 2
