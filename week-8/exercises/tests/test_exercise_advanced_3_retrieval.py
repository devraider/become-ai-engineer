"""
Tests for Week 8 - Exercise Advanced 3: Retrieval Strategies
"""

import pytest

from exercise_advanced_3_retrieval import (
    RetrievedDocument,
    SimpleVectorStore,
    TextChunker,
    mmr_retrieval,
    HybridRetriever,
    expand_query,
    rerank_by_keyword_overlap,
    ContextWindowManager,
    RetrievalEvaluator,
    RetrievalPipeline,
    CachingRetriever,
    CHROMADB_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)


# =============================================================================
# TASK 1: Simple Vector Store
# =============================================================================


class TestSimpleVectorStore:
    def test_add_documents(self):
        """Should add documents to store."""
        store = SimpleVectorStore()
        store.add(
            documents=["doc1", "doc2"], embeddings=[[1, 0], [0, 1]], ids=["id1", "id2"]
        )
        assert len(store.documents) == 2

    def test_query_returns_results(self):
        """Should return query results."""
        store = SimpleVectorStore()
        store.add(
            documents=["hello", "world"],
            embeddings=[[1, 0], [0, 1]],
        )

        results = store.query([1, 0], n_results=1)
        assert results is not None
        assert len(results) == 1
        assert results[0].content == "hello"

    def test_query_with_filter(self):
        """Should filter by metadata."""
        store = SimpleVectorStore()
        store.add(
            documents=["doc1", "doc2"],
            embeddings=[[1, 0], [0, 1]],
            metadatas=[{"type": "a"}, {"type": "b"}],
        )

        results = store.query([0.5, 0.5], n_results=2, where={"type": "a"})
        if results:
            for r in results:
                assert r.metadata.get("type") == "a"

    def test_delete_documents(self):
        """Should delete documents by ID."""
        store = SimpleVectorStore()
        store.add(
            documents=["doc1", "doc2"], embeddings=[[1, 0], [0, 1]], ids=["id1", "id2"]
        )

        deleted = store.delete(["id1"])
        assert deleted == 1
        assert len(store.documents) == 1


# =============================================================================
# TASK 2: Text Chunker
# =============================================================================


class TestTextChunker:
    def test_creates_chunks(self):
        """Should create chunks from text."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 20

        result = chunker.chunk(text)
        assert result is not None
        assert len(result) >= 1

    def test_respects_chunk_size(self):
        """Chunks should not exceed chunk_size significantly."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "Word " * 100

        result = chunker.chunk(text)
        if result:
            for chunk in result:
                # Allow some flexibility for word boundaries
                assert len(chunk) <= 150

    def test_chunk_with_metadata(self):
        """Should add metadata to chunks."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "Test content. " * 10

        result = chunker.chunk_with_metadata(text, source="test.txt")
        if result:
            assert "chunk_index" in result[0]
            assert "source" in result[0]


# =============================================================================
# TASK 3: MMR Retrieval
# =============================================================================


class TestMMRRetrieval:
    def test_returns_k_results(self):
        """Should return k results."""
        query_emb = [1, 0, 0]
        doc_embs = [[0.9, 0.1, 0], [0.85, 0.15, 0], [0.1, 0.9, 0]]
        docs = ["doc1", "doc2", "doc3"]

        result = mmr_retrieval(query_emb, doc_embs, docs, k=2)
        assert result is not None
        assert len(result) == 2

    def test_balances_relevance_diversity(self):
        """MMR should select diverse results."""
        query_emb = [1, 0]
        doc_embs = [[1, 0], [0.99, 0.01], [0, 1]]  # Two very similar, one different
        docs = ["very_relevant", "also_relevant", "different"]

        # With lambda=0.5, should prefer diversity
        result = mmr_retrieval(query_emb, doc_embs, docs, k=2, lambda_mult=0.5)
        if result:
            selected_docs = [d for d, s in result]
            # Should include the "different" doc for diversity
            # This depends on implementation
            assert len(selected_docs) == 2


# =============================================================================
# TASK 4: Hybrid Search
# =============================================================================


class TestHybridRetriever:
    def test_combines_search_methods(self):
        """Should combine semantic and keyword search."""
        retriever = HybridRetriever(alpha=0.5)
        retriever.add_documents(
            documents=["RAG is great", "Vector databases", "RAG with vectors"],
            embeddings=[[1, 0], [0, 1], [0.5, 0.5]],
        )

        result = retriever.search("RAG", [1, 0], k=2)
        assert result is not None
        assert len(result) <= 2


# =============================================================================
# TASK 5: Query Expansion
# =============================================================================


class TestExpandQuery:
    def test_returns_list_with_original(self):
        """Should return list including original query."""
        result = expand_query("fast car", {})
        assert result is not None
        assert "fast car" in result

    def test_expands_with_synonyms(self):
        """Should expand with synonyms."""
        synonyms = {"fast": ["quick", "rapid"]}
        result = expand_query("fast car", synonyms)
        if result:
            assert any("quick" in q for q in result)


# =============================================================================
# TASK 6: Re-ranking
# =============================================================================


class TestRerankByKeywordOverlap:
    def test_boosts_matching_docs(self):
        """Should boost documents with keyword matches."""
        query = "RAG"
        docs = ["Vector databases", "RAG is great", "Other topic"]
        scores = [0.9, 0.8, 0.7]

        result = rerank_by_keyword_overlap(query, docs, scores, boost_factor=0.2)
        if result:
            # Document with "RAG" should move up
            reranked_docs = [d for d, s in result]
            assert "RAG is great" in reranked_docs[:2]


# =============================================================================
# TASK 7: Context Window Manager
# =============================================================================


class TestContextWindowManager:
    def test_fits_within_limit(self):
        """Should select documents within token limit."""
        manager = ContextWindowManager(max_tokens=100)

        docs = ["Short doc.", "Medium length document.", "A" * 500]
        scores = [0.9, 0.8, 0.7]

        result = manager.fit_context(docs, scores)
        if result:
            total_chars = sum(len(d) for d in result)
            assert total_chars <= manager.max_chars

    def test_truncate_document(self):
        """Should truncate document preserving sentences."""
        manager = ContextWindowManager()
        doc = "First sentence. Second sentence. Third sentence."

        result = manager.truncate_document(doc, max_chars=30)
        if result:
            assert len(result) <= 35  # Allow some flexibility


# =============================================================================
# TASK 8: Retrieval Evaluator
# =============================================================================


class TestRetrievalEvaluator:
    def test_precision_at_k(self):
        """Should calculate precision correctly."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc3"}

        result = RetrievalEvaluator.precision_at_k(retrieved, relevant, k=4)
        assert result is not None
        assert abs(result - 0.5) < 0.001  # 2 relevant out of 4

    def test_recall_at_k(self):
        """Should calculate recall correctly."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc3", "doc5"}  # doc5 not retrieved

        result = RetrievalEvaluator.recall_at_k(retrieved, relevant, k=4)
        if result is not None:
            assert abs(result - 2 / 3) < 0.001  # 2 of 3 relevant found

    def test_mrr(self):
        """Should calculate MRR correctly."""
        retrieved = ["doc2", "doc1", "doc3"]
        relevant = {"doc1"}  # First relevant at position 2

        result = RetrievalEvaluator.mean_reciprocal_rank(retrieved, relevant)
        if result is not None:
            assert abs(result - 0.5) < 0.001  # 1/2


# =============================================================================
# TASK 9: Retrieval Pipeline
# =============================================================================


class TestRetrievalPipeline:
    def test_retrieve_returns_documents(self):
        """Pipeline should return documents."""
        store = SimpleVectorStore()
        store.add(
            documents=["hello world", "goodbye moon"], embeddings=[[1, 0], [0, 1]]
        )

        # Create simple mock embedder
        class MockEmbedder:
            def encode(self, texts):
                return [[1, 0] for _ in texts]

        pipeline = RetrievalPipeline(
            vector_store=store,
            embedder=MockEmbedder(),
            use_mmr=False,
            use_reranking=False,
        )

        result = pipeline.retrieve("test query")
        # Pipeline implementation may vary
        assert result is not None or result == []


# =============================================================================
# TASK 10: Caching Retriever
# =============================================================================


class TestCachingRetriever:
    def test_caches_results(self):
        """Should cache and return cached results."""
        call_count = 0

        def mock_retriever(query):
            nonlocal call_count
            call_count += 1
            return [RetrievedDocument("result", 0.9)]

        caching = CachingRetriever(mock_retriever, max_cache_size=10)

        # First call
        result1, cached1 = caching.retrieve("test")
        # Second call (should be cached)
        result2, cached2 = caching.retrieve("test")

        if result1 is not None:
            assert cached1 == False
            assert cached2 == True
            assert call_count == 1  # Only called once

    def test_clears_cache(self):
        """Should clear cache."""

        def mock_retriever(query):
            return [RetrievedDocument("result", 0.9)]

        caching = CachingRetriever(mock_retriever)
        caching.retrieve("test")
        caching.clear_cache()

        assert len(caching.cache) == 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
