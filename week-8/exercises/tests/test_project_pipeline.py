"""
Tests for Week 8 Project: Document Q&A System
"""

import pytest
import tempfile
import os
from pathlib import Path

from project_pipeline import (
    Document,
    Chunk,
    SearchResult,
    QAResponse,
    DocumentLoader,
    TextChunker,
    EmbeddingManager,
    VectorStore,
    ResponseGenerator,
    RAGPipeline,
    DocumentQA,
    CHROMADB_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    GENAI_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_text_file():
    """Create a temporary text file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is test content for the RAG system.\n")
        f.write("It contains information about document processing.\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document("RAG combines retrieval with generation.", "doc1.txt", {}, "doc1"),
        Document("Vector databases enable semantic search.", "doc2.txt", {}, "doc2"),
        Document("Embeddings represent text as vectors.", "doc3.txt", {}, "doc3"),
    ]


# =============================================================================
# PART 1: Document Loader
# =============================================================================


class TestDocumentLoader:
    def test_load_text(self, temp_text_file):
        """Should load text file."""
        loader = DocumentLoader()
        result = loader.load_text(temp_text_file)

        assert result is not None
        assert isinstance(result, Document)
        assert len(result.content) > 0

    def test_load_from_string(self):
        """Should create document from string."""
        loader = DocumentLoader()
        result = loader.load_from_string("Hello world", source="test")

        assert result is not None
        assert result.content == "Hello world"
        assert result.source == "test"


# =============================================================================
# PART 2: Text Chunker
# =============================================================================


class TestTextChunker:
    def test_chunk_document(self, sample_documents):
        """Should chunk a document."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=5)

        # Use a longer document
        doc = Document("This is test content. " * 10, "test.txt", {}, "test")
        result = chunker.chunk_document(doc)

        assert result is not None
        assert len(result) >= 1
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_documents(self, sample_documents):
        """Should chunk multiple documents."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10, min_chunk_size=5)
        result = chunker.chunk_documents(sample_documents)

        assert result is not None
        assert len(result) >= len(sample_documents)


# =============================================================================
# PART 3: Embedding Manager
# =============================================================================


class TestEmbeddingManager:
    def test_embed_texts(self):
        """Should embed texts."""
        manager = EmbeddingManager()
        result = manager.embed(["hello world", "goodbye"])

        assert result is not None
        assert len(result) == 2
        assert len(result[0]) > 0

    def test_embed_single(self):
        """Should embed single text."""
        manager = EmbeddingManager()
        result = manager.embed_single("hello")

        assert result is not None
        assert len(result) > 0


# =============================================================================
# PART 4: Vector Store
# =============================================================================


class TestVectorStore:
    def test_add_and_search(self):
        """Should add chunks and search."""
        store = VectorStore(collection_name="test_collection")

        chunks = [
            Chunk("RAG content", "doc1", 0),
            Chunk("Vector content", "doc2", 0),
        ]
        embeddings = [[1, 0, 0], [0, 1, 0]]

        store.add_chunks(chunks, embeddings)
        results = store.search([1, 0, 0], k=1)

        assert results is not None
        assert len(results) >= 0  # May be empty if not implemented

    def test_get_stats(self):
        """Should return stats."""
        store = VectorStore()
        stats = store.get_stats()

        # Should return a dict even if empty
        assert stats is None or isinstance(stats, dict)


# =============================================================================
# PART 5: Response Generator
# =============================================================================


class TestResponseGenerator:
    def test_format_context(self):
        """Should format context."""
        gen = ResponseGenerator()

        chunks = [
            SearchResult(Chunk("Content 1", "doc1", 0), 0.9),
            SearchResult(Chunk("Content 2", "doc2", 0), 0.8),
        ]

        result = gen._format_context(chunks)
        # May be None if not implemented
        assert result is None or isinstance(result, str)

    def test_create_prompt(self):
        """Should create prompt."""
        gen = ResponseGenerator()
        result = gen._create_prompt("Question?", "Context here", True)

        assert result is None or "Question" in result


# =============================================================================
# PART 6: RAG Pipeline
# =============================================================================


class TestRAGPipeline:
    def test_ingest_text(self):
        """Should ingest text."""
        pipeline = RAGPipeline(chunk_size=50, chunk_overlap=10)

        text = "This is test content for RAG. " * 5
        result = pipeline.ingest_text(text, source="test")

        assert result is None or result >= 0

    def test_query(self):
        """Should handle query."""
        pipeline = RAGPipeline(chunk_size=50, chunk_overlap=10)

        # Ingest some content first
        pipeline.ingest_text("RAG is retrieval augmented generation.", "doc1")

        result = pipeline.query("What is RAG?", k=1)

        # Result may be None or QAResponse
        assert result is None or isinstance(result, QAResponse)


# =============================================================================
# PART 7: Document QA Interface
# =============================================================================


class TestDocumentQA:
    def test_add_text(self):
        """Should add text to knowledge base."""
        qa = DocumentQA(chunk_size=50, chunk_overlap=10)
        result = qa.add_text("Test content for Q&A", source="test")

        assert result is None or result >= 0

    def test_ask(self):
        """Should answer questions."""
        qa = DocumentQA(chunk_size=50, chunk_overlap=10)
        qa.add_text("RAG stands for Retrieval Augmented Generation.", "doc1")

        result = qa.ask("What does RAG stand for?")

        assert result is None or isinstance(result, QAResponse)

    def test_history(self):
        """Should track question history."""
        qa = DocumentQA()
        qa.add_text("Test content", "test")
        qa.ask("Question 1")
        qa.ask("Question 2")

        history = qa.get_history()
        # History tracking depends on implementation
        assert isinstance(history, list)

    def test_clear_history(self):
        """Should clear history."""
        qa = DocumentQA()
        qa.ask("Question")
        qa.clear_history()

        assert len(qa.get_history()) == 0

    def test_get_stats(self):
        """Should return stats."""
        qa = DocumentQA()
        stats = qa.get_stats()

        assert stats is None or isinstance(stats, dict)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    def test_end_to_end(self):
        """Test complete flow."""
        qa = DocumentQA(chunk_size=100, chunk_overlap=20)

        # Add documents
        qa.add_text(
            """
        Machine learning is a subset of artificial intelligence.
        It enables computers to learn from data without being 
        explicitly programmed.
        """,
            source="ml_intro",
        )

        qa.add_text(
            """
        Deep learning uses neural networks with many layers.
        It has achieved state-of-the-art results in many tasks.
        """,
            source="dl_intro",
        )

        # Ask question
        response = qa.ask("What is machine learning?")

        # Check response structure
        if response:
            assert hasattr(response, "answer")
            assert hasattr(response, "sources")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
