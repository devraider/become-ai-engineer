"""
Tests for Week 13 - Project: Semantic Document Search Engine

Run with: pytest week-13/exercises/tests/test_project_pipeline.py -v
"""

import pytest
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from project_pipeline import (
    DocumentType,
    DocumentSource,
    DocumentChunk,
    Document,
    ChunkingStrategy,
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    MarkdownChunker,
    MockEmbeddingProvider,
    SentenceTransformerProvider,
    DocumentStore,
    InMemoryDocumentStore,
    FileDocumentStore,
    VectorIndex,
    ChromaVectorIndex,
    SearchHit,
    SearchResult,
    DocumentProcessor,
    TextCleaner,
    MetadataExtractor,
    IngestionPipeline,
    SemanticSearchEngine,
    SearchEngineConfig,
    SearchEngineFactory,
    BatchIngestor,
)


# =============================================================================
# Document Model Tests
# =============================================================================
class TestDocumentSource:
    """Tests for DocumentSource."""

    def test_from_file(self):
        """Test creating source from file path."""
        source = DocumentSource.from_file("/path/to/doc.txt")

        assert source.source_type == "file"
        assert source.location == "/path/to/doc.txt"

    def test_from_url(self):
        """Test creating source from URL."""
        source = DocumentSource.from_url("https://example.com/doc")

        assert source.source_type == "url"
        assert source.location == "https://example.com/doc"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        source = DocumentSource(source_type="file", location="/path/to/doc.txt")

        d = source.to_dict()

        assert d["source_type"] == "file"
        assert d["location"] == "/path/to/doc.txt"


class TestDocumentChunk:
    """Tests for DocumentChunk."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = DocumentChunk(
            chunk_id="c1",
            document_id="doc1",
            content="Test content",
            chunk_index=0,
            start_char=0,
            end_char=12,
        )

        assert chunk.chunk_id == "c1"
        assert chunk.document_id == "doc1"

    def test_length_property(self):
        """Test length property."""
        chunk = DocumentChunk(
            chunk_id="c1",
            document_id="doc1",
            content="Hello world",
            chunk_index=0,
            start_char=0,
            end_char=11,
        )

        assert chunk.length == 11

    def test_word_count(self):
        """Test word count property."""
        chunk = DocumentChunk(
            chunk_id="c1",
            document_id="doc1",
            content="Hello world this is a test",
            chunk_index=0,
            start_char=0,
            end_char=26,
        )

        assert chunk.word_count == 6


class TestDocument:
    """Tests for Document model."""

    def test_create_document(self):
        """Test creating document with auto-generated ID."""
        doc = Document.create(content="Test content", title="Test Document")

        assert doc.document_id is not None
        assert doc.title == "Test Document"
        assert doc.content == "Test content"

    def test_document_with_source(self):
        """Test document with source."""
        source = DocumentSource.from_file("/test.txt")
        doc = Document.create(content="Test", source=source)

        assert doc.source.source_type == "file"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = Document.create(
            content="Test content", title="Test", metadata={"key": "value"}
        )

        d = doc.to_dict()

        assert "document_id" in d
        assert d["title"] == "Test"
        assert d["metadata"]["key"] == "value"


# =============================================================================
# Chunking Strategy Tests
# =============================================================================
class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_small_document(self):
        """Test document smaller than chunk size."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        doc = Document.create(content="Short content")

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Short content"

    def test_long_document(self):
        """Test document larger than chunk size."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=10, min_chunk_size=20)
        doc = Document.create(content="A" * 150)

        chunks = chunker.chunk(doc)

        assert len(chunks) > 1

    def test_overlap(self):
        """Test chunk overlap."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=20, min_chunk_size=20)
        doc = Document.create(content="A" * 100)

        chunks = chunker.chunk(doc)

        # With overlap, second chunk should start before first ends
        if len(chunks) > 1:
            assert chunks[1].start_char < chunks[0].end_char

    def test_chunk_ids_unique(self):
        """Test all chunks have unique IDs."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        doc = Document.create(content="A" * 200)

        chunks = chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]

        assert len(ids) == len(set(ids))


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_split_by_sentences(self):
        """Test splitting into sentences."""
        chunker = SentenceChunker(max_sentences=2)
        doc = Document.create(
            content="First sentence. Second sentence. Third sentence."
        )

        chunks = chunker.chunk(doc)

        # Should split into chunks of max 2 sentences
        assert len(chunks) >= 1

    def test_respects_sentence_boundaries(self):
        """Test that chunks don't split sentences."""
        chunker = SentenceChunker(max_sentences=1, min_sentences=1)
        doc = Document.create(content="Complete sentence one. Complete sentence two.")

        chunks = chunker.chunk(doc)

        # Each chunk should be a complete sentence
        for chunk in chunks:
            assert chunk.content.strip().endswith(".")


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_chunk_by_headers(self):
        """Test chunking markdown by headers."""
        chunker = MarkdownChunker(max_chunk_size=1000)

        content = """# Section 1
Content for section 1.

## Subsection 1.1
Content for subsection.

# Section 2
Content for section 2.
"""
        doc = Document.create(content=content, doc_type=DocumentType.MARKDOWN)
        chunks = chunker.chunk(doc)

        # Should create chunks based on headers
        assert len(chunks) >= 1

    def test_preserves_header_info(self):
        """Test header info in chunk metadata."""
        chunker = MarkdownChunker()

        content = "# Main Header\nContent here."
        doc = Document.create(content=content, doc_type=DocumentType.MARKDOWN)
        chunks = chunker.chunk(doc)

        # Chunks should have header info in metadata or content
        assert len(chunks) >= 1


# =============================================================================
# Embedding Provider Tests
# =============================================================================
class TestMockEmbeddingProvider:
    """Tests for MockEmbeddingProvider."""

    def test_embed_returns_correct_dimension(self):
        """Test embedding has correct dimension."""
        provider = MockEmbeddingProvider(dimension=384)
        embedding = provider.embed("test")

        assert len(embedding) == 384

    def test_embed_deterministic(self):
        """Test same input gives same embedding."""
        provider = MockEmbeddingProvider(dimension=128)

        e1 = provider.embed("hello world")
        e2 = provider.embed("hello world")

        assert e1 == e2

    def test_embed_batch(self):
        """Test batch embedding."""
        provider = MockEmbeddingProvider(dimension=64)

        embeddings = provider.embed_batch(["a", "b", "c"])

        assert len(embeddings) == 3
        assert all(len(e) == 64 for e in embeddings)

    def test_different_texts_different_embeddings(self):
        """Test different texts get different embeddings."""
        provider = MockEmbeddingProvider(dimension=64)

        e1 = provider.embed("hello")
        e2 = provider.embed("world")

        assert e1 != e2


# =============================================================================
# Document Store Tests
# =============================================================================
class TestInMemoryDocumentStore:
    """Tests for InMemoryDocumentStore."""

    def test_save_and_get(self):
        """Test saving and retrieving document."""
        store = InMemoryDocumentStore()
        doc = Document.create(content="Test content", title="Test")

        store.save(doc)
        retrieved = store.get(doc.document_id)

        assert retrieved is not None
        assert retrieved.content == "Test content"

    def test_get_nonexistent(self):
        """Test getting nonexistent document."""
        store = InMemoryDocumentStore()

        assert store.get("nonexistent") is None

    def test_delete(self):
        """Test deleting document."""
        store = InMemoryDocumentStore()
        doc = Document.create(content="Test")

        store.save(doc)
        result = store.delete(doc.document_id)

        assert result is True
        assert store.get(doc.document_id) is None

    def test_list_all(self):
        """Test listing all document IDs."""
        store = InMemoryDocumentStore()

        doc1 = Document.create(content="Test 1")
        doc2 = Document.create(content="Test 2")

        store.save(doc1)
        store.save(doc2)

        ids = store.list_all()

        assert len(ids) == 2
        assert doc1.document_id in ids
        assert doc2.document_id in ids


# =============================================================================
# Search Result Tests
# =============================================================================
class TestSearchHit:
    """Tests for SearchHit."""

    def test_search_hit_creation(self):
        """Test creating a search hit."""
        hit = SearchHit(
            chunk_id="c1",
            document_id="doc1",
            content="Test content",
            score=0.95,
            metadata={"key": "value"},
        )

        assert hit.score == 0.95
        assert hit.metadata["key"] == "value"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hit = SearchHit(chunk_id="c1", document_id="doc1", content="Test", score=0.9)

        d = hit.to_dict()

        assert d["score"] == 0.9
        assert d["document_id"] == "doc1"


class TestSearchResult:
    """Tests for SearchResult."""

    def test_search_result_creation(self):
        """Test creating search result."""
        hits = [
            SearchHit("c1", "doc1", "Content 1", 0.9),
            SearchHit("c2", "doc1", "Content 2", 0.8),
        ]

        result = SearchResult(
            query="test query", hits=hits, total_hits=2, search_time_ms=10.5
        )

        assert result.total_hits == 2
        assert result.search_time_ms == 10.5

    def test_filter_by_score(self):
        """Test filtering by minimum score."""
        hits = [
            SearchHit("c1", "doc1", "Content 1", 0.9),
            SearchHit("c2", "doc1", "Content 2", 0.5),
            SearchHit("c3", "doc2", "Content 3", 0.3),
        ]

        result = SearchResult(query="test", hits=hits, total_hits=3, search_time_ms=10)
        filtered = result.filter_by_score(min_score=0.5)

        assert len(filtered.hits) == 2

    def test_get_unique_documents(self):
        """Test getting unique document IDs."""
        hits = [
            SearchHit("c1", "doc1", "Content 1", 0.9),
            SearchHit("c2", "doc1", "Content 2", 0.8),
            SearchHit("c3", "doc2", "Content 3", 0.7),
        ]

        result = SearchResult(query="test", hits=hits, total_hits=3, search_time_ms=10)
        unique_docs = result.get_unique_documents()

        assert len(unique_docs) == 2
        assert "doc1" in unique_docs
        assert "doc2" in unique_docs


# =============================================================================
# Document Processor Tests
# =============================================================================
class TestTextCleaner:
    """Tests for TextCleaner."""

    def test_remove_extra_whitespace(self):
        """Test removing extra whitespace."""
        cleaner = TextCleaner(remove_extra_whitespace=True)
        doc = Document.create(content="Hello    world\n\n\ntest")

        processed = cleaner.process(doc)

        assert "    " not in processed.content
        assert "\n\n\n" not in processed.content

    def test_lowercase(self):
        """Test lowercase conversion."""
        cleaner = TextCleaner(lowercase=True)
        doc = Document.create(content="Hello World")

        processed = cleaner.process(doc)

        assert processed.content == "hello world"


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    def test_extract_word_count(self):
        """Test word count extraction."""
        extractor = MetadataExtractor()
        doc = Document.create(content="Hello world this is a test")

        processed = extractor.process(doc)

        assert "word_count" in processed.metadata
        assert processed.metadata["word_count"] == 6

    def test_extract_char_count(self):
        """Test character count extraction."""
        extractor = MetadataExtractor()
        doc = Document.create(content="Hello")

        processed = extractor.process(doc)

        assert "char_count" in processed.metadata
        assert processed.metadata["char_count"] == 5


class TestIngestionPipeline:
    """Tests for IngestionPipeline."""

    def test_pipeline_execution(self):
        """Test pipeline processes in order."""
        pipeline = IngestionPipeline()
        pipeline.add_processor(TextCleaner(remove_extra_whitespace=True))
        pipeline.add_processor(MetadataExtractor())

        doc = Document.create(content="Hello    world")
        processed = pipeline.process(doc)

        # Should have cleaned text
        assert "    " not in processed.content
        # Should have extracted metadata
        assert "word_count" in processed.metadata


# =============================================================================
# Search Engine Tests
# =============================================================================
class TestSemanticSearchEngine:
    """Tests for SemanticSearchEngine."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mock components."""
        provider = MockEmbeddingProvider(dimension=64)
        store = InMemoryDocumentStore()

        # Create mock index that stores in memory
        class MockIndex:
            def __init__(self):
                self._data = {}

            def add(self, chunks):
                for chunk in chunks:
                    self._data[chunk.chunk_id] = {
                        "chunk": chunk,
                        "embedding": chunk.embedding,
                    }
                return len(chunks)

            def search(self, query_vector, k, filter=None):
                # Return all as mock results
                results = []
                for cid, data in list(self._data.items())[:k]:
                    results.append((cid, 0.9))
                return results

            def delete_by_document(self, document_id):
                to_delete = [
                    cid
                    for cid, data in self._data.items()
                    if data["chunk"].document_id == document_id
                ]
                for cid in to_delete:
                    del self._data[cid]
                return len(to_delete)

            def count(self):
                return len(self._data)

        index = MockIndex()

        return SemanticSearchEngine(
            embedding_provider=provider, vector_index=index, document_store=store
        )

    def test_ingest_document(self, mock_engine):
        """Test document ingestion."""
        doc_id = mock_engine.ingest(
            content="Test content for ingestion",
            title="Test Document",
            metadata={"topic": "test"},
        )

        assert doc_id is not None

    def test_get_document(self, mock_engine):
        """Test getting document after ingestion."""
        doc_id = mock_engine.ingest(content="Test content")

        doc = mock_engine.get_document(doc_id)

        assert doc is not None
        assert doc.content == "Test content"

    def test_delete_document(self, mock_engine):
        """Test deleting document."""
        doc_id = mock_engine.ingest(content="Test content")

        result = mock_engine.delete_document(doc_id)

        assert result is True
        assert mock_engine.get_document(doc_id) is None

    def test_list_documents(self, mock_engine):
        """Test listing documents."""
        mock_engine.ingest(content="Doc 1")
        mock_engine.ingest(content="Doc 2")

        doc_ids = mock_engine.list_documents()

        assert len(doc_ids) == 2


class TestSearchEngineConfig:
    """Tests for SearchEngineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SearchEngineConfig()

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 500
        assert config.collection_name == "documents"

    def test_custom_config(self):
        """Test custom configuration."""
        config = SearchEngineConfig(
            embedding_model="all-mpnet-base-v2",
            chunk_size=1000,
            collection_name="my_docs",
        )

        assert config.embedding_model == "all-mpnet-base-v2"
        assert config.chunk_size == 1000


# =============================================================================
# Integration Tests
# =============================================================================
class TestIntegration:
    """Integration tests for the search engine."""

    def test_end_to_end_workflow(self):
        """Test complete ingestion and search workflow."""
        # Create components
        provider = MockEmbeddingProvider(dimension=64)
        store = InMemoryDocumentStore()

        # Simple in-memory index
        class SimpleIndex:
            def __init__(self):
                self._chunks = {}

            def add(self, chunks):
                for chunk in chunks:
                    self._chunks[chunk.chunk_id] = chunk
                return len(chunks)

            def search(self, query_vector, k, filter=None):
                return [(cid, 0.9) for cid in list(self._chunks.keys())[:k]]

            def delete_by_document(self, doc_id):
                to_del = [
                    c for c, ch in self._chunks.items() if ch.document_id == doc_id
                ]
                for c in to_del:
                    del self._chunks[c]
                return len(to_del)

            def count(self):
                return len(self._chunks)

        index = SimpleIndex()
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        pipeline = IngestionPipeline()
        pipeline.add_processor(MetadataExtractor())

        engine = SemanticSearchEngine(
            embedding_provider=provider,
            vector_index=index,
            document_store=store,
            chunking_strategy=chunker,
            ingestion_pipeline=pipeline,
        )

        # Ingest documents
        docs = [
            ("Python is a great programming language", {"topic": "programming"}),
            ("Machine learning enables computers to learn", {"topic": "ai"}),
            ("Vector databases store embeddings", {"topic": "databases"}),
        ]

        doc_ids = []
        for content, metadata in docs:
            doc_id = engine.ingest(content=content, metadata=metadata)
            doc_ids.append(doc_id)

        assert len(doc_ids) == 3

        # Search
        result = engine.search("programming language", k=2)

        assert result is not None
        assert result.total_hits >= 0

        # Delete
        for doc_id in doc_ids:
            engine.delete_document(doc_id)

        assert len(engine.list_documents()) == 0

    def test_chunking_integration(self):
        """Test chunking with different document sizes."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20, min_chunk_size=50)

        # Small document - should be single chunk
        small_doc = Document.create(content="Small content")
        small_chunks = chunker.chunk(small_doc)
        assert len(small_chunks) == 1

        # Large document - should be multiple chunks
        large_doc = Document.create(content="A" * 500)
        large_chunks = chunker.chunk(large_doc)
        assert len(large_chunks) > 1
