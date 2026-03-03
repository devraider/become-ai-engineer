"""
Tests for Week 13 - Exercise 2: ChromaDB Integration

Run with: pytest week-13/exercises/tests/test_exercise_intermediate_2_chromadb.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_intermediate_2_chromadb import (
    Document,
    CollectionConfig,
    QueryFilter,
    QueryBuilder,
    ChromaSearchResult,
    SearchResults,
    ChromaDBClient,
    DocumentCollection,
    DocumentPreprocessor,
    MetadataFilterBuilder,
    CollectionStats,
    DocumentSearchSystem,
)


# =============================================================================
# Part 1: Document Tests
# =============================================================================
class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(id="doc1", content="Test content", metadata={"topic": "test"})

        assert doc.id == "doc1"
        assert doc.content == "Test content"
        assert doc.metadata["topic"] == "test"

    def test_from_text(self):
        """Test creating document from text."""
        doc = Document.from_text("Hello, world!", metadata={"source": "test"})

        assert doc.id is not None
        assert doc.content == "Hello, world!"
        assert doc.metadata["source"] == "test"

    def test_from_text_generates_id(self):
        """Test that from_text generates unique IDs."""
        doc1 = Document.from_text("Content 1")
        doc2 = Document.from_text("Content 2")

        assert doc1.id != doc2.id

    def test_to_chroma_format(self):
        """Test conversion to ChromaDB format."""
        doc = Document(id="doc1", content="Test", metadata={"key": "value"})

        chroma_format = doc.to_chroma_format()

        assert "id" in chroma_format
        assert "document" in chroma_format
        assert "metadata" in chroma_format

    def test_validate_valid_document(self):
        """Test validation of valid document."""
        doc = Document(
            id="doc1",
            content="Valid content",
            metadata={"str_key": "value", "int_key": 42, "bool_key": True},
        )

        assert doc.validate() is True

    def test_validate_empty_id(self):
        """Test validation fails for empty ID."""
        doc = Document(id="", content="Test")
        assert doc.validate() is False

    def test_validate_empty_content(self):
        """Test validation fails for empty content."""
        doc = Document(id="doc1", content="")
        assert doc.validate() is False


# =============================================================================
# Part 2: Collection Config Tests
# =============================================================================
class TestCollectionConfig:
    """Tests for CollectionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CollectionConfig(name="test")

        assert config.name == "test"
        assert config.distance_metric == "cosine"
        assert config.model_name == "all-MiniLM-L6-v2"

    def test_to_metadata(self):
        """Test metadata conversion."""
        config = CollectionConfig(name="test", distance_metric="l2")

        metadata = config.to_metadata()

        assert isinstance(metadata, dict)
        assert "hnsw:space" in metadata or config.distance_metric in str(metadata)


# =============================================================================
# Part 3: Query Builder Tests
# =============================================================================
class TestQueryFilter:
    """Tests for QueryFilter."""

    def test_equality_filter(self):
        """Test equality filter format."""
        filter = QueryFilter(field="category", operator="$eq", value="test")
        chroma_format = filter.to_chroma_format()

        assert "category" in chroma_format
        assert chroma_format["category"]["$eq"] == "test"

    def test_comparison_filters(self):
        """Test comparison filter formats."""
        filters = [
            QueryFilter(field="score", operator="$gt", value=0.5),
            QueryFilter(field="count", operator="$lte", value=100),
        ]

        for f in filters:
            result = f.to_chroma_format()
            assert f.field in result


class TestQueryBuilder:
    """Tests for QueryBuilder."""

    def test_with_text(self):
        """Test setting query text."""
        query = QueryBuilder().with_text("test query").build()

        assert "query_texts" in query
        assert query["query_texts"] == ["test query"]

    def test_with_texts(self):
        """Test setting multiple query texts."""
        query = QueryBuilder().with_texts(["q1", "q2"]).build()

        assert query["query_texts"] == ["q1", "q2"]

    def test_limit(self):
        """Test setting result limit."""
        query = QueryBuilder().with_text("test").limit(5).build()

        assert query["n_results"] == 5

    def test_where_filter(self):
        """Test adding where filter."""
        filter = QueryFilter(field="category", operator="$eq", value="test")
        query = QueryBuilder().with_text("test").where(filter).build()

        assert "where" in query

    def test_where_and(self):
        """Test AND filters."""
        filters = [
            QueryFilter(field="a", operator="$eq", value=1),
            QueryFilter(field="b", operator="$eq", value=2),
        ]
        query = QueryBuilder().with_text("test").where_and(filters).build()

        assert "where" in query
        assert "$and" in query["where"]

    def test_where_or(self):
        """Test OR filters."""
        filters = [
            QueryFilter(field="a", operator="$eq", value=1),
            QueryFilter(field="b", operator="$eq", value=2),
        ]
        query = QueryBuilder().with_text("test").where_or(filters).build()

        assert "where" in query
        assert "$or" in query["where"]

    def test_include_options(self):
        """Test include options."""
        query = (
            QueryBuilder()
            .with_text("test")
            .include_distances()
            .include_embeddings()
            .build()
        )

        assert "include" in query
        assert "distances" in query["include"]
        assert "embeddings" in query["include"]


# =============================================================================
# Part 4: Search Result Tests
# =============================================================================
class TestChromaSearchResult:
    """Tests for ChromaSearchResult."""

    def test_result_creation(self):
        """Test search result creation."""
        result = ChromaSearchResult(
            id="doc1", document="Test content", metadata={"key": "value"}, distance=0.1
        )

        assert result.id == "doc1"
        assert result.distance == 0.1

    def test_score_property(self):
        """Test score calculation from distance."""
        result = ChromaSearchResult(
            id="doc1", document="Test", metadata={}, distance=0.2
        )

        # For cosine distance, score = 1 - distance
        assert result.score == pytest.approx(0.8)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ChromaSearchResult(
            id="doc1", document="Test", metadata={"key": "value"}, distance=0.1
        )

        d = result.to_dict()

        assert d["id"] == "doc1"
        assert d["document"] == "Test"
        assert "score" in d


class TestSearchResults:
    """Tests for SearchResults container."""

    @pytest.fixture
    def raw_results(self):
        """Create sample raw ChromaDB results."""
        return {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["Content 1", "Content 2", "Content 3"]],
            "metadatas": [[{"key": "a"}, {"key": "b"}, {"key": "c"}]],
            "distances": [[0.1, 0.3, 0.5]],
        }

    def test_parse_results(self, raw_results):
        """Test parsing raw results."""
        results = SearchResults(raw_results)

        assert len(results) == 3

    def test_iteration(self, raw_results):
        """Test iterating over results."""
        results = SearchResults(raw_results)

        ids = [r.id for r in results]
        assert ids == ["doc1", "doc2", "doc3"]

    def test_indexing(self, raw_results):
        """Test accessing results by index."""
        results = SearchResults(raw_results)

        assert results[0].id == "doc1"
        assert results[2].id == "doc3"

    def test_filter_by_score(self, raw_results):
        """Test filtering by score."""
        results = SearchResults(raw_results)
        filtered = results.filter_by_score(min_score=0.7)

        # Only results with score >= 0.7 (distance <= 0.3)
        assert len(filtered) <= 2

    def test_to_documents(self, raw_results):
        """Test converting to Document objects."""
        results = SearchResults(raw_results)
        docs = results.to_documents()

        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)


# =============================================================================
# Part 5: Document Preprocessor Tests
# =============================================================================
class TestDocumentPreprocessor:
    """Tests for DocumentPreprocessor."""

    def test_clean_text_whitespace(self):
        """Test whitespace cleaning."""
        preprocessor = DocumentPreprocessor()

        text = "Hello   world\n\n\ntest"
        cleaned = preprocessor.clean_text(text)

        assert "   " not in cleaned

    def test_chunk_document(self):
        """Test document chunking."""
        preprocessor = DocumentPreprocessor(
            chunk_size=100, chunk_overlap=20, min_chunk_size=50
        )

        doc = Document(id="doc1", content="A" * 300)  # 300 character document

        chunks = preprocessor.chunk_document(doc)

        assert len(chunks) > 1

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        preprocessor = DocumentPreprocessor(chunk_size=100)

        doc = Document(id="doc1", content="A" * 300, metadata={"source": "test"})

        chunks = preprocessor.chunk_document(doc)

        # Each chunk should reference parent document
        for chunk in chunks:
            assert (
                chunk.metadata.get("parent_id") == "doc1" or "source" in chunk.metadata
            )

    def test_extract_metadata(self):
        """Test metadata extraction."""
        preprocessor = DocumentPreprocessor()

        doc = Document(id="doc1", content="Hello world! This is a test.")

        metadata = preprocessor.extract_metadata(doc)

        assert "word_count" in metadata
        assert "char_count" in metadata


# =============================================================================
# Part 6: Metadata Filter Builder Tests
# =============================================================================
class TestMetadataFilterBuilder:
    """Tests for MetadataFilterBuilder."""

    def test_equals_filter(self):
        """Test equality filter."""
        builder = MetadataFilterBuilder()
        filter = builder.equals("category", "test").build()

        assert filter["category"]["$eq"] == "test"

    def test_not_equals_filter(self):
        """Test inequality filter."""
        builder = MetadataFilterBuilder()
        filter = builder.not_equals("category", "test").build()

        assert filter["category"]["$ne"] == "test"

    def test_comparison_filters(self):
        """Test comparison filters."""
        filter = MetadataFilterBuilder().greater_than("score", 0.5).build()

        assert filter["score"]["$gt"] == 0.5

    def test_in_list_filter(self):
        """Test in-list filter."""
        filter = MetadataFilterBuilder().in_list("category", ["a", "b", "c"]).build()

        assert filter["category"]["$in"] == ["a", "b", "c"]

    def test_combined_and_filters(self):
        """Test AND combination."""
        filter = (
            MetadataFilterBuilder()
            .use_and()
            .equals("type", "article")
            .greater_than("score", 0.5)
            .build()
        )

        assert "$and" in filter

    def test_combined_or_filters(self):
        """Test OR combination."""
        filter = (
            MetadataFilterBuilder()
            .use_or()
            .equals("category", "a")
            .equals("category", "b")
            .build()
        )

        assert "$or" in filter


# =============================================================================
# Part 7: Document Collection Tests (Mock-based)
# =============================================================================
class TestDocumentCollection:
    """Tests for DocumentCollection."""

    def test_add_document(self):
        """Test adding a document."""
        # This would require mocking ChromaDBClient
        # For now, test the interface is correct
        pass

    def test_search_interface(self):
        """Test search method signature."""
        # Verify QueryBuilder integration
        query = QueryBuilder().with_text("test").limit(5).build()

        assert "query_texts" in query
        assert "n_results" in query


# =============================================================================
# Part 8: Document Search System Tests
# =============================================================================
class TestDocumentSearchSystem:
    """Tests for DocumentSearchSystem."""

    def test_initialization(self):
        """Test system initialization."""
        system = DocumentSearchSystem(
            persist_directory=None, collection_name="test"  # In-memory
        )

        assert system._collection_name == "test"

    def test_add_document_interface(self):
        """Test add_document method signature."""
        system = DocumentSearchSystem(collection_name="test")

        # Method should accept content, metadata, and chunk flag
        # Full test would require ChromaDB setup
        assert callable(system.add_document)
        assert callable(system.search)

    def test_search_interface(self):
        """Test search method signature."""
        system = DocumentSearchSystem(collection_name="test")

        # Verify method exists and is callable
        assert callable(system.search)


# =============================================================================
# Integration Tests (require ChromaDB)
# =============================================================================
@pytest.mark.skip(reason="Requires ChromaDB installation")
class TestChromaDBIntegration:
    """Integration tests requiring actual ChromaDB."""

    def test_full_workflow(self):
        """Test complete add-search-delete workflow."""
        system = DocumentSearchSystem(
            persist_directory=None, collection_name="integration_test"
        )
        system.initialize()

        # Add documents
        doc_ids = system.add_documents(
            [
                ("Python is a programming language.", {"topic": "programming"}),
                ("Machine learning enables AI.", {"topic": "ai"}),
            ]
        )

        assert len(doc_ids) == 2

        # Search
        results = system.search("programming language", n_results=2)
        assert len(results) > 0

        # Delete
        for doc_id in doc_ids:
            system.delete_document(doc_id)

        # Verify deletion
        stats = system.get_stats()
        assert stats.get("count", 0) == 0
