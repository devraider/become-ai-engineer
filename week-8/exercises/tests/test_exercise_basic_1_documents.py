"""
Tests for Week 8 - Exercise Basic 1: Document Loading & Processing
"""

import pytest
import os
import tempfile
from pathlib import Path

# Import exercise functions
from exercise_basic_1_documents import (
    Document,
    load_text_file,
    load_directory,
    parse_html_content,
    load_webpage,
    clean_text,
    extract_metadata,
    split_by_separator,
    split_into_sentences,
    process_documents,
    deduplicate_documents,
    PYPDF_AVAILABLE,
    BS4_AVAILABLE,
    REQUESTS_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_text_file():
    """Create a temporary text file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world!\nThis is a test file.\nLine three.")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_directory():
    """Create a temporary directory with text files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "doc1.txt").write_text("Document one content")
        (Path(tmpdir) / "doc2.txt").write_text("Document two content")
        (Path(tmpdir) / "ignore.md").write_text("Markdown file")
        yield tmpdir


# =============================================================================
# TASK 1: Load Text File
# =============================================================================


class TestLoadTextFile:
    def test_returns_document(self, temp_text_file):
        """Should return a Document object."""
        result = load_text_file(temp_text_file)
        assert result is not None
        assert isinstance(result, Document)

    def test_loads_content(self, temp_text_file):
        """Should load file content."""
        result = load_text_file(temp_text_file)
        assert "Hello world!" in result.content
        assert "test file" in result.content

    def test_includes_metadata(self, temp_text_file):
        """Should include metadata."""
        result = load_text_file(temp_text_file)
        assert "line_count" in result.metadata
        assert result.metadata["line_count"] == 3


# =============================================================================
# TASK 2: Load Directory
# =============================================================================


class TestLoadDirectory:
    def test_loads_matching_files(self, temp_directory):
        """Should load files with matching extension."""
        result = load_directory(temp_directory, ".txt")
        assert result is not None
        assert len(result) == 2

    def test_filters_by_extension(self, temp_directory):
        """Should only load files with specified extension."""
        result = load_directory(temp_directory, ".md")
        assert len(result) == 1


# =============================================================================
# TASK 3: Parse HTML Content
# =============================================================================


class TestParseHtmlContent:
    def test_extracts_text(self):
        """Should extract text from HTML."""
        html = "<html><body><p>Hello World</p></body></html>"
        result = parse_html_content(html)
        assert result is not None
        assert "Hello World" in result

    def test_removes_tags(self):
        """Should remove HTML tags."""
        html = "<p><b>Bold</b> text</p>"
        result = parse_html_content(html)
        assert "<b>" not in result
        assert "Bold text" in result or "Bold" in result

    @pytest.mark.skipif(not BS4_AVAILABLE, reason="BeautifulSoup not installed")
    def test_removes_scripts(self):
        """Should remove script content."""
        html = "<html><script>alert('hi')</script><p>Content</p></html>"
        result = parse_html_content(html)
        assert "alert" not in result
        assert "Content" in result


# =============================================================================
# TASK 4: Load Webpage
# =============================================================================


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
class TestLoadWebpage:
    def test_returns_document(self):
        """Should return a Document object."""
        # Use a simple, reliable test URL
        result = load_webpage("https://example.com", timeout=5)
        if result:
            assert isinstance(result, Document)
            assert len(result.content) > 0


# =============================================================================
# TASK 5: Text Cleaning
# =============================================================================


class TestCleanText:
    def test_removes_extra_whitespace(self):
        """Should collapse multiple spaces."""
        result = clean_text("Hello    World")
        assert result is not None
        assert "    " not in result

    def test_normalizes_line_endings(self):
        """Should normalize line endings."""
        result = clean_text("Line1\r\nLine2\rLine3")
        assert "\r" not in result

    def test_strips_text(self):
        """Should strip leading/trailing whitespace."""
        result = clean_text("  Hello World  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")


# =============================================================================
# TASK 6: Extract Metadata
# =============================================================================


class TestExtractMetadata:
    def test_returns_dict(self):
        """Should return a dictionary."""
        result = extract_metadata("Hello world")
        assert result is not None
        assert isinstance(result, dict)

    def test_char_count(self):
        """Should count characters."""
        result = extract_metadata("Hello")
        assert result["char_count"] == 5

    def test_word_count(self):
        """Should count words."""
        result = extract_metadata("Hello world how are you")
        assert result["word_count"] == 5

    def test_line_count(self):
        """Should count lines."""
        result = extract_metadata("Line 1\nLine 2\nLine 3")
        assert result["line_count"] == 3


# =============================================================================
# TASK 7: Split by Separator
# =============================================================================


class TestSplitBySeparator:
    def test_splits_paragraphs(self):
        """Should split by double newline."""
        text = "Para 1\n\nPara 2\n\nPara 3"
        result = split_by_separator(text, "\n\n")
        assert result is not None
        assert len(result) == 3

    def test_filters_empty(self):
        """Should filter empty strings."""
        text = "Part 1\n\n\n\nPart 2"
        result = split_by_separator(text, "\n\n")
        assert "" not in result


# =============================================================================
# TASK 8: Sentence Splitter
# =============================================================================


class TestSplitIntoSentences:
    def test_splits_sentences(self):
        """Should split on sentence boundaries."""
        text = "Hello world. How are you? I'm fine!"
        result = split_into_sentences(text)
        assert result is not None
        assert len(result) == 3

    def test_handles_abbreviations(self):
        """Should handle common abbreviations."""
        text = "Dr. Smith went to work. He arrived at 9 a.m."
        result = split_into_sentences(text)
        # Should not split on Dr. or a.m.
        assert len(result) <= 3


# =============================================================================
# TASK 9: Process Documents
# =============================================================================


class TestProcessDocuments:
    def test_cleans_content(self):
        """Should clean document content."""
        docs = [Document("  Hello    World  ", "test", {})]
        result = process_documents(docs, clean=True, extract_meta=False)
        assert result is not None
        assert "    " not in result[0].content

    def test_extracts_metadata(self):
        """Should extract metadata."""
        docs = [Document("Hello world", "test", {})]
        result = process_documents(docs, clean=False, extract_meta=True)
        assert "word_count" in result[0].metadata


# =============================================================================
# TASK 10: Document Deduplication
# =============================================================================


class TestDeduplicateDocuments:
    def test_removes_exact_duplicates(self):
        """Should remove exact duplicates."""
        docs = [
            Document("Hello world", "doc1", {}),
            Document("Hello world", "doc2", {}),
            Document("Different text", "doc3", {}),
        ]
        result = deduplicate_documents(docs, similarity_threshold=1.0)
        assert result is not None
        assert len(result) == 2

    def test_removes_near_duplicates(self):
        """Should remove near-duplicates."""
        docs = [
            Document("Hello world", "doc1", {}),
            Document("Hello world!", "doc2", {}),  # Very similar
            Document("Completely different content here", "doc3", {}),
        ]
        result = deduplicate_documents(docs, similarity_threshold=0.8)
        # Should keep doc1 and doc3, remove doc2 as near-duplicate
        assert len(result) <= 2


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
