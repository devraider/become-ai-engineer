"""
Week 8 - Exercise Basic 1: Document Loading & Processing
========================================================

Learn to load and process documents from various sources for RAG pipelines.

Instructions:
- Complete each TODO with your implementation
- Run tests with: pytest tests/test_exercise_basic_1_documents.py -v
- Check solutions in solutions/solution_basic_1_documents.py
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Optional imports - handled gracefully
try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    PYPDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Document:
    """Represents a loaded document."""

    content: str
    source: str
    metadata: Dict[str, Any]


# =============================================================================
# TASK 1: Load Text File
# =============================================================================


def load_text_file(file_path: str) -> Document:
    """
    Load a plain text file and return a Document object.

    Args:
        file_path: Path to the text file

    Returns:
        Document with content, source path, and metadata (file size, line count)

    Example:
        >>> doc = load_text_file("example.txt")
        >>> print(doc.content[:50])
        >>> print(doc.metadata['line_count'])
    """
    # TODO: Implement this function
    # 1. Read the file content
    # 2. Calculate metadata (file_size, line_count, word_count)
    # 3. Return Document with content, source, and metadata
    pass


# =============================================================================
# TASK 2: Load Multiple Files
# =============================================================================


def load_directory(directory_path: str, extension: str = ".txt") -> List[Document]:
    """
    Load all files with given extension from a directory.

    Args:
        directory_path: Path to directory
        extension: File extension to filter (default: ".txt")

    Returns:
        List of Document objects

    Example:
        >>> docs = load_directory("./documents", ".txt")
        >>> print(f"Loaded {len(docs)} documents")
    """
    # TODO: Implement this function
    # 1. Find all files with the given extension
    # 2. Load each file using load_text_file
    # 3. Return list of Documents
    pass


# =============================================================================
# TASK 3: Parse HTML Content
# =============================================================================


def parse_html_content(html: str) -> str:
    """
    Extract clean text from HTML content.

    Args:
        html: Raw HTML string

    Returns:
        Clean text without HTML tags, scripts, or styles

    Example:
        >>> html = "<html><body><p>Hello <b>World</b></p></body></html>"
        >>> text = parse_html_content(html)
        >>> print(text)  # "Hello World"
    """
    # TODO: Implement this function
    # 1. Parse HTML with BeautifulSoup (or regex fallback)
    # 2. Remove script and style elements
    # 3. Extract and clean text
    # 4. Return clean text
    pass


# =============================================================================
# TASK 4: Load Webpage (Optional - requires requests)
# =============================================================================


def load_webpage(url: str, timeout: int = 10) -> Document:
    """
    Load and parse a webpage.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Document with parsed content and URL as source

    Example:
        >>> doc = load_webpage("https://example.com")
        >>> print(doc.content[:100])
    """
    # TODO: Implement this function
    # 1. Fetch the webpage using requests
    # 2. Parse HTML content
    # 3. Return Document with content and metadata (url, status_code)
    pass


# =============================================================================
# TASK 5: Text Cleaning
# =============================================================================


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.

    Operations:
    - Remove extra whitespace
    - Normalize line endings
    - Remove special characters (keep basic punctuation)
    - Strip leading/trailing whitespace

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text

    Example:
        >>> text = "  Hello    World!  \n\n\n  How are you?  "
        >>> clean = clean_text(text)
        >>> print(clean)  # "Hello World!\n\nHow are you?"
    """
    # TODO: Implement this function
    # 1. Normalize line endings
    # 2. Remove extra whitespace
    # 3. Clean special characters
    # 4. Strip and return
    pass


# =============================================================================
# TASK 6: Extract Metadata
# =============================================================================


def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extract useful metadata from text content.

    Args:
        text: Text content

    Returns:
        Dictionary with:
        - char_count: Number of characters
        - word_count: Number of words
        - line_count: Number of lines
        - paragraph_count: Number of paragraphs (separated by blank lines)
        - avg_word_length: Average word length

    Example:
        >>> metadata = extract_metadata("Hello world. How are you?")
        >>> print(metadata['word_count'])  # 5
    """
    # TODO: Implement this function
    # Calculate all the metadata fields
    pass


# =============================================================================
# TASK 7: Document Splitter (Basic)
# =============================================================================


def split_by_separator(text: str, separator: str = "\n\n") -> List[str]:
    """
    Split text by a separator (e.g., paragraphs).

    Args:
        text: Text to split
        separator: String to split on

    Returns:
        List of non-empty text segments

    Example:
        >>> text = "Para 1\n\nPara 2\n\nPara 3"
        >>> parts = split_by_separator(text)
        >>> print(len(parts))  # 3
    """
    # TODO: Implement this function
    # Split by separator and filter empty strings
    pass


# =============================================================================
# TASK 8: Sentence Splitter
# =============================================================================


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Text to split

    Returns:
        List of sentences

    Example:
        >>> text = "Hello world. How are you? I'm fine!"
        >>> sentences = split_into_sentences(text)
        >>> print(len(sentences))  # 3
    """
    # TODO: Implement this function
    # Use regex to split on sentence boundaries (.!?)
    # Handle abbreviations (Mr., Dr., etc.) gracefully
    pass


# =============================================================================
# TASK 9: Batch Document Processor
# =============================================================================


def process_documents(
    documents: List[Document], clean: bool = True, extract_meta: bool = True
) -> List[Document]:
    """
    Process a batch of documents with optional cleaning and metadata extraction.

    Args:
        documents: List of Documents to process
        clean: Whether to clean the text
        extract_meta: Whether to extract metadata

    Returns:
        List of processed Documents with updated content and metadata
    """
    # TODO: Implement this function
    # 1. Iterate through documents
    # 2. Clean content if requested
    # 3. Extract metadata if requested
    # 4. Return processed documents
    pass


# =============================================================================
# TASK 10: Document Deduplication
# =============================================================================


def deduplicate_documents(
    documents: List[Document], similarity_threshold: float = 0.9
) -> List[Document]:
    """
    Remove near-duplicate documents based on content similarity.

    Uses simple approach: Jaccard similarity of word sets.

    Args:
        documents: List of Documents
        similarity_threshold: Threshold for considering duplicates (0.0 to 1.0)

    Returns:
        List of unique Documents

    Example:
        >>> docs = [
        ...     Document("Hello world", "doc1", {}),
        ...     Document("Hello world!", "doc2", {}),  # Near-duplicate
        ...     Document("Goodbye moon", "doc3", {})
        ... ]
        >>> unique = deduplicate_documents(docs, 0.8)
        >>> print(len(unique))  # 2
    """
    # TODO: Implement this function
    # 1. For each document, calculate word set
    # 2. Compare with existing documents using Jaccard similarity
    # 3. Keep only non-duplicates
    pass


# =============================================================================
# MAIN - Test your implementations
# =============================================================================

if __name__ == "__main__":
    print("Week 8 - Exercise Basic 1: Document Loading & Processing")
    print("=" * 60)

    # Test text cleaning
    print("\n1. Testing text cleaning:")
    dirty_text = "  Hello    World!  \n\n\n  How are you?  "
    cleaned = clean_text(dirty_text)
    print(f"Cleaned: '{cleaned}'")

    # Test metadata extraction
    print("\n2. Testing metadata extraction:")
    sample = "Hello world. How are you? I'm doing great today!"
    metadata = extract_metadata(sample)
    print(f"Metadata: {metadata}")

    # Test sentence splitting
    print("\n3. Testing sentence splitting:")
    sentences = split_into_sentences(sample)
    print(f"Sentences: {sentences}")

    # Test paragraph splitting
    print("\n4. Testing paragraph splitting:")
    para_text = "Paragraph 1 here.\n\nParagraph 2 here.\n\nParagraph 3."
    paragraphs = split_by_separator(para_text)
    print(f"Paragraphs: {paragraphs}")
