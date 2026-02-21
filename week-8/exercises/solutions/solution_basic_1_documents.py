"""
Week 8 - Exercise Basic 1: Document Loading & Processing - SOLUTIONS
====================================================================
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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
    """Load a plain text file and return a Document object."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    lines = content.split("\n")
    words = content.split()

    metadata = {
        "file_size": path.stat().st_size,
        "line_count": len(lines),
        "word_count": len(words),
        "filename": path.name,
    }

    return Document(content=content, source=str(path), metadata=metadata)


# =============================================================================
# TASK 2: Load Multiple Files
# =============================================================================


def load_directory(directory_path: str, extension: str = ".txt") -> List[Document]:
    """Load all files with given extension from a directory."""
    path = Path(directory_path)
    documents = []

    for file_path in path.glob(f"*{extension}"):
        if file_path.is_file():
            doc = load_text_file(str(file_path))
            documents.append(doc)

    return documents


# =============================================================================
# TASK 3: Parse HTML Content
# =============================================================================


def parse_html_content(html: str) -> str:
    """Extract clean text from HTML content."""
    if BS4_AVAILABLE and BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "head", "meta", "link"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)
    else:
        # Fallback: regex-based parsing
        # Remove script and style tags with content
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')

    # Clean up whitespace
    text = " ".join(text.split())
    return text.strip()


# =============================================================================
# TASK 4: Load Webpage
# =============================================================================


def load_webpage(url: str, timeout: int = 10) -> Document:
    """Load and parse a webpage."""
    if not REQUESTS_AVAILABLE:
        return Document(
            content="", source=url, metadata={"error": "requests not available"}
        )

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        content = parse_html_content(response.text)

        metadata = {
            "url": url,
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "content_length": len(content),
        }

        return Document(content=content, source=url, metadata=metadata)

    except Exception as e:
        return Document(content="", source=url, metadata={"error": str(e)})


# =============================================================================
# TASK 5: Text Cleaning
# =============================================================================


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple spaces (but preserve newlines)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Collapse multiple spaces in each line
        cleaned_line = " ".join(line.split())
        cleaned_lines.append(cleaned_line)

    # Rejoin lines, collapsing multiple blank lines
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    return text.strip()


# =============================================================================
# TASK 6: Extract Metadata
# =============================================================================


def extract_metadata(text: str) -> Dict[str, Any]:
    """Extract useful metadata from text content."""
    words = text.split()
    lines = text.split("\n")
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    word_lengths = [len(w) for w in words]
    avg_word_length = sum(word_lengths) / len(words) if words else 0

    return {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": len(lines),
        "paragraph_count": len(paragraphs),
        "avg_word_length": round(avg_word_length, 2),
    }


# =============================================================================
# TASK 7: Document Splitter (Basic)
# =============================================================================


def split_by_separator(text: str, separator: str = "\n\n") -> List[str]:
    """Split text by a separator (e.g., paragraphs)."""
    parts = text.split(separator)
    # Filter empty strings and strip whitespace
    return [part.strip() for part in parts if part.strip()]


# =============================================================================
# TASK 8: Sentence Splitter
# =============================================================================


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Common abbreviations to avoid splitting
    abbreviations = [
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Sr.",
        "Jr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "Inc.",
        "Ltd.",
        "Corp.",
        "a.m.",
        "p.m.",
        "U.S.",
        "U.K.",
    ]

    # Protect abbreviations by replacing periods with placeholder
    protected = text
    for abbr in abbreviations:
        protected = protected.replace(abbr, abbr.replace(".", "<<DOT>>"))

    # Split on sentence boundaries
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, protected)

    # Restore periods in abbreviations
    sentences = [s.replace("<<DOT>>", ".").strip() for s in sentences]

    # Filter empty strings
    return [s for s in sentences if s]


# =============================================================================
# TASK 9: Batch Document Processor
# =============================================================================


def process_documents(
    documents: List[Document], clean: bool = True, extract_meta: bool = True
) -> List[Document]:
    """Process a batch of documents with optional cleaning and metadata extraction."""
    processed = []

    for doc in documents:
        content = doc.content
        metadata = dict(doc.metadata)

        if clean:
            content = clean_text(content)

        if extract_meta:
            new_meta = extract_metadata(content)
            metadata.update(new_meta)

        processed.append(
            Document(content=content, source=doc.source, metadata=metadata)
        )

    return processed


# =============================================================================
# TASK 10: Document Deduplication
# =============================================================================


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def deduplicate_documents(
    documents: List[Document], similarity_threshold: float = 0.9
) -> List[Document]:
    """Remove near-duplicate documents based on content similarity."""
    if not documents:
        return []

    unique = []
    unique_word_sets = []

    for doc in documents:
        # Create word set for document
        words = set(doc.content.lower().split())

        # Check if similar to any existing document
        is_duplicate = False
        for existing_words in unique_word_sets:
            similarity = _jaccard_similarity(words, existing_words)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(doc)
            unique_word_sets.append(words)

    return unique


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Week 8 - Exercise Basic 1: Document Loading - SOLUTIONS")
    print("=" * 60)

    # Test text cleaning
    print("\n1. Text Cleaning:")
    dirty = "  Hello    World!  \n\n\n  How are you?  "
    print(f"Cleaned: '{clean_text(dirty)}'")

    # Test metadata extraction
    print("\n2. Metadata Extraction:")
    sample = "Hello world. How are you? I'm doing great today!"
    print(f"Metadata: {extract_metadata(sample)}")

    # Test sentence splitting
    print("\n3. Sentence Splitting:")
    print(f"Sentences: {split_into_sentences(sample)}")

    # Test HTML parsing
    print("\n4. HTML Parsing:")
    html = "<html><body><p>Hello <b>World</b></p></body></html>"
    print(f"Parsed: '{parse_html_content(html)}'")

    # Test deduplication
    print("\n5. Deduplication:")
    docs = [
        Document("Hello world", "doc1", {}),
        Document("Hello world!", "doc2", {}),
        Document("Goodbye moon", "doc3", {}),
    ]
    unique = deduplicate_documents(docs, 0.8)
    print(f"Unique docs: {len(unique)}")
