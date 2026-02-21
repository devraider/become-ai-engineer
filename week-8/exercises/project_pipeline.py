"""
Week 8 Project: Document Q&A System
====================================

Build a complete RAG (Retrieval Augmented Generation) system.

This project combines everything learned this week:
- Document loading and processing
- Text chunking
- Embeddings and vector storage
- Retrieval strategies
- Response generation

Instructions:
- Complete each TODO section
- Run tests with: pytest tests/test_project_pipeline.py -v
- Check solutions in solutions/solution_project_pipeline.py
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re

# Optional imports
try:
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Document:
    """Represents a document with metadata."""

    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class Chunk:
    """A chunk of a document."""

    content: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.document_id}_chunk_{self.chunk_index}"


@dataclass
class SearchResult:
    """A search result with relevance score."""

    chunk: Chunk
    score: float


@dataclass
class QAResponse:
    """A Q&A response with sources."""

    question: str
    answer: str
    sources: List[SearchResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PART 1: Document Loader
# =============================================================================


class DocumentLoader:
    """Load documents from various sources."""

    def load_text(self, file_path: str) -> Document:
        """
        Load a text file.

        Args:
            file_path: Path to text file

        Returns:
            Document object with content and metadata
        """
        # TODO: Implement this method
        # 1. Read file content
        # 2. Create Document with source and metadata (filename, size, etc.)
        pass

    def load_directory(
        self, directory_path: str, extensions: List[str] = None
    ) -> List[Document]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to directory
            extensions: List of extensions to include (e.g., ['.txt', '.md'])

        Returns:
            List of Document objects
        """
        # TODO: Implement this method
        pass

    def load_from_string(self, content: str, source: str = "inline") -> Document:
        """Create a Document from a string."""
        # TODO: Implement this method
        pass


# =============================================================================
# PART 2: Text Chunker
# =============================================================================


class TextChunker:
    """Split documents into chunks for embedding."""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_size: int = 50
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (skip smaller chunks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        # TODO: Implement this method
        # 1. Split text into chunks with overlap
        # 2. Create Chunk objects with metadata
        # 3. Skip chunks smaller than min_chunk_size
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        # TODO: Implement this method
        pass


# =============================================================================
# PART 3: Embedding Manager
# =============================================================================


class EmbeddingManager:
    """Manage text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager.

        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # TODO: Implement this method
        # Use self.model.encode() if available
        # Fall back to simple TF-IDF or random if not
        pass

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        # TODO: Implement this method
        pass


# =============================================================================
# PART 4: Vector Store
# =============================================================================


class VectorStore:
    """Store and retrieve document chunks."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name for the collection
            persist_directory: Directory to persist data (optional)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # In-memory fallback storage
        self._chunks: List[Chunk] = []
        self._embeddings: List[List[float]] = []

        # Try to use ChromaDB
        self._chroma_client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB if available."""
        if CHROMADB_AVAILABLE:
            try:
                if self.persist_directory:
                    self._chroma_client = chromadb.PersistentClient(
                        path=self.persist_directory
                    )
                else:
                    self._chroma_client = chromadb.Client()

                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self._chroma_client = None

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks with their embeddings.

        Args:
            chunks: List of Chunk objects
            embeddings: Corresponding embeddings
        """
        # TODO: Implement this method
        # If ChromaDB available, use it
        # Otherwise, use in-memory storage
        pass

    def search(
        self,
        query_embedding: List[float],
        k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            k: Number of results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        # TODO: Implement this method
        pass

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        # TODO: Implement this method
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        # TODO: Implement this method
        # Return: total_chunks, unique_sources, etc.
        pass


# =============================================================================
# PART 5: Response Generator
# =============================================================================


class ResponseGenerator:
    """Generate responses using retrieved context."""

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the generator.

        Args:
            model_name: LLM model name
        """
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the LLM."""
        if GENAI_AVAILABLE:
            try:
                load_dotenv()
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel(self.model_name)
            except Exception:
                self.model = None

    def generate(
        self,
        question: str,
        context_chunks: List[SearchResult],
        include_sources: bool = True,
    ) -> str:
        """
        Generate an answer using retrieved context.

        Args:
            question: User's question
            context_chunks: Retrieved chunks with scores
            include_sources: Whether to cite sources in response

        Returns:
            Generated answer text
        """
        # TODO: Implement this method
        # 1. Format context from chunks
        # 2. Create prompt with context and question
        # 3. Generate response with LLM (or return context-based answer if no LLM)
        pass

    def _format_context(self, chunks: List[SearchResult]) -> str:
        """Format chunks into context string."""
        # TODO: Implement this method
        pass

    def _create_prompt(self, question: str, context: str, include_sources: bool) -> str:
        """Create the prompt for the LLM."""
        # TODO: Implement this method
        pass


# =============================================================================
# PART 6: RAG Pipeline
# =============================================================================


class RAGPipeline:
    """Complete RAG pipeline for document Q&A."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "rag_docs",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence transformer model
            collection_name: Vector store collection name
            persist_directory: Directory for persistence
        """
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingManager(embedding_model)
        self.vector_store = VectorStore(collection_name, persist_directory)
        self.generator = ResponseGenerator()

    def ingest_text(self, text: str, source: str = "inline") -> int:
        """
        Ingest text content.

        Args:
            text: Text content to ingest
            source: Source identifier

        Returns:
            Number of chunks created
        """
        # TODO: Implement this method
        # 1. Create Document from text
        # 2. Chunk the document
        # 3. Embed chunks
        # 4. Add to vector store
        # 5. Return chunk count
        pass

    def ingest_file(self, file_path: str) -> int:
        """Ingest a file."""
        # TODO: Implement this method
        pass

    def ingest_directory(
        self, directory_path: str, extensions: List[str] = None
    ) -> int:
        """Ingest all files from a directory."""
        # TODO: Implement this method
        pass

    def query(
        self, question: str, k: int = 3, include_sources: bool = True
    ) -> QAResponse:
        """
        Query the RAG system.

        Args:
            question: User's question
            k: Number of chunks to retrieve
            include_sources: Whether to include source citations

        Returns:
            QAResponse with answer and sources
        """
        # TODO: Implement this method
        # 1. Embed the question
        # 2. Search for relevant chunks
        # 3. Generate response
        # 4. Return QAResponse
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        # TODO: Implement this method
        pass


# =============================================================================
# PART 7: Query Interface
# =============================================================================


class DocumentQA:
    """
    High-level interface for document Q&A.

    Usage:
        qa = DocumentQA()
        qa.add_document("path/to/doc.txt")
        qa.add_text("Some inline content")
        response = qa.ask("What is the main topic?")
        print(response.answer)
    """

    def __init__(self, **kwargs):
        """Initialize with optional pipeline configuration."""
        self.pipeline = RAGPipeline(**kwargs)
        self.history: List[QAResponse] = []

    def add_document(self, file_path: str) -> int:
        """Add a document to the knowledge base."""
        # TODO: Implement this method
        pass

    def add_documents(self, directory_path: str, extensions: List[str] = None) -> int:
        """Add all documents from a directory."""
        # TODO: Implement this method
        pass

    def add_text(self, text: str, source: str = "inline") -> int:
        """Add text content to the knowledge base."""
        # TODO: Implement this method
        pass

    def ask(
        self, question: str, k: int = 3, include_sources: bool = True
    ) -> QAResponse:
        """
        Ask a question about the documents.

        Args:
            question: Question to ask
            k: Number of relevant chunks to use
            include_sources: Whether to include source citations

        Returns:
            QAResponse with answer and sources
        """
        # TODO: Implement this method
        pass

    def get_history(self) -> List[QAResponse]:
        """Get conversation history."""
        return self.history

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self.pipeline.get_stats()
        stats["questions_asked"] = len(self.history)
        return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Week 8 Project: Document Q&A System")
    print("=" * 60)

    # Create Q&A system
    qa = DocumentQA(chunk_size=200, chunk_overlap=20)

    # Add some sample content
    sample_docs = [
        """
        Retrieval Augmented Generation (RAG) is a technique that combines 
        information retrieval with text generation. It helps LLMs provide 
        more accurate and up-to-date responses by grounding them in 
        retrieved documents.
        """,
        """
        Vector databases store embeddings, which are numerical representations 
        of text. They enable semantic search, where queries find documents 
        based on meaning rather than just keyword matching.
        """,
        """
        Chunking is the process of splitting documents into smaller pieces.
        Good chunking strategies preserve context while keeping chunks 
        small enough for effective retrieval.
        """,
    ]

    print("\nIngesting sample documents...")
    total_chunks = 0
    for i, doc in enumerate(sample_docs):
        chunks = qa.add_text(doc.strip(), source=f"sample_doc_{i+1}")
        if chunks:
            total_chunks += chunks
            print(f"  Document {i+1}: {chunks} chunks")

    print(f"\nTotal chunks: {total_chunks}")

    # Ask questions
    questions = [
        "What is RAG?",
        "How do vector databases work?",
        "Why is chunking important?",
    ]

    print("\nAsking questions...")
    for q in questions:
        print(f"\nQ: {q}")
        response = qa.ask(q)
        if response:
            print(f"A: {response.answer[:200]}...")
            if response.sources:
                print(f"Sources: {len(response.sources)} chunks")

    # Print stats
    print("\nSystem Stats:")
    stats = qa.get_stats()
    if stats:
        for key, value in stats.items():
            print(f"  {key}: {value}")
