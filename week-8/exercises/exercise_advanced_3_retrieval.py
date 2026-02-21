"""
Week 8 - Exercise Advanced 3: Retrieval Strategies
====================================================

Learn advanced retrieval techniques for RAG systems.

Instructions:
- Complete each TODO with your implementation
- Run tests with: pytest tests/test_exercise_advanced_3_retrieval.py -v
- Check solutions in solutions/solution_advanced_3_retrieval.py
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import json
import math

# Optional imports
try:
    import chromadb
    from chromadb.utils import embedding_functions

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    embedding_functions = None
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class RetrievedDocument:
    """A retrieved document with score and metadata."""

    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


# =============================================================================
# TASK 1: Simple Vector Store
# =============================================================================


class SimpleVectorStore:
    """
    A simple in-memory vector store implementation.

    Demonstrates core vector store concepts without external dependencies.
    """

    def __init__(self):
        """Initialize the vector store."""
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            ids: Optional document IDs
            metadatas: Optional metadata dictionaries

        Example:
            >>> store = SimpleVectorStore()
            >>> store.add(
            ...     documents=["hello", "world"],
            ...     embeddings=[[1, 0], [0, 1]],
            ...     ids=["doc1", "doc2"]
            ... )
        """
        # TODO: Implement this method
        # 1. Add documents, embeddings to lists
        # 2. Generate IDs if not provided
        # 3. Add metadata (empty dict if not provided)
        pass

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """
        Query the vector store for similar documents.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter (simple equality only)

        Returns:
            List of RetrievedDocument objects sorted by similarity

        Example:
            >>> results = store.query([0.5, 0.5], n_results=2)
            >>> print(results[0].content, results[0].score)
        """
        # TODO: Implement this method
        # 1. Filter by metadata if 'where' is provided
        # 2. Calculate cosine similarity with query
        # 3. Sort by similarity
        # 4. Return top n_results as RetrievedDocument objects
        pass

    def delete(self, ids: List[str]) -> int:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        # TODO: Implement this method
        pass

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Update documents by ID.

        Args:
            ids: Document IDs to update
            documents: New document texts
            embeddings: New embeddings
            metadatas: New metadata

        Returns:
            Number of documents updated
        """
        # TODO: Implement this method
        pass


# =============================================================================
# TASK 2: Text Chunker with Overlap
# =============================================================================


class TextChunker:
    """Split text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: Split boundaries (default: paragraphs, newlines, sentences, spaces)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks

        Example:
            >>> chunker = TextChunker(chunk_size=100, chunk_overlap=20)
            >>> chunks = chunker.chunk("A long document...")
        """
        # TODO: Implement this method
        # 1. Try to split at natural boundaries
        # 2. Respect chunk_size limit
        # 3. Add overlap between chunks
        pass

    def chunk_with_metadata(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Chunk text and add metadata about position.

        Returns:
            List of dicts with keys: text, chunk_index, start_char, source
        """
        # TODO: Implement this method
        pass


# =============================================================================
# TASK 3: Maximum Marginal Relevance (MMR)
# =============================================================================


def mmr_retrieval(
    query_embedding: List[float],
    document_embeddings: List[List[float]],
    documents: List[str],
    k: int = 3,
    lambda_mult: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Retrieve documents using Maximum Marginal Relevance.

    MMR balances relevance to query with diversity among results.

    Formula: MMR = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected))

    Args:
        query_embedding: Query vector
        document_embeddings: Document vectors
        documents: Document texts
        k: Number of results to return
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        List of (document, score) tuples

    Example:
        >>> # With lambda=1.0, behaves like standard similarity search
        >>> # With lambda=0.5, balances relevance and diversity
        >>> results = mmr_retrieval(query, embeds, docs, k=3, lambda_mult=0.5)
    """
    # TODO: Implement this function
    # 1. Start with no selected documents
    # 2. For k iterations:
    #    - For each candidate, calculate MMR score
    #    - Select document with highest MMR score
    #    - Add to selected set
    # 3. Return selected documents with scores
    pass


# =============================================================================
# TASK 4: Hybrid Search (Dense + Sparse)
# =============================================================================


class HybridRetriever:
    """Combine dense (semantic) and sparse (keyword) search."""

    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid retriever.

        Args:
            alpha: Weight for semantic search (1-alpha for keyword search)
        """
        self.alpha = alpha
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self._tfidf_matrix = None
        self._vocabulary = {}

    def add_documents(
        self, documents: List[str], embeddings: List[List[float]]
    ) -> None:
        """Add documents with their embeddings."""
        # TODO: Implement this method
        # Store documents and embeddings
        # Build TF-IDF representation for keyword search
        pass

    def search(
        self, query: str, query_embedding: List[float], k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search.

        Args:
            query: Query text (for keyword matching)
            query_embedding: Query embedding (for semantic matching)
            k: Number of results

        Returns:
            List of (document, combined_score) tuples

        Example:
            >>> retriever = HybridRetriever(alpha=0.7)  # 70% semantic, 30% keyword
            >>> results = retriever.search("RAG systems", query_emb, k=5)
        """
        # TODO: Implement this method
        # 1. Calculate semantic scores (cosine similarity)
        # 2. Calculate keyword scores (TF-IDF similarity)
        # 3. Combine: alpha * semantic + (1-alpha) * keyword
        # 4. Return top-k
        pass


# =============================================================================
# TASK 5: Query Expansion
# =============================================================================


def expand_query(query: str, synonyms: Dict[str, List[str]] = None) -> List[str]:
    """
    Expand a query with synonyms and variations.

    Args:
        query: Original query
        synonyms: Dictionary of word -> synonyms

    Returns:
        List of expanded queries

    Example:
        >>> synonyms = {"fast": ["quick", "rapid"], "big": ["large", "huge"]}
        >>> expand_query("fast car", synonyms)
        >>> # Returns ["fast car", "quick car", "rapid car"]
    """
    # TODO: Implement this function
    # 1. Start with original query
    # 2. For each word in query, if it has synonyms:
    #    - Create new queries with synonym substitutions
    # 3. Return all query variations
    pass


# =============================================================================
# TASK 6: Re-ranking
# =============================================================================


def rerank_by_keyword_overlap(
    query: str,
    documents: List[str],
    initial_scores: List[float],
    boost_factor: float = 0.2,
) -> List[Tuple[str, float]]:
    """
    Re-rank documents based on keyword overlap with query.

    Args:
        query: Query string
        documents: Retrieved documents
        initial_scores: Initial similarity scores
        boost_factor: How much to boost keyword matches (0-1)

    Returns:
        Re-ranked list of (document, score) tuples

    Example:
        >>> docs = ["RAG is great", "Vector databases", "RAG with vectors"]
        >>> scores = [0.8, 0.75, 0.7]
        >>> reranked = rerank_by_keyword_overlap("RAG", docs, scores)
        >>> # Documents mentioning "RAG" get score boost
    """
    # TODO: Implement this function
    # 1. Extract keywords from query
    # 2. For each document:
    #    - Count keyword matches
    #    - Calculate boost
    #    - Add boost to original score
    # 3. Sort by new scores
    pass


# =============================================================================
# TASK 7: Context Window Manager
# =============================================================================


class ContextWindowManager:
    """Manage context to fit within token limits."""

    def __init__(self, max_tokens: int = 4000, avg_chars_per_token: float = 4.0):
        """
        Initialize context manager.

        Args:
            max_tokens: Maximum tokens for context
            avg_chars_per_token: Approximate characters per token
        """
        self.max_tokens = max_tokens
        self.avg_chars_per_token = avg_chars_per_token
        self.max_chars = int(max_tokens * avg_chars_per_token)

    def fit_context(self, documents: List[str], scores: List[float]) -> List[str]:
        """
        Select documents to fit within token limit.

        Prioritizes higher-scored documents.

        Args:
            documents: Retrieved documents
            scores: Relevance scores

        Returns:
            Documents that fit within token limit

        Example:
            >>> manager = ContextWindowManager(max_tokens=1000)
            >>> context = manager.fit_context(long_docs, scores)
        """
        # TODO: Implement this method
        # 1. Sort documents by score
        # 2. Add documents until limit reached
        # 3. Return selected documents
        pass

    def truncate_document(self, document: str, max_chars: int) -> str:
        """Truncate document to fit character limit, preserving sentences."""
        # TODO: Implement this method
        # Truncate at sentence boundary if possible
        pass


# =============================================================================
# TASK 8: Retrieval Evaluator
# =============================================================================


class RetrievalEvaluator:
    """Evaluate retrieval quality."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        """
        Calculate precision at k.

        Precision@k = |relevant ∩ retrieved@k| / k

        Args:
            retrieved: List of retrieved documents
            relevant: Set of relevant document identifiers
            k: Cutoff for evaluation

        Returns:
            Precision score (0.0 to 1.0)
        """
        # TODO: Implement this method
        pass

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        """
        Calculate recall at k.

        Recall@k = |relevant ∩ retrieved@k| / |relevant|
        """
        # TODO: Implement this method
        pass

    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: set) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR = 1 / rank_of_first_relevant_doc
        """
        # TODO: Implement this method
        pass

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str], relevance_scores: Dict[str, float], k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        Args:
            retrieved: List of retrieved documents
            relevance_scores: Dict mapping doc -> relevance score
            k: Cutoff

        Returns:
            NDCG score (0.0 to 1.0)
        """
        # TODO: Implement this method
        # DCG = sum(rel_i / log2(i+1))
        # IDCG = DCG of ideal ranking
        # NDCG = DCG / IDCG
        pass


# =============================================================================
# TASK 9: Retrieval Pipeline
# =============================================================================


class RetrievalPipeline:
    """Complete retrieval pipeline with multiple stages."""

    def __init__(
        self,
        vector_store: SimpleVectorStore,
        embedder=None,
        use_mmr: bool = True,
        use_reranking: bool = True,
        top_k: int = 10,
        final_k: int = 3,
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            vector_store: Vector store for initial retrieval
            embedder: Embedding model (with encode method)
            use_mmr: Whether to apply MMR for diversity
            use_reranking: Whether to apply keyword reranking
            top_k: Initial retrieval count
            final_k: Final number of results
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_mmr = use_mmr
        self.use_reranking = use_reranking
        self.top_k = top_k
        self.final_k = final_k

    def retrieve(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Run the full retrieval pipeline.

        Steps:
        1. Embed query
        2. Initial retrieval from vector store
        3. (Optional) Apply MMR for diversity
        4. (Optional) Apply keyword reranking
        5. Return top results

        Args:
            query: Query string
            filters: Optional metadata filters

        Returns:
            List of RetrievedDocument objects
        """
        # TODO: Implement this method
        pass


# =============================================================================
# TASK 10: Caching Retriever
# =============================================================================


class CachingRetriever:
    """Retriever with query caching for repeated queries."""

    def __init__(
        self,
        retriever: Callable[[str], List[RetrievedDocument]],
        max_cache_size: int = 100,
    ):
        """
        Initialize caching retriever.

        Args:
            retriever: Function that takes query and returns documents
            max_cache_size: Maximum number of cached queries
        """
        self.retriever = retriever
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, List[RetrievedDocument]] = {}
        self.cache_order: List[str] = []  # LRU tracking

    def retrieve(self, query: str) -> Tuple[List[RetrievedDocument], bool]:
        """
        Retrieve documents, using cache if available.

        Args:
            query: Query string

        Returns:
            Tuple of (documents, was_cached)
        """
        # TODO: Implement this method
        # 1. Check cache
        # 2. If cached, return results and update LRU
        # 3. If not cached, call retriever
        # 4. Cache results (evict oldest if needed)
        # 5. Return results
        pass

    def clear_cache(self) -> None:
        """Clear all cached queries."""
        # TODO: Implement this method
        pass


# =============================================================================
# MAIN - Test your implementations
# =============================================================================

if __name__ == "__main__":
    print("Week 8 - Exercise Advanced 3: Retrieval Strategies")
    print("=" * 60)

    # Test SimpleVectorStore
    print("\n1. Simple Vector Store:")
    store = SimpleVectorStore()
    store.add(
        documents=["RAG is great", "Vectors are useful", "LLMs are powerful"],
        embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ids=["doc1", "doc2", "doc3"],
        metadatas=[{"topic": "rag"}, {"topic": "vectors"}, {"topic": "llm"}],
    )
    results = store.query([0.9, 0.1, 0], n_results=2)
    if results:
        print(f"Top result: {results[0].content} (score: {results[0].score:.3f})")

    # Test chunking
    print("\n2. Text Chunking:")
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    long_text = "This is a long document. " * 20
    chunks = chunker.chunk(long_text)
    if chunks:
        print(f"Created {len(chunks)} chunks")

    # Test MMR
    print("\n3. MMR Retrieval:")
    query_emb = [1, 0, 0]
    doc_embs = [[0.9, 0.1, 0], [0.85, 0.15, 0], [0.1, 0.9, 0]]
    docs = ["Similar A", "Similar B", "Different"]
    mmr_results = mmr_retrieval(query_emb, doc_embs, docs, k=2, lambda_mult=0.5)
    if mmr_results:
        print(f"MMR selected: {[d for d, s in mmr_results]}")
