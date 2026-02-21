"""
Week 8 - Exercise Advanced 3: Retrieval Strategies - SOLUTIONS
===============================================================
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field

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
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


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
    """A simple in-memory vector store implementation."""

    def __init__(self):
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
        """Add documents to the vector store."""
        n = len(documents)

        # Generate IDs if not provided
        if ids is None:
            start_id = len(self.ids)
            ids = [f"doc_{start_id + i}" for i in range(n)]

        # Default metadata
        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.ids.extend(ids)
        self.metadata.extend(metadatas)

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """Query the vector store for similar documents."""
        candidates = []

        for i, (doc, emb, meta, doc_id) in enumerate(
            zip(self.documents, self.embeddings, self.metadata, self.ids)
        ):
            # Filter by metadata
            if where:
                match = all(meta.get(k) == v for k, v in where.items())
                if not match:
                    continue

            # Calculate similarity
            sim = cosine_similarity(query_embedding, emb)
            candidates.append(
                RetrievedDocument(content=doc, score=sim, metadata=meta, source=doc_id)
            )

        # Sort by score (descending)
        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates[:n_results]

    def delete(self, ids: List[str]) -> int:
        """Delete documents by ID."""
        deleted = 0
        indices_to_remove = []

        for i, doc_id in enumerate(self.ids):
            if doc_id in ids:
                indices_to_remove.append(i)
                deleted += 1

        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.documents[i]
            del self.embeddings[i]
            del self.metadata[i]
            del self.ids[i]

        return deleted

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Update documents by ID."""
        updated = 0

        for j, target_id in enumerate(ids):
            for i, doc_id in enumerate(self.ids):
                if doc_id == target_id:
                    if documents and j < len(documents):
                        self.documents[i] = documents[j]
                    if embeddings and j < len(embeddings):
                        self.embeddings[i] = embeddings[j]
                    if metadatas and j < len(metadatas):
                        self.metadata[i] = metadatas[j]
                    updated += 1
                    break

        return updated


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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # Find best break point
            chunk = text[start:end]
            best_break = len(chunk)

            for sep in self.separators:
                idx = chunk.rfind(sep)
                if idx > self.chunk_size * 0.3:
                    best_break = idx + len(sep)
                    break

            chunks.append(text[start : start + best_break].strip())
            start = start + best_break - self.chunk_overlap

        return [c for c in chunks if c]

    def chunk_with_metadata(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Chunk text and add metadata about position."""
        chunks = self.chunk(text)
        result = []

        position = 0
        for i, chunk in enumerate(chunks):
            start_char = text.find(chunk, position)
            if start_char == -1:
                start_char = position

            result.append(
                {
                    "text": chunk,
                    "chunk_index": i,
                    "start_char": start_char,
                    "source": source,
                }
            )
            position = start_char + 1

        return result


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
    """Retrieve documents using Maximum Marginal Relevance."""
    if not documents:
        return []

    # Calculate query similarities
    query_sims = [
        cosine_similarity(query_embedding, emb) for emb in document_embeddings
    ]

    selected = []
    selected_embeddings = []
    remaining = list(range(len(documents)))

    for _ in range(min(k, len(documents))):
        best_idx = -1
        best_score = -float("inf")

        for i in remaining:
            # Relevance to query
            relevance = query_sims[i]

            # Diversity from selected
            if selected_embeddings:
                max_sim = max(
                    cosine_similarity(document_embeddings[i], sel_emb)
                    for sel_emb in selected_embeddings
                )
            else:
                max_sim = 0

            # MMR score
            score = lambda_mult * relevance - (1 - lambda_mult) * max_sim

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected.append((documents[best_idx], query_sims[best_idx]))
            selected_embeddings.append(document_embeddings[best_idx])
            remaining.remove(best_idx)

    return selected


# =============================================================================
# TASK 4: Hybrid Search
# =============================================================================


class HybridRetriever:
    """Combine dense (semantic) and sparse (keyword) search."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self._tfidf_vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    def add_documents(
        self, documents: List[str], embeddings: List[List[float]]
    ) -> None:
        """Add documents with their embeddings."""
        self.documents = documents
        self.embeddings = embeddings

        # Build TF-IDF vocabulary
        doc_term_sets = []
        all_terms = set()

        for doc in documents:
            terms = set(doc.lower().split())
            doc_term_sets.append(terms)
            all_terms.update(terms)

        self._tfidf_vocab = {term: i for i, term in enumerate(sorted(all_terms))}

        n_docs = len(documents)
        for term in self._tfidf_vocab:
            doc_freq = sum(1 for ts in doc_term_sets if term in ts)
            self._idf[term] = math.log(n_docs / (doc_freq + 1)) + 1

    def _tfidf_similarity(self, query: str, document: str) -> float:
        """Calculate TF-IDF similarity."""
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        doc_term_set = set(doc_terms)

        score = 0
        for term in query_terms:
            if term in doc_term_set:
                tf = doc_terms.count(term) / len(doc_terms)
                idf = self._idf.get(term, 1)
                score += tf * idf

        return score

    def search(
        self, query: str, query_embedding: List[float], k: int = 3
    ) -> List[Tuple[str, float]]:
        """Perform hybrid search."""
        results = []

        for doc, emb in zip(self.documents, self.embeddings):
            semantic_score = cosine_similarity(query_embedding, emb)
            keyword_score = self._tfidf_similarity(query, doc)

            # Normalize keyword score
            keyword_score = min(1.0, keyword_score / 2)

            combined = self.alpha * semantic_score + (1 - self.alpha) * keyword_score
            results.append((doc, combined))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# =============================================================================
# TASK 5: Query Expansion
# =============================================================================


def expand_query(query: str, synonyms: Dict[str, List[str]] = None) -> List[str]:
    """Expand a query with synonyms and variations."""
    if synonyms is None:
        synonyms = {}

    queries = [query]
    words = query.split()

    for i, word in enumerate(words):
        lower_word = word.lower()
        if lower_word in synonyms:
            for syn in synonyms[lower_word]:
                new_words = words.copy()
                new_words[i] = syn
                queries.append(" ".join(new_words))

    return queries


# =============================================================================
# TASK 6: Re-ranking
# =============================================================================


def rerank_by_keyword_overlap(
    query: str,
    documents: List[str],
    initial_scores: List[float],
    boost_factor: float = 0.2,
) -> List[Tuple[str, float]]:
    """Re-rank documents based on keyword overlap with query."""
    query_words = set(query.lower().split())
    results = []

    for doc, score in zip(documents, initial_scores):
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)

        # Calculate boost
        boost = boost_factor * (overlap / len(query_words)) if query_words else 0
        new_score = score + boost

        results.append((doc, new_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# TASK 7: Context Window Manager
# =============================================================================


class ContextWindowManager:
    """Manage context to fit within token limits."""

    def __init__(self, max_tokens: int = 4000, avg_chars_per_token: float = 4.0):
        self.max_tokens = max_tokens
        self.avg_chars_per_token = avg_chars_per_token
        self.max_chars = int(max_tokens * avg_chars_per_token)

    def fit_context(self, documents: List[str], scores: List[float]) -> List[str]:
        """Select documents to fit within token limit."""
        # Sort by score
        sorted_pairs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        selected = []
        total_chars = 0

        for doc, score in sorted_pairs:
            if total_chars + len(doc) <= self.max_chars:
                selected.append(doc)
                total_chars += len(doc)
            else:
                # Try to fit truncated version
                remaining = self.max_chars - total_chars
                if remaining > 100:
                    truncated = self.truncate_document(doc, remaining)
                    if truncated:
                        selected.append(truncated)
                break

        return selected

    def truncate_document(self, document: str, max_chars: int) -> str:
        """Truncate document at sentence boundary."""
        if len(document) <= max_chars:
            return document

        truncated = document[:max_chars]

        # Find last sentence boundary
        for sep in [". ", "! ", "? "]:
            idx = truncated.rfind(sep)
            if idx > max_chars * 0.5:
                return truncated[: idx + 1].strip()

        # Fallback: truncate at word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.5:
            return truncated[:last_space].strip() + "..."

        return truncated.strip() + "..."


# =============================================================================
# TASK 8: Retrieval Evaluator
# =============================================================================


class RetrievalEvaluator:
    """Evaluate retrieval quality."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        """Calculate precision at k."""
        top_k = retrieved[:k]
        relevant_found = sum(1 for doc in top_k if doc in relevant)
        return relevant_found / k if k > 0 else 0

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        """Calculate recall at k."""
        top_k = retrieved[:k]
        relevant_found = sum(1 for doc in top_k if doc in relevant)
        return relevant_found / len(relevant) if relevant else 0

    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: set) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1 / (i + 1)
        return 0

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str], relevance_scores: Dict[str, float], k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # DCG
        dcg = 0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc, 0)
            dcg += rel / math.log2(i + 2)

        # IDCG (ideal ranking)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0


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
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_mmr = use_mmr
        self.use_reranking = use_reranking
        self.top_k = top_k
        self.final_k = final_k

    def retrieve(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """Run the full retrieval pipeline."""
        # Embed query
        if self.embedder and hasattr(self.embedder, "encode"):
            query_emb = list(self.embedder.encode([query])[0])
        else:
            query_emb = [0.0] * 384

        # Initial retrieval
        results = self.vector_store.query(query_emb, self.top_k, filters)

        if not results:
            return []

        # MMR re-ranking
        if self.use_mmr and len(results) > self.final_k:
            docs = [r.content for r in results]
            # Get embeddings from store
            embs = []
            for r in results:
                idx = (
                    self.vector_store.ids.index(r.source)
                    if r.source in self.vector_store.ids
                    else -1
                )
                if idx >= 0:
                    embs.append(self.vector_store.embeddings[idx])
                else:
                    embs.append([0.0] * len(query_emb))

            mmr_results = mmr_retrieval(query_emb, embs, docs, self.final_k)
            selected_docs = [d for d, s in mmr_results]
            results = [r for r in results if r.content in selected_docs][: self.final_k]

        # Keyword reranking
        if self.use_reranking and results:
            docs = [r.content for r in results]
            scores = [r.score for r in results]
            reranked = rerank_by_keyword_overlap(query, docs, scores)

            # Update scores
            for i, (doc, new_score) in enumerate(reranked):
                for r in results:
                    if r.content == doc:
                        r.score = new_score
                        break

            results.sort(key=lambda r: r.score, reverse=True)

        return results[: self.final_k]


# =============================================================================
# TASK 10: Caching Retriever
# =============================================================================


class CachingRetriever:
    """Retriever with query caching."""

    def __init__(
        self,
        retriever: Callable[[str], List[RetrievedDocument]],
        max_cache_size: int = 100,
    ):
        self.retriever = retriever
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, List[RetrievedDocument]] = {}
        self.cache_order: List[str] = []

    def retrieve(self, query: str) -> Tuple[List[RetrievedDocument], bool]:
        """Retrieve documents, using cache if available."""
        if query in self.cache:
            # Move to end (LRU)
            self.cache_order.remove(query)
            self.cache_order.append(query)
            return self.cache[query], True

        # Call retriever
        results = self.retriever(query)

        # Cache results
        if len(self.cache) >= self.max_cache_size:
            # Evict oldest
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        self.cache[query] = results
        self.cache_order.append(query)

        return results, False

    def clear_cache(self) -> None:
        """Clear all cached queries."""
        self.cache.clear()
        self.cache_order.clear()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Week 8 - Exercise Advanced 3: Retrieval - SOLUTIONS")
    print("=" * 60)

    # Test Vector Store
    print("\n1. Vector Store:")
    store = SimpleVectorStore()
    store.add(
        documents=["RAG is great", "Vectors are useful", "LLMs are powerful"],
        embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ids=["d1", "d2", "d3"],
    )
    results = store.query([0.9, 0.1, 0], n_results=2)
    for r in results:
        print(f"  {r.content}: {r.score:.3f}")

    # Test MMR
    print("\n2. MMR Retrieval:")
    mmr = mmr_retrieval(
        [1, 0],
        [[1, 0], [0.99, 0.1], [0, 1]],
        ["doc1", "doc2", "doc3"],
        k=2,
        lambda_mult=0.5,
    )
    print(f"  Selected: {[d for d, s in mmr]}")

    # Test Evaluator
    print("\n3. Evaluation:")
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    print(f"  P@4: {RetrievalEvaluator.precision_at_k(retrieved, relevant, 4)}")
    print(f"  R@4: {RetrievalEvaluator.recall_at_k(retrieved, relevant, 4)}")
    print(f"  MRR: {RetrievalEvaluator.mean_reciprocal_rank(retrieved, relevant)}")
