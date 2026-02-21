# Week 8 - Retrieval Augmented Generation (RAG)

> Build AI systems that ground responses in your own data

## References

- [LangChain RAG Guide](https://python.langchain.com/docs/tutorials/rag/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Sentence Transformers](https://www.sbert.net/)

## Installation

```bash
# Core dependencies
pip install chromadb sentence-transformers
pip install tiktoken langchain langchain-community

# LLM providers
pip install google-generativeai openai

# Document processing
pip install pypdf python-docx beautifulsoup4

# Optional: Pinecone for production
pip install pinecone-client
```

## Why RAG?

Large Language Models have limitations:

- **Knowledge cutoff**: Training data has a date limit
- **Hallucinations**: Can generate plausible but false information
- **No private data**: Can't access your documents or databases
- **Token limits**: Can't process entire document collections

**RAG solves these problems** by:

1. Storing your documents in a searchable database
2. Retrieving relevant context for each query
3. Grounding LLM responses in your actual data

---

## Concepts

### 🔹 RAG Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ Documents│──▶│ Chunking │──▶│Embeddings│──▶│   Vector Store   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘ │
│                                                        │            │
│                                                        ▼            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ Response │◀──│   LLM    │◀──│ Context  │◀──│   Retrieval      │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘ │
│                                                        ▲            │
│                                                        │            │
│                                               ┌────────┴───────┐    │
│                                               │   User Query   │    │
│                                               └────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

**Key Components**:

- **Document Loading**: Ingest various file formats (PDF, TXT, HTML, etc.)
- **Chunking**: Split documents into processable pieces
- **Embeddings**: Convert text to numerical vectors
- **Vector Store**: Database for similarity search
- **Retrieval**: Find relevant chunks for a query
- **Generation**: LLM creates response using retrieved context

---

### 🔹 Document Loading & Processing

Different document types require different loaders:

```python
from pathlib import Path

def load_text_file(file_path: str) -> str:
    """Load plain text file."""
    return Path(file_path).read_text(encoding='utf-8')

def load_pdf(file_path: str) -> str:
    """Load PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

def load_webpage(url: str) -> str:
    """Load webpage content."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    return soup.get_text(separator='\n', strip=True)
```

📝 **Exercise: Basic 1** - Implement document loaders in [exercises/exercise_basic_1_documents.py](exercises/exercise_basic_1_documents.py)

---

### 🔹 Text Chunking Strategies

Chunking is **critical** for RAG quality. Too large = irrelevant noise. Too small = lost context.

**Strategy 1: Fixed-Size Chunks**

```python
def chunk_by_size(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.5:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks
```

**Strategy 2: Recursive Character Splitting**

```python
def recursive_split(
    text: str,
    chunk_size: int = 500,
    separators: list = ["\n\n", "\n", ". ", " "]
) -> list:
    """Recursively split text at natural boundaries."""

    if len(text) <= chunk_size:
        return [text]

    # Try each separator
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""

            for part in parts:
                if len(current) + len(part) + len(sep) <= chunk_size:
                    current += part + sep
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part + sep

            if current:
                chunks.append(current.strip())

            return chunks

    # Fallback to character split
    return chunk_by_size(text, chunk_size)
```

**Strategy 3: Semantic Chunking**

```python
def semantic_chunk(sentences: list, embedder, threshold: float = 0.7) -> list:
    """Group sentences by semantic similarity."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    embeddings = embedder.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            [current_embedding], [embeddings[i]]
        )[0][0]

        if similarity >= threshold:
            current_chunk.append(sentences[i])
            # Update centroid
            current_embedding = np.mean(
                embeddings[:len(current_chunk)], axis=0
            )
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_embedding = embeddings[i]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

| Strategy   | Best For        | Trade-offs                 |
| ---------- | --------------- | -------------------------- |
| Fixed-size | Simple docs     | May split mid-sentence     |
| Recursive  | Structured docs | Requires tuning separators |
| Semantic   | Complex docs    | Computationally expensive  |

---

### 🔹 Embeddings: Text to Vectors

Embeddings convert text into numerical vectors where **similar meanings are close together**.

**Popular Embedding Models**:

| Model                  | Dimensions | Speed  | Quality   | Use Case        |
| ---------------------- | ---------- | ------ | --------- | --------------- |
| all-MiniLM-L6-v2       | 384        | Fast   | Good      | General purpose |
| all-mpnet-base-v2      | 768        | Medium | Better    | High quality    |
| text-embedding-ada-002 | 1536       | API    | Excellent | Production      |
| BGE-large              | 1024       | Slow   | Excellent | Multilingual    |

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads on first use)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed single text
text = "Retrieval Augmented Generation improves LLM accuracy"
embedding = model.encode(text)
print(f"Shape: {embedding.shape}")  # (384,)

# Embed multiple texts
texts = [
    "RAG combines retrieval with generation",
    "Vector databases store embeddings",
    "The weather is sunny today"
]
embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)
```

**Similarity Search**:

```python
from sklearn.metrics.pairwise import cosine_similarity

query = "How does RAG work?"
query_embedding = model.encode([query])

# Find most similar
similarities = cosine_similarity(query_embedding, embeddings)[0]
most_similar_idx = similarities.argmax()
print(f"Most similar: {texts[most_similar_idx]}")
print(f"Score: {similarities[most_similar_idx]:.3f}")
```

📝 **Exercise: Intermediate 2** - Implement embedding and similarity in [exercises/exercise_intermediate_2_embeddings.py](exercises/exercise_intermediate_2_embeddings.py)

---

### 🔹 Vector Stores

Vector stores are specialized databases optimized for similarity search.

**ChromaDB** (Local, easy to use):

```python
import chromadb
from chromadb.utils import embedding_functions

# Create client
client = chromadb.Client()  # In-memory
# client = chromadb.PersistentClient(path="/path/to/db")  # Persistent

# Embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.create_collection(
    name="my_documents",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["Doc 1 content", "Doc 2 content", "Doc 3 content"],
    metadatas=[{"source": "file1.txt"}, {"source": "file2.txt"}, {"source": "file3.txt"}],
    ids=["doc1", "doc2", "doc3"]
)

# Query
results = collection.query(
    query_texts=["What is RAG?"],
    n_results=2
)
print(results['documents'])  # Most similar documents
print(results['distances'])  # Similarity scores
```

**Key Operations**:

```python
# Update documents
collection.update(
    ids=["doc1"],
    documents=["Updated content"]
)

# Delete documents
collection.delete(ids=["doc3"])

# Get all documents
all_docs = collection.get()

# Filter by metadata
filtered = collection.query(
    query_texts=["machine learning"],
    where={"source": "textbook.pdf"},
    n_results=5
)
```

---

### 🔹 Retrieval Strategies

Different retrieval strategies for different needs:

**1. Basic Similarity Search**

```python
def basic_retrieval(collection, query: str, k: int = 3):
    """Simple top-k retrieval."""
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results['documents'][0]
```

**2. Maximum Marginal Relevance (MMR)**

```python
def mmr_retrieval(
    collection,
    query: str,
    k: int = 3,
    lambda_mult: float = 0.5
) -> list:
    """Retrieve diverse, relevant results."""
    # Get more candidates than needed
    results = collection.query(
        query_texts=[query],
        n_results=k * 3,
        include=['embeddings', 'documents']
    )

    if not results['documents'][0]:
        return []

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    docs = results['documents'][0]
    embeddings = np.array(results['embeddings'][0])

    # Get query embedding
    query_emb = collection._embedding_function([query])[0]

    selected = []
    selected_embs = []

    for _ in range(min(k, len(docs))):
        best_score = -float('inf')
        best_idx = -1

        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            if doc in selected:
                continue

            # Relevance to query
            relevance = cosine_similarity([query_emb], [emb])[0][0]

            # Diversity from selected
            if selected_embs:
                max_sim = max(
                    cosine_similarity([emb], selected_embs)[0]
                )
                diversity = 1 - max_sim
            else:
                diversity = 1

            # MMR score
            score = lambda_mult * relevance + (1 - lambda_mult) * diversity

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected.append(docs[best_idx])
            selected_embs.append(embeddings[best_idx])

    return selected
```

**3. Hybrid Search (Dense + Sparse)**

```python
def hybrid_search(
    collection,
    query: str,
    documents: list,
    alpha: float = 0.5
) -> list:
    """Combine semantic and keyword search."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Semantic scores
    semantic_results = collection.query(
        query_texts=[query],
        n_results=len(documents)
    )

    # Keyword scores (BM25/TF-IDF)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    keyword_scores = cosine_similarity(
        tfidf_matrix[-1:], tfidf_matrix[:-1]
    )[0]

    # Combine scores
    combined = []
    for i, doc in enumerate(documents):
        semantic_score = 1 - semantic_results['distances'][0][i]
        keyword_score = keyword_scores[i]
        combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
        combined.append((doc, combined_score))

    # Sort by combined score
    combined.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in combined]
```

📝 **Exercise: Advanced 3** - Implement retrieval strategies in [exercises/exercise_advanced_3_retrieval.py](exercises/exercise_advanced_3_retrieval.py)

---

### 🔹 Building the RAG Pipeline

Putting it all together:

```python
import google.generativeai as genai
import chromadb
from pathlib import Path

class RAGPipeline:
    """Complete RAG pipeline."""

    def __init__(self, collection_name: str = "rag_docs"):
        # Initialize vector store
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize LLM
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def ingest(self, documents: list[dict]) -> None:
        """Ingest documents into vector store.

        Each document: {"text": str, "source": str, "id": str}
        """
        self.collection.add(
            documents=[d["text"] for d in documents],
            metadatas=[{"source": d["source"]} for d in documents],
            ids=[d["id"] for d in documents]
        )

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0]

    def generate(self, query: str, context: list[str]) -> str:
        """Generate response with context."""
        context_text = "\n\n---\n\n".join(context)

        prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context_text}

Question: {query}

Answer:"""

        response = self.model.generate_content(prompt)
        return response.text

    def query(self, question: str, k: int = 3) -> dict:
        """Full RAG query pipeline."""
        # Retrieve
        context = self.retrieve(question, k)

        # Generate
        answer = self.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": context
        }
```

---

### 🔹 Advanced RAG Techniques

**1. Query Rewriting**

```python
def rewrite_query(model, original_query: str) -> list[str]:
    """Generate alternative queries for better retrieval."""
    prompt = f"""Generate 3 alternative versions of this question
to improve search results. Return only the questions, one per line.

Original: {original_query}"""

    response = model.generate_content(prompt)
    queries = [original_query] + response.text.strip().split('\n')
    return queries[:4]  # Original + 3 alternatives
```

**2. Self-Query (Metadata Filtering)**

```python
def extract_filters(model, query: str) -> dict:
    """Extract metadata filters from natural language query."""
    prompt = f"""Extract search filters from this query.
Return JSON with keys: topic, date_range, source_type

Query: {query}

JSON:"""

    response = model.generate_content(prompt)
    # Parse JSON from response
    import json
    return json.loads(response.text)
```

**3. Contextual Compression**

```python
def compress_context(model, query: str, documents: list[str]) -> list[str]:
    """Extract only relevant parts from retrieved documents."""
    compressed = []

    for doc in documents:
        prompt = f"""Extract only the sentences relevant to this question.
If nothing is relevant, return "NOT_RELEVANT".

Question: {query}

Document: {doc}

Relevant sentences:"""

        response = model.generate_content(prompt)
        if "NOT_RELEVANT" not in response.text:
            compressed.append(response.text)

    return compressed
```

**4. Re-ranking**

```python
def rerank_results(
    model,
    query: str,
    documents: list[str],
    top_k: int = 3
) -> list[str]:
    """Use LLM to rerank retrieved documents."""
    prompt = f"""Rank these documents by relevance to the question.
Return document numbers in order of relevance (most relevant first).

Question: {query}

Documents:
{chr(10).join(f'{i+1}. {doc[:200]}...' for i, doc in enumerate(documents))}

Ranking (numbers only):"""

    response = model.generate_content(prompt)

    # Parse ranking
    import re
    numbers = re.findall(r'\d+', response.text)
    indices = [int(n) - 1 for n in numbers if int(n) <= len(documents)]

    return [documents[i] for i in indices[:top_k]]
```

---

### 🔹 Evaluation Metrics

**1. Retrieval Metrics**

```python
def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Precision of top-k results."""
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in top_k if doc in relevant)
    return relevant_retrieved / k

def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Recall of top-k results."""
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in top_k if doc in relevant)
    return relevant_retrieved / len(relevant) if relevant else 0

def mrr(relevant: set, retrieved: list) -> float:
    """Mean Reciprocal Rank."""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (i + 1)
    return 0
```

**2. Generation Metrics**

```python
def answer_relevancy(model, question: str, answer: str) -> float:
    """Check if answer addresses the question."""
    prompt = f"""Rate how well this answer addresses the question.
Score from 0.0 to 1.0.

Question: {question}
Answer: {answer}

Score:"""

    response = model.generate_content(prompt)
    import re
    match = re.search(r'0?\.\d+|1\.0', response.text)
    return float(match.group()) if match else 0.5

def faithfulness(model, context: list[str], answer: str) -> float:
    """Check if answer is grounded in context."""
    context_text = "\n".join(context)

    prompt = f"""Is this answer fully supported by the context?
Score from 0.0 (not supported) to 1.0 (fully supported).

Context: {context_text}

Answer: {answer}

Score:"""

    response = model.generate_content(prompt)
    import re
    match = re.search(r'0?\.\d+|1\.0', response.text)
    return float(match.group()) if match else 0.5
```

---

## 🛠️ Weekly Project

Build a **Document Q&A System** that can:

1. Ingest documents from multiple sources (PDF, TXT, web)
2. Chunk documents intelligently
3. Store in a vector database
4. Retrieve relevant context
5. Generate accurate, grounded answers
6. Cite sources in responses

See [exercises/project_pipeline.py](exercises/project_pipeline.py) for the complete project.

---

## Interview Questions

1. **What is RAG and why is it important?**
   - RAG grounds LLM responses in actual data, reducing hallucinations and enabling access to private/current information without fine-tuning.

2. **How do embeddings enable semantic search?**
   - Embeddings convert text to vectors where similar meanings are close in vector space. Similarity search finds semantically related content even without keyword matches.

3. **What chunking strategies would you use for different document types?**
   - Fixed-size for homogeneous docs, recursive for structured (code, markdown), semantic for complex narratives. Consider overlap to preserve context.

4. **How do you handle the trade-off between chunk size and retrieval quality?**
   - Smaller chunks = more precise but less context. Larger = more context but more noise. Use metadata to link related chunks, or hierarchical chunking.

5. **What is Maximum Marginal Relevance (MMR)?**
   - MMR balances relevance and diversity in retrieval. It prevents returning redundant results by penalizing similarity to already-selected documents.

6. **How would you evaluate a RAG system?**
   - Retrieval: Precision@k, Recall@k, MRR. Generation: Answer relevancy, faithfulness/groundedness, citation accuracy.

7. **What are common failure modes in RAG systems?**
   - Poor chunking (losing context), embedding model mismatch (domain-specific vocab), retrieval failures (no relevant docs), hallucination despite context.

8. **How do you handle documents that exceed the context window?**
   - Hierarchical summarization, map-reduce generation, iterative refinement, or selecting only the most relevant chunks.

9. **When would you use hybrid search vs pure semantic search?**
   - Hybrid when exact terms matter (product SKUs, legal citations, code). Pure semantic for conceptual queries where synonyms/paraphrases are important.

10. **How do you handle multi-hop questions requiring information from multiple documents?**
    - Query decomposition, iterative retrieval, chain-of-thought reasoning, or using an agent to plan retrieval steps.

---

## Takeaway Checklist

After completing this week, you should be able to:

- [ ] Explain the RAG architecture and its components
- [ ] Load and process documents from various sources
- [ ] Implement different chunking strategies
- [ ] Create and use text embeddings
- [ ] Work with vector databases (ChromaDB)
- [ ] Implement similarity search
- [ ] Use retrieval strategies (basic, MMR, hybrid)
- [ ] Build an end-to-end RAG pipeline
- [ ] Apply advanced techniques (reranking, compression)
- [ ] Evaluate RAG system performance

---

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Begin Week 9](../week-9/README.md)**
