# Week 13 - Vector Databases and Semantic Search

## Overview

Vector databases are the backbone of modern AI applications, enabling semantic search, RAG (Retrieval Augmented Generation), recommendation systems, and similarity matching at scale. This week, you'll master vector database fundamentals and integration with Python applications.

## References

- [Pinecone Documentation](https://docs.pinecone.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Sentence Transformers](https://www.sbert.net/)

## Installation

```bash
# Core vector database clients
uv add chromadb pinecone-client qdrant-client weaviate-client

# Embeddings and utilities
uv add sentence-transformers numpy faiss-cpu

# Optional: GPU support for FAISS
# uv add faiss-gpu

# For testing
uv add pytest pytest-asyncio
```

## Concepts

### 1. Vector Database Fundamentals

Vector databases store and index high-dimensional vectors (embeddings) for efficient similarity search. Unlike traditional databases optimized for exact matches, vector databases excel at finding "similar" items.

**Key Concepts:**

- **Embeddings**: Dense vector representations of data (text, images, etc.)
- **Similarity Metrics**: Cosine similarity, Euclidean distance, dot product
- **Approximate Nearest Neighbors (ANN)**: Algorithms for fast similarity search
- **Indexing**: HNSW, IVF, PQ for efficient retrieval

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["Hello world", "Machine learning is great"]
embeddings = model.encode(texts)

# Cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.4f}")
```

**Exercise 1** - [exercise_basic_1_vector_fundamentals.py](exercises/exercise_basic_1_vector_fundamentals.py): Implement core vector operations and similarity calculations.

---

### 2. ChromaDB: Local Vector Database

ChromaDB is a lightweight, embeddable vector database perfect for development and smaller applications. It supports persistent storage and metadata filtering.

**Features:**

- Embedded or client-server mode
- Automatic embedding generation
- Metadata filtering
- Collection management

```python
import chromadb
from chromadb.utils import embedding_functions

# Create client
client = chromadb.Client()  # In-memory
# client = chromadb.PersistentClient(path="./chroma_db")  # Persistent

# Use sentence transformers for embeddings
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.create_collection(
    name="documents",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["Python is great", "Machine learning rocks"],
    metadatas=[{"topic": "programming"}, {"topic": "ai"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["AI and programming"],
    n_results=2,
    where={"topic": {"$in": ["programming", "ai"]}}
)

print(results)
```

**Key Operations:**

- `add()`: Add documents with optional embeddings and metadata
- `query()`: Semantic search with filters
- `update()`: Update existing documents
- `delete()`: Remove documents by ID or filter
- `get()`: Retrieve documents by ID

**Exercise 2** - [exercise_intermediate_2_chromadb.py](exercises/exercise_intermediate_2_chromadb.py): Build a complete document search system with ChromaDB.

---

### 3. Pinecone: Cloud Vector Database

Pinecone is a managed cloud vector database designed for production workloads with high availability and scalability.

**Features:**

- Fully managed infrastructure
- Real-time index updates
- Metadata filtering
- Namespaces for multi-tenancy

```python
from pinecone import Pinecone, ServerlessSpec
import os

# Initialize client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create index
index_name = "my-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Match your embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Get index reference
index = pc.Index(index_name)

# Upsert vectors
index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1] * 384,
            "metadata": {"topic": "ai"}
        },
        {
            "id": "vec2",
            "values": [0.2] * 384,
            "metadata": {"topic": "ml"}
        }
    ],
    namespace="articles"
)

# Query
results = index.query(
    namespace="articles",
    vector=[0.1] * 384,
    top_k=5,
    include_metadata=True,
    filter={"topic": {"$eq": "ai"}}
)

print(results)
```

**Namespaces:**
Namespaces partition data within an index for multi-tenancy:

```python
# Different namespaces for different users/projects
index.upsert(vectors=[...], namespace="user-123")
index.upsert(vectors=[...], namespace="user-456")

# Query only user's data
results = index.query(vector=query_vec, namespace="user-123")
```

---

### 4. Qdrant: High-Performance Vector Search

Qdrant offers advanced filtering, payload indexing, and both cloud and self-hosted options.

**Features:**

- Rich filtering with nested conditions
- Payload (metadata) indexing for fast filtering
- Quantization for memory efficiency
- Batch operations

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, SearchParams
)

# Connect to Qdrant
client = QdrantClient(":memory:")  # In-memory for testing
# client = QdrantClient(url="http://localhost:6333")  # Server
# client = QdrantClient(url="https://xyz.qdrant.io", api_key="...")  # Cloud

# Create collection
client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# Insert points
client.upsert(
    collection_name="articles",
    points=[
        PointStruct(
            id=1,
            vector=[0.1] * 384,
            payload={"title": "AI Guide", "category": "tech", "year": 2024}
        ),
        PointStruct(
            id=2,
            vector=[0.2] * 384,
            payload={"title": "ML Basics", "category": "tech", "year": 2023}
        )
    ]
)

# Search with filtering
results = client.search(
    collection_name="articles",
    query_vector=[0.15] * 384,
    limit=5,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="tech")
            ),
            FieldCondition(
                key="year",
                range={"gte": 2023}
            )
        ]
    )
)

for result in results:
    print(f"{result.payload['title']}: {result.score:.4f}")
```

---

### 5. FAISS: Facebook AI Similarity Search

FAISS is a library for efficient similarity search optimized for dense vectors. It's not a database but a highly efficient indexing library.

**Index Types:**

- `IndexFlatL2`: Exact search (brute force)
- `IndexIVFFlat`: Inverted file index (faster, approximate)
- `IndexHNSWFlat`: Hierarchical Navigable Small World (fast, high recall)
- `IndexIVFPQ`: Product quantization (memory efficient)

```python
import faiss
import numpy as np

# Sample data
dimension = 384
num_vectors = 10000
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Flat index (exact search)
index_flat = faiss.IndexFlatL2(dimension)
index_flat.add(vectors)

# HNSW index (fast approximate search)
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
index_hnsw.add(vectors)

# IVF index (requires training)
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index_ivf.train(vectors)  # Train on sample data
index_ivf.add(vectors)
index_ivf.nprobe = 10  # Number of clusters to search

# Search
query = np.random.random((1, dimension)).astype('float32')
k = 5  # Number of neighbors

distances, indices = index_hnsw.search(query, k)
print(f"Nearest neighbors: {indices[0]}")
print(f"Distances: {distances[0]}")
```

**Persistence:**

```python
# Save index
faiss.write_index(index_hnsw, "vectors.index")

# Load index
loaded_index = faiss.read_index("vectors.index")
```

**Exercise 3** - [exercise_advanced_3_vector_search.py](exercises/exercise_advanced_3_vector_search.py): Build a hybrid search system combining multiple vector databases.

---

### 6. Embedding Strategies

Choosing and optimizing embeddings is crucial for vector search quality.

**Popular Embedding Models:**

| Model                  | Dimensions | Use Case              |
| ---------------------- | ---------- | --------------------- |
| all-MiniLM-L6-v2       | 384        | General purpose, fast |
| all-mpnet-base-v2      | 768        | Higher quality        |
| text-embedding-3-small | 1536       | OpenAI, high quality  |
| text-embedding-3-large | 3072       | OpenAI, best quality  |
| e5-large-v2            | 1024       | Document retrieval    |

```python
from sentence_transformers import SentenceTransformer
import openai

# Local models (Sentence Transformers)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world"], normalize_embeddings=True)

# OpenAI embeddings
client = openai.OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world"]
)
embedding = response.data[0].embedding
```

**Chunking Strategies:**

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size // 2:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks
```

---

### 7. Metadata and Filtering

Effective use of metadata enables powerful filtering and organization.

**Metadata Best Practices:**

- Store structured data (dates, categories, IDs) as metadata
- Use metadata for access control and multi-tenancy
- Index frequently filtered fields
- Keep metadata lightweight

```python
# ChromaDB metadata filtering
results = collection.query(
    query_texts=["AI research"],
    where={
        "$and": [
            {"category": {"$eq": "research"}},
            {"year": {"$gte": 2023}},
            {"$or": [
                {"department": "ai"},
                {"department": "ml"}
            ]}
        ]
    }
)

# Qdrant advanced filtering
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

filter = Filter(
    must=[
        FieldCondition(key="status", match=MatchValue(value="published")),
        FieldCondition(key="score", range=Range(gte=0.8))
    ],
    should=[
        FieldCondition(key="tags", match=MatchValue(value="featured"))
    ],
    must_not=[
        FieldCondition(key="archived", match=MatchValue(value=True))
    ]
)
```

---

### 8. RAG (Retrieval Augmented Generation)

Vector databases are essential for RAG pipelines that ground LLM responses in relevant context.

```python
from sentence_transformers import SentenceTransformer
import chromadb
import openai

class RAGPipeline:
    def __init__(self, collection_name: str = "knowledge_base"):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma = chromadb.Client()
        self.collection = self.chroma.create_collection(collection_name)
        self.llm = openai.OpenAI()

    def add_documents(self, documents: list[str], metadatas: list[dict] = None):
        """Add documents to the knowledge base."""
        embeddings = self.embedder.encode(documents).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0]

    def generate(self, query: str, k: int = 3) -> str:
        """Generate response with RAG."""
        # Retrieve context
        context_docs = self.retrieve(query, k)
        context = "\n\n".join(context_docs)

        # Generate with context
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"Answer based on this context:\n\n{context}"
                },
                {"role": "user", "content": query}
            ]
        )

        return response.choices[0].message.content

# Usage
rag = RAGPipeline()
rag.add_documents([
    "Python is a programming language created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence.",
    "Vector databases store embeddings for similarity search."
])

answer = rag.generate("What is Python?")
print(answer)
```

---

### 9. Performance Optimization

**Batch Operations:**

```python
# Instead of individual inserts
for doc in documents:
    collection.add(documents=[doc], ids=[doc.id])

# Use batch operations
collection.add(
    documents=[d.text for d in documents],
    ids=[d.id for d in documents],
    metadatas=[d.metadata for d in documents]
)
```

**Index Optimization:**

```python
# ChromaDB - specify index parameters
collection = client.create_collection(
    name="optimized",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:M": 48
    }
)

# FAISS - product quantization for memory
dimension = 384
nlist = 100
m = 8  # Number of subquantizers
nbits = 8  # Bits per subquantizer

quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
```

**Caching:**

```python
from functools import lru_cache

class CachedEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @lru_cache(maxsize=10000)
    def embed(self, text: str) -> tuple:
        """Cache embeddings as tuples (hashable)."""
        return tuple(self.model.encode([text])[0].tolist())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed with caching."""
        return [list(self.embed(t)) for t in texts]
```

---

### 10. Testing Vector Search

```python
import pytest
import numpy as np
from your_module import VectorSearch

class TestVectorSearch:
    @pytest.fixture
    def search_engine(self):
        return VectorSearch()

    def test_add_and_retrieve(self, search_engine):
        """Test basic add and search."""
        search_engine.add(["Python is great"], ids=["doc1"])
        results = search_engine.search("What is Python?", k=1)

        assert len(results) == 1
        assert results[0].id == "doc1"

    def test_similarity_ordering(self, search_engine):
        """Test that results are ordered by similarity."""
        search_engine.add([
            "Python programming language",
            "Java programming language",
            "Cooking recipes"
        ], ids=["py", "java", "cook"])

        results = search_engine.search("Python code", k=3)

        # Python should be most similar
        assert results[0].id == "py"
        # Cooking should be least similar
        assert results[-1].id == "cook"

    def test_metadata_filtering(self, search_engine):
        """Test metadata filtering."""
        search_engine.add(
            ["Doc 1", "Doc 2"],
            ids=["d1", "d2"],
            metadatas=[{"type": "a"}, {"type": "b"}]
        )

        results = search_engine.search(
            "document",
            filter={"type": "a"}
        )

        assert len(results) == 1
        assert results[0].id == "d1"

    def test_embedding_consistency(self, search_engine):
        """Test that same text produces same embedding."""
        text = "Test document"
        emb1 = search_engine.embed(text)
        emb2 = search_engine.embed(text)

        np.testing.assert_array_almost_equal(emb1, emb2)
```

---

## Weekly Project

**Semantic Document Search Engine** - [project_pipeline.py](exercises/project_pipeline.py)

Build a production-ready document search system with:

- Multi-backend support (ChromaDB, FAISS, in-memory)
- Document processing pipeline (chunking, embedding)
- Hybrid search (combining semantic and keyword search)
- Result reranking
- Query expansion
- Performance metrics and monitoring

---

## Interview Questions

1. **What is a vector database and how does it differ from traditional databases?**
   - Vector databases store high-dimensional vectors and optimize for similarity search using distance metrics (cosine, Euclidean), while traditional databases optimize for exact matches and structured queries.

2. **Explain the difference between exact and approximate nearest neighbor search.**
   - Exact search (brute force) compares query against all vectors for guaranteed best results but O(n) complexity. ANN uses indexing (HNSW, IVF) for sub-linear search with slight accuracy trade-off.

3. **What is HNSW and why is it popular for vector search?**
   - Hierarchical Navigable Small World graphs provide O(log n) search complexity with high recall by building a multi-layer graph structure for efficient navigation.

4. **How do you choose the right embedding model?**
   - Consider: use case (general vs. domain-specific), latency requirements, dimension (affects storage/speed), quality benchmarks (MTEB), and fine-tuning possibilities.

5. **What is the purpose of chunking in RAG systems?**
   - Chunking splits large documents into smaller pieces that fit embedding context windows, improve retrieval precision, and provide focused context to LLMs.

6. **Explain metadata filtering in vector databases.**
   - Metadata filtering combines vector similarity with structured attribute filters (e.g., date ranges, categories) to narrow results before or after vector search.

7. **How do you handle updates in a vector database?**
   - Most vector DBs support upsert (insert or update by ID). For immutable indexes like FAISS, maintain ID mappings and periodically rebuild indexes.

8. **What are namespaces in Pinecone and when would you use them?**
   - Namespaces partition data within an index for multi-tenancy, allowing separate data spaces without creating multiple indexes.

9. **How do you evaluate vector search quality?**
   - Metrics: Recall@K, MRR (Mean Reciprocal Rank), nDCG (normalized Discounted Cumulative Gain), latency percentiles, and A/B testing with user feedback.

10. **What is product quantization and when is it useful?**
    - PQ compresses vectors by splitting into subvectors and quantizing each, reducing memory by 4-64x with ~5% recall loss. Useful for large-scale deployments.

---

## Takeaways Checklist

- [ ] Understand vector embeddings and similarity metrics
- [ ] Know when to use different vector database options
- [ ] Can implement ChromaDB for local development
- [ ] Understand Pinecone for production cloud deployments
- [ ] Can use Qdrant's advanced filtering capabilities
- [ ] Understand FAISS indexing strategies
- [ ] Know how to choose and optimize embedding models
- [ ] Can implement effective chunking strategies
- [ ] Understand RAG pipeline architecture
- [ ] Know performance optimization techniques
- [ ] Can test vector search systems effectively

---

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Begin Week 14](../week-14/README.md)**
