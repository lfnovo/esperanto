# Voyage

## Overview

Voyage provides specialized embeddings and reranking optimized for retrieval tasks, with strong performance on code search and domain-specific applications.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ❌ | Not available |
| Embeddings | ✅ | voyage-3, voyage-code-2, voyage-law-2 |
| Reranking | ✅ | rerank-2, rerank-1 |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://docs.voyageai.com

## Prerequisites

### Account Requirements
- Voyage AI account (sign up at https://www.voyageai.com)
- API key with credits

### Getting API Keys
1. Visit https://dash.voyageai.com
2. Navigate to API Keys section
3. Click "Create new API key"
4. Copy and store the key securely

## Environment Variables

```bash
# Voyage API key (required)
VOYAGE_API_KEY="pa-..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`VOYAGE_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Embedding model
embedder = AIFactory.create_embedding("voyage", "voyage-3")

# Reranker model
reranker = AIFactory.create_reranker("voyage", "rerank-2")
```

### Direct Instantiation

```python
from esperanto.providers.embedding.voyage import VoyageEmbeddingModel
from esperanto.providers.reranker.voyage import VoyageRerankerModel

# Embedding model
embedder = VoyageEmbeddingModel(
    api_key="your-api-key",
    model_name="voyage-3"
)

# Reranker model
reranker = VoyageRerankerModel(
    api_key="your-api-key",
    model_name="rerank-2"
)
```

## Capabilities

### Embeddings

**Available Models:**

| Model | Dimensions | Context | Best For |
|-------|------------|---------|----------|
| **voyage-3** | 1024 | 32K | Latest, general purpose (default) |
| **voyage-2** | 1024 | 16K | Previous generation |
| **voyage-code-2** | 1536 | 16K | Code search and understanding |
| **voyage-law-2** | 1024 | 16K | Legal domain specialization |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_embedding(
    "voyage",
    "voyage-3",
    config={
        "timeout": 60.0
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
model = AIFactory.create_embedding("voyage", "voyage-3")

# Generate embeddings
texts = ["Hello, world!", "Another text"]
response = model.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
```

**Example - General Purpose Search:**

```python
# Use voyage-3 for general retrieval tasks
model = AIFactory.create_embedding("voyage", "voyage-3")

# Embed documents for search
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
    "Deep learning uses neural networks.",
    "Natural language processing enables text understanding."
]

doc_embeddings = model.embed(documents)

# Embed query
query = "What is machine learning?"
query_embedding = model.embed([query])

# Use embeddings for similarity search
```

**Example - Code Search:**

```python
# Use voyage-code-2 for code retrieval
code_model = AIFactory.create_embedding("voyage", "voyage-code-2")

# Embed code snippets
code_snippets = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "class UserManager: def __init__(self): self.users = []",
    "async function fetchData() { return await fetch(url); }",
    "public class Calculator { public int add(int a, int b) { return a + b; } }"
]

code_embeddings = code_model.embed(code_snippets)

# Search with natural language
query = "function to calculate fibonacci sequence"
query_embedding = code_model.embed([query])
```

**Example - Legal Domain:**

```python
# Use voyage-law-2 for legal documents
legal_model = AIFactory.create_embedding("voyage", "voyage-law-2")

# Embed legal texts
legal_docs = [
    "The parties agree to arbitration under the rules of...",
    "This contract is governed by the laws of...",
    "Indemnification clause: The party agrees to hold harmless...",
    "Termination: Either party may terminate with 30 days notice..."
]

legal_embeddings = legal_model.embed(legal_docs)

# Search legal documents
query = "contract termination conditions"
query_embedding = legal_model.embed([query])
```

**Example - Large Context:**

```python
# voyage-3 supports up to 32K tokens
model = AIFactory.create_embedding("voyage", "voyage-3")

# Handle very long documents
long_document = "..." * 5000  # Large document (up to 32K tokens)

embeddings = model.embed([long_document])
print(f"Embedded document with {len(long_document)} characters")
```

**Example - Batch Processing:**

```python
# Process large batches efficiently
model = AIFactory.create_embedding("voyage", "voyage-3")

# Large corpus
documents = [f"Document {i} content..." for i in range(1000)]

# Process in batches (Voyage handles batching)
response = model.embed(documents)
print(f"Processed {len(response.data)} embeddings")
```

**Example - Async Embeddings:**

```python
import asyncio

async def embed_async():
    model = AIFactory.create_embedding("voyage", "voyage-3")
    response = await model.aembed(texts)
    return response

# Run async embeddings
response = asyncio.run(embed_async())
```

### Reranking

**Available Models:**

| Model | Best For |
|-------|----------|
| **rerank-2** | Latest, highest accuracy (default) |
| **rerank-1** | Previous generation |

**Configuration:**

```python
from esperanto.factory import AIFactory

reranker = AIFactory.create_reranker(
    "voyage",
    "rerank-2",
    config={
        "timeout": 30.0
    }
)
```

**Example - Basic Reranking:**

```python
from esperanto.factory import AIFactory

# Create reranker
reranker = AIFactory.create_reranker("voyage", "rerank-2")

query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "The weather forecast shows rain tomorrow.",
    "Python is a popular programming language for machine learning.",
    "Deep learning uses neural networks with multiple layers.",
    "Coffee is best served hot in the morning."
]

# Rerank documents by relevance
results = reranker.rerank(query, documents, top_k=3)

# Results sorted by relevance (highest first)
for i, result in enumerate(results.results):
    print(f"{i+1}. Score: {result.relevance_score:.3f}")
    print(f"   Document: {result.document[:60]}...\n")
```

**Example - Code Search Reranking:**

```python
# Rerank code search results
reranker = AIFactory.create_reranker("voyage", "rerank-2")

query = "function to sort an array"
code_results = [
    "def bubble_sort(arr): for i in range(len(arr)): ...",
    "class DatabaseConnection: def __init__(self): ...",
    "def quicksort(arr): if len(arr) <= 1: return arr ...",
    "import numpy as np # Matrix operations",
    "def merge_sort(arr): if len(arr) > 1: ..."
]

results = reranker.rerank(query, code_results, top_k=3)
print("Most relevant code:")
for result in results.results:
    print(f"Score: {result.relevance_score:.3f}")
    print(f"Code: {result.document[:50]}...\n")
```

**Example - RAG Pipeline:**

```python
# Complete RAG pipeline with Voyage
from esperanto.factory import AIFactory

# Step 1: Create embedder
embedder = AIFactory.create_embedding("voyage", "voyage-3")

# Step 2: Create reranker
reranker = AIFactory.create_reranker("voyage", "rerank-2")

# Step 3: Query process
query = "How does photosynthesis work?"

# Initial retrieval (vector search)
query_embedding = embedder.embed([query])
initial_docs = vector_search(query_embedding, top_k=20)  # Your vector DB

# Rerank for precision
reranked = reranker.rerank(query, initial_docs, top_k=5)
final_docs = [r.document for r in reranked.results]

# Step 4: Use for generation
# Pass final_docs to your LLM for answer generation
```

**Example - Async Reranking:**

```python
import asyncio

async def rerank_async():
    reranker = AIFactory.create_reranker("voyage", "rerank-2")
    results = await reranker.arerank(query, documents, top_k=3)
    return results

# Run async reranking
results = asyncio.run(rerank_async())
```

**Example - Two-Stage Retrieval:**

```python
# Stage 1: Fast vector search with embeddings
embedder = AIFactory.create_embedding("voyage", "voyage-3")
query_embedding = embedder.embed(["user query"])
candidates = vector_search(query_embedding, top_k=100)  # Retrieve many

# Stage 2: Precise reranking
reranker = AIFactory.create_reranker("voyage", "rerank-2")
final_results = reranker.rerank("user query", candidates, top_k=10)

# Get top 10 most relevant documents
top_docs = [r.document for r in final_results.results]
```

## Advanced Features

### Domain-Specific Models

Voyage provides specialized models for specific domains:

```python
# Code search optimization
code_model = AIFactory.create_embedding("voyage", "voyage-code-2")
code_embeddings = code_model.embed(code_snippets)

# Legal domain optimization
legal_model = AIFactory.create_embedding("voyage", "voyage-law-2")
legal_embeddings = legal_model.embed(legal_documents)

# General purpose
general_model = AIFactory.create_embedding("voyage", "voyage-3")
general_embeddings = general_model.embed(general_texts)
```

### Large Context Window

Voyage-3 supports very large contexts:

```python
# Handle documents up to 32K tokens
model = AIFactory.create_embedding("voyage", "voyage-3")

# No chunking needed for most documents
very_long_doc = read_entire_document("large_file.txt")  # Up to 32K tokens
embedding = model.embed([very_long_doc])
```

### Timeout Configuration

Customize request timeouts:

```python
# Embedding with custom timeout
embedder = AIFactory.create_embedding(
    "voyage",
    "voyage-3",
    config={"timeout": 120.0}  # 2 minutes
)

# Reranker with custom timeout
reranker = AIFactory.create_reranker(
    "voyage",
    "rerank-2",
    config={"timeout": 60.0}  # 1 minute
)
```

### LangChain Integration

Convert to LangChain models:

```python
from esperanto.factory import AIFactory

# Embedding model
embedder = AIFactory.create_embedding("voyage", "voyage-3")
langchain_embedder = embedder.to_langchain()

# Use with LangChain
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(texts, langchain_embedder)

# Reranker model
reranker = AIFactory.create_reranker("voyage", "rerank-2")
langchain_reranker = reranker.to_langchain()

# Use with LangChain compression
from langchain.schema import Document
docs = [Document(page_content=text) for text in texts]
compressed = langchain_reranker.compress_documents(docs, query)
```

## Model Selection Guide

### Voyage-3
**Best for:** General-purpose retrieval, latest model
- 32K context window (largest)
- 1024 dimensions
- Best general performance
- Latest improvements

```python
model = AIFactory.create_embedding("voyage", "voyage-3")
```

### Voyage-Code-2
**Best for:** Code search and understanding
- Optimized for code retrieval
- Understands programming concepts
- 1536 dimensions (more detail)
- 16K context window

```python
code_model = AIFactory.create_embedding("voyage", "voyage-code-2")
```

### Voyage-Law-2
**Best for:** Legal documents and contracts
- Specialized for legal domain
- Understands legal terminology
- 1024 dimensions
- 16K context window

```python
legal_model = AIFactory.create_embedding("voyage", "voyage-law-2")
```

### Rerank-2
**Best for:** High-accuracy reranking
- Latest reranking model
- Best accuracy
- Production ready

```python
reranker = AIFactory.create_reranker("voyage", "rerank-2")
```

## Performance Characteristics

### Embedding Performance

**Context Windows:**
- Voyage-3: 32K tokens (industry-leading)
- Voyage-2: 16K tokens
- Voyage-Code-2: 16K tokens
- Voyage-Law-2: 16K tokens

**Dimensions:**
- Voyage-3: 1024
- Voyage-2: 1024
- Voyage-Code-2: 1536 (richer representations)
- Voyage-Law-2: 1024

**Throughput:**
- Fast API response times
- Efficient batch processing
- Scalable for production

### Reranking Performance

**Speed:**
- Fast inference
- Real-time applications
- Efficient for large candidate sets

**Accuracy:**
- State-of-the-art reranking quality
- Strong performance on diverse domains
- Production-ready

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key is correct and has credits.

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Check your rate limits and consider upgrading your plan.

**Context Length Error:**
```
Error: Input too long
```
**Solution:** Ensure your text is within model limits (32K for voyage-3, 16K for others).

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase timeout: `config={"timeout": 120.0}`

**Model Not Available:**
```
Error: Model not found
```
**Solution:** Verify model name is correct. Use "voyage-3", "voyage-code-2", etc.

### Best Practices

1. **Choose Right Model:** Use domain-specific models for specialized content.

2. **Large Context:** Take advantage of 32K context window with voyage-3.

3. **Batch Processing:** Process multiple texts in batches for efficiency.

4. **Reranking Strategy:** Retrieve 20-100 candidates, rerank to top 3-10.

5. **Code Search:** Use voyage-code-2 for code-related tasks.

6. **Legal Content:** Use voyage-law-2 for legal documents.

7. **Monitor Usage:** Track your API usage and costs.

## Use Cases

### General Search Engine

```python
# Build search with voyage-3
embedder = AIFactory.create_embedding("voyage", "voyage-3")
reranker = AIFactory.create_reranker("voyage", "rerank-2")

# Embed documents
documents = ["doc1", "doc2", "doc3", ...]
doc_embeddings = embedder.embed(documents)

# Store in vector database
# ...

# Search pipeline
query_embedding = embedder.embed(["user query"])
candidates = vector_db.search(query_embedding, top_k=50)
final_results = reranker.rerank("user query", candidates, top_k=10)
```

### Code Search Platform

```python
# Specialized code search
code_embedder = AIFactory.create_embedding("voyage", "voyage-code-2")
reranker = AIFactory.create_reranker("voyage", "rerank-2")

# Index code repositories
code_files = read_repository_files()
code_embeddings = code_embedder.embed(code_files)

# Search with natural language
query = "authentication middleware implementation"
query_embedding = code_embedder.embed([query])
code_matches = search_code(query_embedding)
ranked_matches = reranker.rerank(query, code_matches, top_k=5)
```

### Legal Document Retrieval

```python
# Legal document search
legal_embedder = AIFactory.create_embedding("voyage", "voyage-law-2")
reranker = AIFactory.create_reranker("voyage", "rerank-2")

# Index legal documents
legal_docs = read_legal_documents()
legal_embeddings = legal_embedder.embed(legal_docs)

# Search legal database
query = "force majeure clause interpretation"
query_embedding = legal_embedder.embed([query])
relevant_docs = search_legal_db(query_embedding)
ranked_docs = reranker.rerank(query, relevant_docs, top_k=10)
```

### Research Paper Search

```python
# Academic paper retrieval
embedder = AIFactory.create_embedding("voyage", "voyage-3")
reranker = AIFactory.create_reranker("voyage", "rerank-2")

# Handle long research papers (32K context)
papers = read_research_papers()  # Full paper abstracts and content
paper_embeddings = embedder.embed(papers)

# Search with complex queries
query = "transformer architectures for natural language understanding"
query_embedding = embedder.embed([query])
paper_matches = search_papers(query_embedding, top_k=50)
top_papers = reranker.rerank(query, paper_matches, top_k=10)
```

### Multi-Domain Application

```python
# Switch models based on content type
def get_embedder(content_type):
    if content_type == "code":
        return AIFactory.create_embedding("voyage", "voyage-code-2")
    elif content_type == "legal":
        return AIFactory.create_embedding("voyage", "voyage-law-2")
    else:
        return AIFactory.create_embedding("voyage", "voyage-3")

# Use appropriate model for each content type
code_embedder = get_embedder("code")
legal_embedder = get_embedder("legal")
general_embedder = get_embedder("general")
```

## Comparison with Other Providers

**vs OpenAI:**
- Larger context window (32K vs 8K)
- Domain-specific models
- Competitive pricing

**vs Jina:**
- No task type parameters (simpler API)
- Strong retrieval performance
- Domain specialization (code, legal)

**vs Transformers:**
- Cloud-based (no local setup)
- Managed service (no maintenance)
- Optimized models

## See Also

- [Embeddings Guide](../capabilities/embedding.md)
- [Reranking Guide](../capabilities/reranking.md)
- [Embedding Providers](../capabilities/embedding.md)
- [OpenAI Provider](./openai.md)
- [Jina Provider](./jina.md)
- [Transformers Provider](./transformers.md)
