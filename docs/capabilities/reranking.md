# Reranking Models

## Overview

Reranking models improve search relevance by re-scoring retrieved documents against a query. Unlike embeddings which compute similarity in vector space, rerankers directly score query-document pairs for more accurate relevance assessment.

## Common Use Cases

- **Search Refinement**: Re-order initial retrieval results for better accuracy
- **Retrieval-Augmented Generation (RAG)**: Select most relevant context for LLMs
- **Question Answering**: Rank candidate passages by relevance to question
- **Document Filtering**: Score and filter documents by relevance threshold

## Interface

### Creating a Reranker

```python
from esperanto.factory import AIFactory

# Basic usage
reranker = AIFactory.create_reranker(
    provider="jina",
    model_name="jina-reranker-v2-base-multilingual"
)

# With configuration
reranker = AIFactory.create_reranker(
    provider="transformers",
    model_name="BAAI/bge-reranker-base",
    config={
        "timeout": 60.0
    }
)
```

### Core Methods

#### `rerank(query, documents, top_k=None)`

Synchronous reranking of documents against a query.

```python
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence",
    "The weather is nice today",
    "Python is a programming language used in ML"
]

response = reranker.rerank(query, documents, top_k=2)

# Results are sorted by relevance (highest first)
for result in response.results:
    print(f"Score: {result.relevance_score:.4f} - {result.document}")
```

#### `arerank(query, documents, top_k=None)`

Asynchronous reranking (identical interface to `rerank`).

```python
response = await reranker.arerank(query, documents, top_k=2)
```

## Parameters

### Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | Required | The search query |
| `documents` | list[str] | Required | List of documents to rerank |
| `top_k` | int | None | Return only top K results (None = all) |

### Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | float | 60.0 | Request timeout in seconds |

## Response Structure

All reranker providers return standardized `RerankResponse` objects:

```python
response = reranker.rerank(query, documents, top_k=3)

# Results are pre-sorted by relevance score (highest first)
response.results[0].document           # Most relevant document text
response.results[0].relevance_score    # Normalized score (0-1)
response.results[0].index              # Original position in input list

# Metadata
response.model                         # Model name used

# Iterate results
for result in response.results:
    print(f"[{result.index}] Score: {result.relevance_score:.4f}")
    print(f"  {result.document[:100]}...")
```

### Relevance Scores

- **Range**: 0.0 to 1.0 (normalized across providers)
- **Interpretation**: Higher scores indicate greater relevance
- **Ordering**: Results are always sorted by score (descending)
- **Comparison**: Scores are relative within a single rerank call

## Provider Selection

→ **See [Provider Comparison](../providers/README.md)** for detailed comparison and selection guide.

### Quick Provider Guide

- **Jina**: Strong multilingual support, API-based, good performance
- **Voyage**: Specialized for retrieval, excellent accuracy
- **DashScope**: Alibaba Cloud Qwen & GTE models, good for Chinese/Multilingual
- **Transformers**: Local deployment, privacy-focused, various model choices

## Examples

### Basic Reranking

```python
from esperanto.factory import AIFactory

reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

query = "How to train neural networks?"
documents = [
    "Neural networks learn through backpropagation and gradient descent",
    "The weather forecast predicts rain tomorrow",
    "Training requires labeled data and computational resources",
    "Cooking pasta takes about 10 minutes"
]

response = reranker.rerank(query, documents, top_k=2)

print("Most relevant documents:")
for i, result in enumerate(response.results, 1):
    print(f"{i}. Score: {result.relevance_score:.4f}")
    print(f"   {result.document}\n")
```

### RAG Pipeline Integration

```python
from esperanto.factory import AIFactory

# Step 1: Initial retrieval with embeddings
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")
query = "What causes climate change?"

# Assume we retrieved 20 candidate documents
candidates = [...]  # Initial retrieval results

# Step 2: Rerank to get best matches
reranker = AIFactory.create_reranker("voyage", "rerank-1")
reranked = reranker.rerank(query, candidates, top_k=5)

# Step 3: Use top results as context for LLM
top_docs = [r.document for r in reranked.results]
context = "\n\n".join(top_docs)

model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
messages = [{
    "role": "user",
    "content": f"Context:\n{context}\n\nQuestion: {query}"
}]

answer = model.chat_complete(messages)
print(answer.content)
```

### Async Batch Processing

```python
reranker = AIFactory.create_reranker("transformers", "BAAI/bge-reranker-base")

queries = [
    "machine learning algorithms",
    "natural language processing",
    "computer vision applications"
]

document_sets = [
    [...],  # Documents for query 1
    [...],  # Documents for query 2
    [...]   # Documents for query 3
]

# Process multiple queries asynchronously
import asyncio

async def rerank_all():
    tasks = [
        reranker.arerank(q, docs, top_k=3)
        for q, docs in zip(queries, document_sets)
    ]
    return await asyncio.gather(*tasks)

results = await rerank_all()
```

### Score Filtering

```python
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

query = "Python web frameworks"
documents = [
    "Django is a high-level Python web framework",
    "Flask is a lightweight Python framework",
    "React is a JavaScript library",
    "FastAPI is a modern Python framework",
    "The ocean is very deep"
]

response = reranker.rerank(query, documents)

# Filter by relevance threshold
threshold = 0.5
relevant_docs = [
    r for r in response.results
    if r.relevance_score >= threshold
]

print(f"Found {len(relevant_docs)} relevant documents:")
for doc in relevant_docs:
    print(f"  [{doc.relevance_score:.4f}] {doc.document}")
```

### Comparing Providers

```python
from esperanto.factory import AIFactory

query = "quantum computing applications"
documents = [
    "Quantum computers use qubits for computation",
    "Classical computers use binary bits",
    "Quantum algorithms can solve certain problems faster"
]

providers = [
    ("jina", "jina-reranker-v2-base-multilingual"),
    ("voyage", "rerank-1"),
    ("transformers", "BAAI/bge-reranker-base")
]

for provider, model in providers:
    reranker = AIFactory.create_reranker(provider, model)
    response = reranker.rerank(query, documents)

    print(f"\n{provider} ({model}):")
    for r in response.results:
        print(f"  [{r.index}] Score: {r.relevance_score:.4f}")
```

## Best Practices

### When to Use Reranking

**Use reranking when:**
- You have an initial candidate set from embedding-based retrieval
- Precision is more important than speed
- You need to score specific query-document pairs

**Don't use reranking when:**
- You need to search millions of documents (use embeddings first)
- Real-time response is critical (reranking adds latency)
- You only have a single document to evaluate

### Optimal Workflow

```python
# 1. Cast wide net with embeddings (fast, approximate)
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")
# ... retrieve top 50-100 candidates

# 2. Refine with reranker (slower, accurate)
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")
final_results = reranker.rerank(query, candidates, top_k=5)

# 3. Use top results
```

### Performance Considerations

- **Batch size**: Rerankers process query-document pairs, so 100 documents = 100 pairs
- **Latency**: Expect 100-500ms for small batches, more for larger sets
- **Local vs API**: Transformers provider runs locally (privacy, no costs), API providers are faster

## Advanced Topics

- **Transformers Advanced Features**: [docs/advanced/transformers-features.md](../advanced/transformers-features.md)
- **Timeout Configuration**: [docs/advanced/timeout-configuration.md](../advanced/timeout-configuration.md)
- **Resource Management**: [docs/advanced/connection-resource-management.md](../advanced/connection-resource-management.md)

## See Also

- [Provider Setup Guides](../providers/README.md)
- [Embedding Models](./embedding.md)
- [Language Models](./llm.md)
