# Jina

## Overview

Jina provides advanced embedding and reranking capabilities with native support for task optimization, late chunking, and output dimension control. Perfect for production RAG systems and search applications requiring cutting-edge performance.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ❌ | Not available |
| Embeddings | ✅ | Task-aware, late chunking, dimension control |
| Reranking | ✅ | Multilingual support (100+ languages) |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://jina.ai/embeddings / https://jina.ai/reranker

## Prerequisites

### Account Requirements
- Jina account (sign up at https://jina.ai)
- API key with credits

### Getting API Keys
1. Visit https://jina.ai/embeddings
2. Click "Get API Key" or sign in
3. Navigate to your account settings
4. Copy your API key

## Environment Variables

```bash
# Jina API key (required)
JINA_API_KEY="jina_..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`JINA_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Embedding model
embedder = AIFactory.create_embedding("jina", "jina-embeddings-v3")

# Reranker model
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")
```

### Direct Instantiation

```python
from esperanto.providers.embedding.jina import JinaEmbeddingModel
from esperanto.providers.reranker.jina import JinaRerankerModel

# Embedding model
embedder = JinaEmbeddingModel(
    api_key="your-api-key",
    model_name="jina-embeddings-v3"
)

# Reranker model
reranker = JinaRerankerModel(
    api_key="your-api-key",
    model_name="jina-reranker-v2-base-multilingual"
)
```

## Capabilities

### Embeddings

**Available Models:**

| Model | Dimensions | Context | Best For |
|-------|------------|---------|----------|
| **jina-embeddings-v3** | 1024 | 8192 | Production, multilingual (default) |
| **jina-embeddings-v4** | 1024 | 8192 | Latest, multimodal support |
| **jina-clip-v2** | 512 | N/A | Text + image embeddings |

**Configuration:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,  # Native API support
        "late_chunking": True,                              # Native API support
        "output_dimensions": 512,                           # Native API support
        "truncate_at_max_length": True,
        "timeout": 60.0
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
model = AIFactory.create_embedding("jina", "jina-embeddings-v3")

# Generate embeddings
texts = ["Hello, world!", "Another text"]
response = model.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
```

**Example - Task-Optimized Embeddings (Native API):**

Jina has **native API support** for task types - not emulation!

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Optimize for search queries (native Jina API)
query_model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

# Optimize for document storage (native Jina API)
document_model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

# Generate optimized embeddings
query_embedding = query_model.embed(["search query"])
doc_embeddings = document_model.embed(["document 1", "document 2"])
```

**Task Type Mappings (Native Jina API):**

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# These map directly to Jina API parameters
EmbeddingTaskType.RETRIEVAL_QUERY       → "retrieval.query"
EmbeddingTaskType.RETRIEVAL_DOCUMENT    → "retrieval.passage"
EmbeddingTaskType.CLASSIFICATION        → "classification"
EmbeddingTaskType.CLUSTERING            → "separation"
EmbeddingTaskType.SIMILARITY            → "text-matching"
EmbeddingTaskType.CODE_RETRIEVAL        → "code.query"
```

**Example - Late Chunking (Native API):**

Perfect for long documents:

```python
# Native late chunking support in Jina API
model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "late_chunking": True,
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)

# Handles documents > 8K tokens intelligently
long_doc = "Very long document content..." * 1000
embeddings = model.embed([long_doc])  # No information loss
```

**Example - Output Dimension Control (Native API):**

```python
# Reduce dimensions for faster search (native Jina API)
model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "output_dimensions": 512,  # Reduce from 1024 to 512
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)

embeddings = model.embed(["Text"])
print(len(embeddings[0]))  # 512 dimensions
# Faster similarity search, lower storage costs
```

**Example - RAG Pipeline Optimization:**

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Create specialized models for RAG
query_embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "output_dimensions": 512  # Faster search
    }
)

doc_embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,      # Handle long documents
        "output_dimensions": 512    # Match query dimensions
    }
)

# Embed query
query = "What is machine learning?"
query_embedding = query_embedder.embed([query])

# Embed documents
documents = ["Long document 1...", "Long document 2..."]
doc_embeddings = doc_embedder.embed(documents)
```

**Example - Multilingual Embeddings:**

```python
# Jina supports 100+ languages
model = AIFactory.create_embedding("jina", "jina-embeddings-v3")

texts = [
    "Hello, world!",           # English
    "Bonjour le monde!",       # French
    "Hola, mundo!",            # Spanish
    "こんにちは世界!",           # Japanese
    "مرحبا بالعالم",          # Arabic
    "Привет, мир!"             # Russian
]

response = model.embed(texts)
print(f"Generated {len(response.data)} multilingual embeddings")
```

**Example - Code Search:**

```python
# Optimize for code retrieval
code_model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.CODE_RETRIEVAL}
)

code_snippets = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "class UserManager: def __init__(self): self.users = []",
    "async function fetchData() { return await fetch(url); }"
]

embeddings = code_model.embed(code_snippets)
```

### Reranking

**Available Models:**

| Model | Languages | Best For |
|-------|-----------|----------|
| **jina-reranker-v2-base-multilingual** | 100+ | Production, multilingual (default) |
| **jina-reranker-v1-base-en** | English | English-only, optimized |

**Configuration:**

```python
from esperanto.factory import AIFactory

reranker = AIFactory.create_reranker(
    "jina",
    "jina-reranker-v2-base-multilingual",
    config={
        "timeout": 30.0
    }
)
```

**Example - Basic Reranking:**

```python
from esperanto.factory import AIFactory

# Create reranker
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

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

**Example - Multilingual Reranking:**

```python
# Use multilingual model for international queries
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

# German query
query = "Was ist maschinelles Lernen?"
documents = [
    "Maschinelles Lernen ist ein Teilbereich der künstlichen Intelligenz.",
    "Das Wetter wird morgen regnerisch.",
    "Python ist eine beliebte Programmiersprache für maschinelles Lernen."
]

results = reranker.rerank(query, documents)
print(f"Top result: {results.results[0].document}")
```

**Example - English-Only Optimization:**

```python
# Use English-only model for better performance on English texts
reranker = AIFactory.create_reranker("jina", "jina-reranker-v1-base-en")

query = "artificial intelligence applications"
documents = [
    "AI is used in healthcare for diagnosis and treatment planning.",
    "The stock market fluctuates daily based on various factors.",
    "Machine learning algorithms power recommendation systems.",
    "Sports teams use data analytics to improve performance.",
    "Natural language processing enables chatbots and virtual assistants."
]

results = reranker.rerank(query, documents, top_k=3)
```

**Example - RAG System Integration:**

```python
# Complete RAG pipeline with Jina
from esperanto.factory import AIFactory

# Step 1: Create embedder
embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "output_dimensions": 512
    }
)

# Step 2: Create reranker
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

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
    reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")
    results = await reranker.arerank(query, documents, top_k=3)
    return results

# Run async reranking
results = asyncio.run(rerank_async())
```

## Advanced Features

### Native Task Optimization

Jina embeddings have native API support for task types:

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# All these task types are sent directly to Jina's API
task_configs = [
    ("retrieval.query", EmbeddingTaskType.RETRIEVAL_QUERY),
    ("retrieval.passage", EmbeddingTaskType.RETRIEVAL_DOCUMENT),
    ("classification", EmbeddingTaskType.CLASSIFICATION),
    ("separation", EmbeddingTaskType.CLUSTERING),
    ("text-matching", EmbeddingTaskType.SIMILARITY),
    ("code.query", EmbeddingTaskType.CODE_RETRIEVAL)
]

for jina_task, esperanto_task in task_configs:
    model = AIFactory.create_embedding(
        "jina",
        "jina-embeddings-v3",
        config={"task_type": esperanto_task}
    )
    # Jina API receives the appropriate task parameter
```

### Native Late Chunking

Jina's API natively supports late chunking:

```python
# No emulation - sent directly to Jina API
model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "late_chunking": True,
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)

# Intelligently handles documents > 8K tokens
documents = [
    "Very long document 1..." * 2000,
    "Very long document 2..." * 2000
]

embeddings = model.embed(documents)
```

### Native Dimension Control

Jina API natively supports dimension reduction:

```python
# Dimension reduction without quality loss (native Jina)
dimensions = [1024, 768, 512, 256]

for dim in dimensions:
    model = AIFactory.create_embedding(
        "jina",
        "jina-embeddings-v3",
        config={"output_dimensions": dim}
    )
    embeddings = model.embed(["Test text"])
    print(f"{dim} dimensions: {len(embeddings[0])} values")
```

### Combined Advanced Features

```python
# Use all advanced features together
model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,           # Handle long documents
        "output_dimensions": 512,        # Optimize for speed
        "truncate_at_max_length": True   # Graceful handling
    }
)

# Process long documents with optimal settings
long_documents = ["Document 1..." * 1000, "Document 2..." * 1000]
embeddings = model.embed(long_documents)
```

### Timeout Configuration

Customize request timeouts:

```python
# Embedding with custom timeout
embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={"timeout": 120.0}  # 2 minutes
)

# Reranker with custom timeout
reranker = AIFactory.create_reranker(
    "jina",
    "jina-reranker-v2-base-multilingual",
    config={"timeout": 60.0}  # 1 minute
)
```

### LangChain Integration

Convert to LangChain models:

```python
from esperanto.factory import AIFactory

# Embedding model
embedder = AIFactory.create_embedding("jina", "jina-embeddings-v3")
langchain_embedder = embedder.to_langchain()

# Use with LangChain
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(texts, langchain_embedder)

# Reranker model
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")
langchain_reranker = reranker.to_langchain()

# Use with LangChain compression
from langchain.schema import Document
docs = [Document(page_content=text) for text in texts]
compressed = langchain_reranker.compress_documents(docs, query)
```

## Performance Considerations

### Embedding Performance

**Throughput:**
- Fast API response times
- Batch processing supported
- Efficient for high-volume applications

**Quality:**
- State-of-the-art embedding quality
- 100+ language support
- Task-specific optimization

### Reranking Performance

**Speed:**
- Fast reranking inference
- Suitable for real-time applications
- Efficient batch processing

**Accuracy:**
- High-accuracy reranking
- Multilingual support (100+ languages)
- Production-ready quality

### Cost Optimization

```python
# Use output dimension control to reduce costs
model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "output_dimensions": 256,  # Smaller dimensions
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)

# Lower dimensions = faster processing = lower costs
# Test to find optimal dimension for your use case
```

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
**Solution:** Enable late chunking: `config={"late_chunking": True}`

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase timeout: `config={"timeout": 120.0}`

**Invalid Dimension:**
```
Error: Invalid output dimensions
```
**Solution:** Use supported dimensions (powers of 2, up to 1024).

### Best Practices

1. **Use Task Types:** Take advantage of native task optimization.

2. **Enable Late Chunking:** For documents that might exceed 8K tokens.

3. **Dimension Control:** Use lower dimensions for faster search when quality permits.

4. **Batch Processing:** Process multiple texts in batches for efficiency.

5. **Reranking Strategy:** Retrieve more documents (20-100), then rerank to top 3-10.

6. **Multilingual Support:** Use v2 reranker for multilingual, v1 for English-only.

7. **Monitor Usage:** Track your API usage and costs.

## Use Cases

### Production RAG Systems

```python
# High-performance RAG with native features
query_embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "output_dimensions": 512
    }
)

doc_embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,
        "output_dimensions": 512
    }
)

reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

# RAG pipeline: embed query → vector search → rerank → generate
```

### Multilingual Search Engine

```python
# Support 100+ languages
embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "output_dimensions": 768
    }
)

reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

# Handle queries in any language
# Automatically understand semantic meaning across languages
```

### Code Search Platform

```python
# Specialized for code search
code_embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.CODE_RETRIEVAL,
        "output_dimensions": 512
    }
)

# Embed code repositories
code_files = ["file1.py", "file2.js", "file3.go"]
embeddings = code_embedder.embed(code_files)

# Search with natural language
query = "function to calculate factorial"
query_embedding = code_embedder.embed([query])
```

### Document Classification

```python
# Optimize for classification tasks
classifier_embedder = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.CLASSIFICATION,
        "output_dimensions": 256
    }
)

# Classify documents by content
documents = ["Document 1...", "Document 2..."]
embeddings = classifier_embedder.embed(documents)
# Use embeddings for classification model
```

## See Also

- [Embeddings Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/embedding.md)
- [Reranking Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/rerank.md)
- [Embedding Providers](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/embedding/providers.md)
- [OpenAI Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/openai.md)
- [Transformers Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/transformers.md)
- [Voyage Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/voyage.md)
