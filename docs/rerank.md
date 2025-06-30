# Reranking Providers 🔄

Esperanto provides a universal interface for reranking providers, allowing you to improve search relevance and document ranking using different models while maintaining a consistent API.

## Overview

Reranking is a crucial component in modern search and RAG (Retrieval-Augmented Generation) systems. It takes an initial set of documents and reorders them based on their relevance to a given query, significantly improving the quality of search results.

### Key Benefits

- **Universal Interface**: Switch between providers without changing code
- **Score Normalization**: Consistent 0-1 relevance scores across all providers
- **Privacy Options**: Local reranking with Transformers for offline processing
- **Performance**: Cloud-based solutions for high-throughput applications
- **LangChain Integration**: Drop-in compatibility with existing workflows

## Supported Providers

| Provider | Models | Type | Key Features |
|----------|--------|------|--------------|
| **Jina** | jina-reranker-v2-base-multilingual, jina-reranker-v1-base-en | Cloud | Multilingual support, high accuracy |
| **Voyage** | rerank-2, rerank-1 | Cloud | Fast processing, good performance |
| **Transformers** | Qwen/Qwen3-Reranker-4B | Local | Privacy-first, offline processing |

## Quick Start

### Using AIFactory (Recommended)

```python
from esperanto.factory import AIFactory

# Create a reranker instance
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

# Rerank documents
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence that uses algorithms to learn from data.",
    "The weather forecast shows rain tomorrow.",
    "Python is a popular programming language for machine learning applications.",
    "Deep learning uses neural networks with multiple layers.",
    "Coffee is best served hot in the morning."
]

# Get top 3 most relevant documents
results = reranker.rerank(query, documents, top_k=3)

# Print results
for i, result in enumerate(results.results):
    print(f"{i+1}. Score: {result.relevance_score:.3f}")
    print(f"   Document: {result.document[:100]}...")
    print(f"   Original Index: {result.index}")
    print()
```

### Async Usage

```python
# Async reranking
results = await reranker.arerank(query, documents, top_k=3)
```

## Provider-Specific Usage

### Jina Reranker

```python
from esperanto.providers.reranker.jina import JinaRerankerModel

reranker = JinaRerankerModel(
    api_key="your-jina-api-key",  # Or set JINA_API_KEY env var
    model_name="jina-reranker-v2-base-multilingual",
    config={
        "return_documents": True  # Always enabled for consistency
    }
)

# Multilingual support
query = "Was ist maschinelles Lernen?"  # German
documents = [
    "Maschinelles Lernen ist ein Teilbereich der künstlichen Intelligenz.",
    "Das Wetter wird morgen regnerisch.",
    "Python ist eine beliebte Programmiersprache für maschinelles Lernen."
]

results = reranker.rerank(query, documents)
```

#### Available Models
- `jina-reranker-v2-base-multilingual`: Latest multilingual model (default)
- `jina-reranker-v1-base-en`: English-only model for better performance on English texts

### Voyage Reranker

```python
from esperanto.providers.reranker.voyage import VoyageRerankerModel

reranker = VoyageRerankerModel(
    api_key="your-voyage-api-key",  # Or set VOYAGE_API_KEY env var
    model_name="rerank-2",
    config={}
)

results = reranker.rerank(query, documents, top_k=5)
```

#### Available Models
- `rerank-2`: Latest Voyage reranking model (default)
- `rerank-1`: Previous generation model

### Transformers Reranker (Local)

```python
from esperanto.providers.reranker.transformers import TransformersRerankerModel

# Local reranker - no API key required
reranker = TransformersRerankerModel(
    model_name="Qwen/Qwen3-Reranker-4B",  # Default model
    config={
        "max_length": 512,          # Maximum sequence length
        "cache_dir": "./models"     # Model cache directory
    }
)

# Works offline - perfect for privacy-sensitive applications
results = reranker.rerank(query, documents)
```

#### Features
- **Privacy-First**: Completely offline processing
- **Auto-Device Detection**: Automatically uses GPU/MPS if available
- **Optimized Tokenization**: Efficient processing with proper attention masking
- **Memory Efficient**: Proper resource management

#### Requirements
```bash
pip install "esperanto[transformers]"
```

## Response Format

All providers return a standardized `RerankResponse` object:

```python
from esperanto.common_types.reranker import RerankResponse, RerankResult

# Response structure
response = reranker.rerank(query, documents)

# Response object
response.results          # List[RerankResult] - sorted by relevance (highest first)
response.model           # str - model name used
response.usage           # Optional[Usage] - token/request usage info

# Individual result
result = response.results[0]
result.index             # int - original document index
result.document          # str - original document text
result.relevance_score   # float - normalized 0-1 relevance score (1.0 = most relevant)
```

### Score Normalization

All providers normalize their scores to a 0-1 range using min-max normalization:
- **1.0**: Most relevant document
- **0.5**: Average relevance
- **0.0**: Least relevant document

This ensures consistency when switching between providers.

## Advanced Usage

### Batch Processing

```python
# Process multiple queries efficiently
queries = [
    "What is machine learning?",
    "How does Python work?",
    "What is artificial intelligence?"
]

# Same documents for all queries
documents = [
    "Machine learning is a subset of AI...",
    "Python is a programming language...",
    "Artificial intelligence involves...",
    "The weather is nice today...",
    "Coffee helps with productivity..."
]

results = []
for query in queries:
    result = reranker.rerank(query, documents, top_k=2)
    results.append(result)
```

### Custom Configuration

```python
# Provider-specific configurations
jina_reranker = AIFactory.create_reranker(
    "jina", 
    "jina-reranker-v2-base-multilingual",
    config={
        "timeout": 30,  # Custom timeout
    }
)

voyage_reranker = AIFactory.create_reranker(
    "voyage", 
    "rerank-2",
    config={
        "timeout": 30,  # Custom timeout
    }
)

transformers_reranker = AIFactory.create_reranker(
    "transformers", 
    "Qwen/Qwen3-Reranker-4B",
    config={
        "max_length": 1024,        # Longer sequences
        "cache_dir": "./models",   # Custom model cache
        "device": "cuda"           # Force specific device
    }
)
```

## LangChain Integration

Convert any reranker to a LangChain-compatible reranker:

```python
from langchain.schema import Document

# Create LangChain documents
documents = [
    Document(page_content="Machine learning is...", metadata={"source": "article1"}),
    Document(page_content="Python programming...", metadata={"source": "article2"}),
    Document(page_content="Weather forecast...", metadata={"source": "article3"})
]

# Convert to LangChain reranker
langchain_reranker = reranker.to_langchain()

# Use with LangChain
query = "What is machine learning?"
reranked_docs = langchain_reranker.compress_documents(documents, query)

# Results include relevance scores in metadata
for doc in reranked_docs:
    print(f"Score: {doc.metadata['relevance_score']:.3f}")
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Source: {doc.metadata['source']}")
    print()
```

## Error Handling

```python
from esperanto.providers.reranker.base import RerankerModel

try:
    results = reranker.rerank(query, documents, top_k=5)
except ValueError as e:
    # Input validation errors
    print(f"Invalid input: {e}")
except RuntimeError as e:
    # API or model errors
    print(f"Reranker error: {e}")
except Exception as e:
    # Other errors
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Cloud Providers (Jina, Voyage)
- **Fast Processing**: Low latency for real-time applications
- **Scalable**: Handle high request volumes
- **Network Dependency**: Requires internet connection
- **Cost**: Pay-per-request pricing

### Local Provider (Transformers)
- **Privacy**: Complete offline processing
- **No API Costs**: One-time model download
- **Higher Latency**: Especially on CPU
- **Memory Usage**: ~8GB VRAM for Qwen3-Reranker-4B

### Recommendations

**For Production APIs**: Use cloud providers (Jina/Voyage) for fast response times
**For Privacy-Sensitive Data**: Use Transformers provider for offline processing
**For Development**: Start with cloud providers, switch to local for privacy needs

## Model Comparison

| Provider | Model | Languages | Speed | Accuracy | Privacy |
|----------|-------|-----------|-------|----------|---------|
| Jina | jina-reranker-v2-base-multilingual | 100+ | Fast | High | Cloud |
| Jina | jina-reranker-v1-base-en | English | Fast | High | Cloud |
| Voyage | rerank-2 | Multilingual | Very Fast | High | Cloud |
| Voyage | rerank-1 | Multilingual | Fast | Good | Cloud |
| Transformers | Qwen/Qwen3-Reranker-4B | Multilingual | Medium | Very High | Local |

## Use Cases

### 1. RAG Systems
```python
# Improve retrieval quality in RAG pipelines
retrieved_docs = vector_search(query, top_k=20)  # Initial retrieval
reranked = reranker.rerank(query, retrieved_docs, top_k=5)  # Rerank top results
context = [r.document for r in reranked.results]  # Use for generation
```

### 2. Search Applications
```python
# Enhance search relevance
search_results = search_engine.search(query, limit=50)
reranked = reranker.rerank(query, search_results, top_k=10)
return reranked.results  # Return top 10 most relevant
```

### 3. Document Filtering
```python
# Filter relevant documents from a large corpus
all_documents = load_document_corpus()
relevant_docs = reranker.rerank(query, all_documents, top_k=20)
high_relevance = [r for r in relevant_docs.results if r.relevance_score > 0.7]
```

## Environment Variables

```bash
# Jina
export JINA_API_KEY="your-jina-api-key"

# Voyage
export VOYAGE_API_KEY="your-voyage-api-key"

# Transformers (optional)
export TRANSFORMERS_CACHE="/path/to/model/cache"
```

## Tips and Best Practices

1. **Provider Selection**:
   - Use Jina for multilingual content
   - Use Voyage for fast English processing
   - Use Transformers for privacy-sensitive applications

2. **Performance Optimization**:
   - Cache model instances using AIFactory
   - Use appropriate `top_k` values to reduce processing time
   - For Transformers: Use GPU when available

3. **Quality Improvement**:
   - Combine with good initial retrieval (embedding-based search)
   - Use reranking as a refinement step, not primary retrieval
   - Consider query preprocessing for better results

4. **Error Handling**:
   - Always validate inputs before reranking
   - Handle network errors gracefully for cloud providers
   - Implement fallback strategies for high-availability systems

## Migration Guide

### From Other Libraries

If you're migrating from other reranking libraries:

```python
# Old approach with different libraries
# from sentence_transformers import CrossEncoder
# from cohere import Client

# New unified approach with Esperanto
from esperanto.factory import AIFactory

# Single interface for all providers
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")
results = reranker.rerank(query, documents, top_k=5)

# Consistent response format across all providers
for result in results.results:
    print(f"Score: {result.relevance_score}, Doc: {result.document}")
```

### Provider Switching

Switch between providers with minimal code changes:

```python
# Easy provider switching
providers = ["jina", "voyage", "transformers"]
models = ["jina-reranker-v2-base-multilingual", "rerank-2", "Qwen/Qwen3-Reranker-4B"]

for provider, model in zip(providers, models):
    reranker = AIFactory.create_reranker(provider, model)
    results = reranker.rerank(query, documents, top_k=3)
    print(f"{provider}: {results.results[0].relevance_score:.3f}")
```

This universal interface ensures that your code remains clean and maintainable while giving you the flexibility to experiment with different reranking providers and models.