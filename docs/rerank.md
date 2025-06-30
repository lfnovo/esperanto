# Reranking Providers 🔄

Esperanto provides a universal interface for reranking providers, allowing you to improve search relevance and document ranking using different models while maintaining a consistent API.

## Overview

Reranking is a crucial component in modern search and RAG (Retrieval-Augmented Generation) systems. It takes an initial set of documents and reorders them based on their relevance to a given query, significantly improving the quality of search results.

### Key Benefits

- **Universal Interface**: Switch between providers without changing code
- **Score Normalization**: Consistent 0-1 relevance scores across all providers
- **Privacy Options**: Local reranking with Transformers for offline processing
- **Performance**: Cloud-based solutions for high-throughput applications
- **Model Flexibility**: Support for 12+ different model architectures
- **LangChain Integration**: Drop-in compatibility with existing workflows

## Supported Providers

| Provider | Models | Type | Key Features |
|----------|--------|------|--------------| 
| **Jina** | jina-reranker-v2-base-multilingual, jina-reranker-v1-base-en | Cloud | Multilingual support, high accuracy |
| **Voyage** | rerank-2, rerank-1 | Cloud | Fast processing, good performance |
| **Transformers** | **12+ Model Types** | Local | **Universal support, privacy-first, offline processing** |

### Transformers Provider - Universal Model Support 🌟

The Transformers provider now supports **4 different reranking strategies** with automatic model detection:

| Strategy | Models | Architecture | Use Case |
|----------|---------|--------------|----------|
| **CrossEncoder** | `cross-encoder/ms-marco-*`, `BAAI/bge-reranker-*`, `mixedbread-ai/*-v1` | sentence-transformers | General-purpose reranking |
| **Sequence Classification** | `jinaai/jina-reranker-*` | AutoModelForSequenceClassification | High-accuracy multilingual |
| **Causal Language Model** | `Qwen/Qwen3-Reranker-*` | AutoModelForCausalLM | Advanced reasoning-based ranking |
| **Mixedbread v2** | `mixedbread-ai/*-v2` | mxbai-rerank library | Latest high-performance models |

#### Supported Models

**CrossEncoder Models (sentence-transformers strategy):**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` - Fast, lightweight
- `cross-encoder/ms-marco-electra-base` - Better accuracy
- `BAAI/bge-reranker-base` - Multilingual support
- `BAAI/bge-reranker-large` - High accuracy
- `mixedbread-ai/mxbai-rerank-base-v1` - Optimized performance
- `mixedbread-ai/mxbai-rerank-large-v1` - Maximum accuracy

**Sequence Classification Models:**
- `jinaai/jina-reranker-v2-base-multilingual` - 100+ languages

**Causal Language Models:**
- `Qwen/Qwen3-Reranker-4B` - Advanced reasoning (default)
- `Qwen/Qwen3-Reranker-0.6B` - Lightweight option

**Mixedbread v2 Models:**
- `mixedbread-ai/mxbai-rerank-base-v2` - Latest generation
- `mixedbread-ai/mxbai-rerank-large-v2` - Maximum performance

## Quick Start

### Universal Transformers Reranker (Recommended)

```python
from esperanto.factory import AIFactory

# Works with ANY supported model - same interface!
models_to_try = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",     # Fast CrossEncoder
    "BAAI/bge-reranker-base",                   # Multilingual CrossEncoder  
    "jinaai/jina-reranker-v2-base-multilingual", # Sequence classification
    "Qwen/Qwen3-Reranker-0.6B",                # Causal language model
    "mixedbread-ai/mxbai-rerank-base-v2"        # Mixedbread v2 (requires: pip install mxbai-rerank)
]

# Same code works for all models!
for model_name in models_to_try:
    try:
        reranker = AIFactory.create_reranker("transformers", model_name)
        
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
        
        print(f"\n--- Results from {model_name} ---")
        for i, result in enumerate(results.results):
            print(f"{i+1}. Score: {result.relevance_score:.3f}")
            print(f"   Document: {result.document[:80]}...")
            print()
            
    except ImportError as e:
        print(f"Skipping {model_name}: {e}")
```

### Cloud Providers

```python
# Jina (Cloud)
jina_reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

# Voyage (Cloud)
voyage_reranker = AIFactory.create_reranker("voyage", "rerank-2")

# Same interface for all providers
results = jina_reranker.rerank(query, documents, top_k=3)
```

## Provider-Specific Usage

### Transformers Reranker (Universal Local)

```python
from esperanto.providers.reranker.transformers import TransformersRerankerModel

# Automatic strategy detection - works with any supported model
reranker = TransformersRerankerModel(
    model_name="BAAI/bge-reranker-base",  # Automatically uses CrossEncoder strategy
    config={
        "cache_dir": "./models",      # Model cache directory
        "device": "auto"              # Auto-detect GPU/MPS/CPU
    }
)

# Privacy-first: Works completely offline
results = reranker.rerank(query, documents)

# Try different models with same code
for model in ["cross-encoder/ms-marco-MiniLM-L-6-v2", "Qwen/Qwen3-Reranker-0.6B"]:
    reranker = TransformersRerankerModel(model_name=model)
    results = reranker.rerank(query, documents, top_k=2)
    print(f"{model}: {results.results[0].relevance_score:.3f}")
```

#### Strategy Auto-Detection

The Transformers provider automatically detects the correct strategy based on model names:

```python
# These all work automatically with the same interface
strategies = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "sentence_transformers",
    "BAAI/bge-reranker-base": "sentence_transformers", 
    "jinaai/jina-reranker-v2-base-multilingual": "sequence_classification",
    "Qwen/Qwen3-Reranker-4B": "causal_lm",
    "mixedbread-ai/mxbai-rerank-base-v2": "mixedbread_v2"
}

for model_name, expected_strategy in strategies.items():
    reranker = AIFactory.create_reranker("transformers", model_name)
    print(f"{model_name} -> {reranker.strategy}")  # Prints detected strategy
```

#### Requirements by Strategy

**Base Requirements** (always needed):
```bash
pip install "esperanto[transformers]"
```

**Optional for Mixedbread v2**:
```bash
pip install mxbai-rerank  # Only needed for mixedbread-ai/*-v2 models
```

#### Performance Features

- **Auto-Device Detection**: Automatically uses GPU/MPS if available, falls back to CPU
- **Optimized Tokenization**: Uses fast tokenizer methods for better performance
- **Memory Efficient**: Proper resource management across all strategies  
- **Warning-Free**: Eliminates tokenizer and processing warnings
- **Caching**: Models cached automatically using HuggingFace cache

### Jina Reranker (Cloud)

```python
from esperanto.providers.reranker.jina import JinaRerankerModel

reranker = JinaRerankerModel(
    api_key="your-jina-api-key",  # Or set JINA_API_KEY env var
    model_name="jina-reranker-v2-base-multilingual",
    config={
        "timeout": 30  # Custom timeout
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

### Voyage Reranker (Cloud)

```python
from esperanto.providers.reranker.voyage import VoyageRerankerModel

reranker = VoyageRerankerModel(
    api_key="your-voyage-api-key",  # Or set VOYAGE_API_KEY env var
    model_name="rerank-2",
    config={
        "timeout": 30  # Custom timeout
    }
)

results = reranker.rerank(query, documents, top_k=5)
```

#### Available Models
- `rerank-2`: Latest Voyage reranking model (default)
- `rerank-1`: Previous generation model

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

### Model Comparison

```python
# Compare different models on the same data
models = [
    ("transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("transformers", "BAAI/bge-reranker-base"),
    ("transformers", "Qwen/Qwen3-Reranker-0.6B"),
    ("jina", "jina-reranker-v2-base-multilingual"),
    ("voyage", "rerank-2")
]

query = "What is artificial intelligence?"
documents = [
    "AI is the simulation of human intelligence in machines.",
    "The weather is sunny today.",
    "Machine learning is a subset of AI.",
    "Neural networks are used in deep learning."
]

for provider, model in models:
    try:
        reranker = AIFactory.create_reranker(provider, model)
        results = reranker.rerank(query, documents, top_k=2)
        
        print(f"\n{provider}/{model}:")
        print(f"Top result: {results.results[0].relevance_score:.3f}")
        print(f"Document: {results.results[0].document[:60]}...")
        
    except Exception as e:
        print(f"Error with {provider}/{model}: {e}")
```

### Async Usage

```python
# All providers support async operations
import asyncio

async def async_rerank():
    reranker = AIFactory.create_reranker("transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    results = await reranker.arerank(query, documents, top_k=3)
    return results

# Run async reranking
results = asyncio.run(async_rerank())
```

### Batch Processing

```python
# Efficient batch processing for multiple queries
queries = [
    "What is machine learning?",
    "How does Python work?", 
    "What is artificial intelligence?"
]

documents = [
    "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "Artificial intelligence refers to the simulation of human intelligence in machines.",
    "The weather forecast predicts rain tomorrow.",
    "Coffee helps improve productivity and focus during work."
]

reranker = AIFactory.create_reranker("transformers", "BAAI/bge-reranker-base")

# Process all queries
results = []
for query in queries:
    result = reranker.rerank(query, documents, top_k=2)
    results.append((query, result))
    
for query, result in results:
    print(f"\nQuery: {query}")
    print(f"Best match: {result.results[0].document[:80]}...")
    print(f"Score: {result.results[0].relevance_score:.3f}")
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

# Works with any provider/model
reranker = AIFactory.create_reranker("transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2")
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
except ImportError as e:
    # Missing dependencies (e.g., mxbai-rerank for Mixedbread v2)
    print(f"Missing dependency: {e}")
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
- **Model Variety**: 12+ different architectures
- **Performance Varies**: 
  - CrossEncoder models: Fast, low memory
  - Qwen models: Higher accuracy, more memory (~4-8GB)
  - Jina local models: Balanced performance

### Performance by Strategy

| Strategy | Speed | Memory | Accuracy | Best For |
|----------|-------|--------|----------|----------|
| CrossEncoder | Very Fast | Low (1-2GB) | Good | General use, production |
| Sequence Classification | Fast | Medium (2-4GB) | High | Multilingual, accuracy |
| Causal LM (Qwen) | Medium | High (4-8GB) | Very High | Research, complex queries |
| Mixedbread v2 | Fast | Medium (2-4GB) | Very High | Latest performance |

### Recommendations

**For Production APIs**: 
- Cloud providers (Jina/Voyage) for fastest response times
- CrossEncoder models for local deployment with good speed/accuracy balance

**For Privacy-Sensitive Data**: 
- Any Transformers model for complete offline processing
- CrossEncoder models for best speed vs privacy trade-off

**For Development**: 
- Start with CrossEncoder models for quick experimentation
- Switch to cloud providers for scalability testing

**For Maximum Accuracy**:
- Qwen models for complex reasoning tasks
- Mixedbread v2 models for latest performance benchmarks

## Model Comparison

| Provider | Model | Languages | Speed | Accuracy | Privacy | Memory |
|----------|-------|-----------|-------|----------|---------|---------|
| Transformers | cross-encoder/ms-marco-* | English | Very Fast | Good | Local | Low |
| Transformers | BAAI/bge-reranker-* | Multilingual | Fast | High | Local | Low-Medium |
| Transformers | jinaai/jina-reranker-* | 100+ | Fast | High | Local | Medium |
| Transformers | Qwen/Qwen3-Reranker-* | Multilingual | Medium | Very High | Local | High |
| Transformers | mixedbread-ai/*-v2 | Multilingual | Fast | Very High | Local | Medium |
| Jina | jina-reranker-v2-* | 100+ | Fast | High | Cloud | N/A |
| Voyage | rerank-2 | Multilingual | Very Fast | High | Cloud | N/A |

## Use Cases

### 1. RAG Systems
```python
# Universal model selection for different RAG needs
models = {
    "speed": "cross-encoder/ms-marco-MiniLM-L-6-v2",      # Fast response
    "accuracy": "Qwen/Qwen3-Reranker-4B",                 # Best accuracy  
    "multilingual": "jinaai/jina-reranker-v2-base-multilingual", # Global use
    "latest": "mixedbread-ai/mxbai-rerank-base-v2"        # Cutting edge
}

# Select based on your requirements
reranker = AIFactory.create_reranker("transformers", models["speed"])

# Improve retrieval quality in RAG pipelines
retrieved_docs = vector_search(query, top_k=20)  # Initial retrieval
reranked = reranker.rerank(query, retrieved_docs, top_k=5)  # Rerank top results
context = [r.document for r in reranked.results]  # Use for generation
```

### 2. Search Applications
```python
# Multi-language search with automatic model selection
def create_reranker_for_language(query_language):
    if query_language in ["en", "english"]:
        return AIFactory.create_reranker("transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    else:
        return AIFactory.create_reranker("transformers", "BAAI/bge-reranker-base")

# Enhance search relevance
search_results = search_engine.search(query, limit=50)
reranker = create_reranker_for_language(detect_language(query))
reranked = reranker.rerank(query, search_results, top_k=10)
return reranked.results  # Return top 10 most relevant
```

### 3. Document Filtering
```python
# Use different models for different quality thresholds
models = {
    "fast_filter": "cross-encoder/ms-marco-MiniLM-L-6-v2",   # Quick filtering
    "precise_filter": "Qwen/Qwen3-Reranker-0.6B"            # Precise ranking
}

# Quick pass with fast model
fast_reranker = AIFactory.create_reranker("transformers", models["fast_filter"])
quick_filter = fast_reranker.rerank(query, all_documents, top_k=50)

# Precise ranking of promising candidates  
precise_reranker = AIFactory.create_reranker("transformers", models["precise_filter"])
final_results = precise_reranker.rerank(query, [r.document for r in quick_filter.results], top_k=10)
```

## Environment Variables

```bash
# Cloud providers
export JINA_API_KEY="your-jina-api-key"
export VOYAGE_API_KEY="your-voyage-api-key"

# Transformers (optional)
export TRANSFORMERS_CACHE="/path/to/model/cache"
export TOKENIZERS_PARALLELISM="false"  # For notebook environments
```

## Tips and Best Practices

1. **Model Selection**:
   - **CrossEncoder models**: Best for speed and low memory usage
   - **BAAI models**: Good balance of multilingual support and performance
   - **Jina models**: Use local version for privacy, cloud for convenience
   - **Qwen models**: Best accuracy for complex queries
   - **Mixedbread v2**: Latest state-of-the-art performance

2. **Performance Optimization**:
   - Cache model instances using AIFactory
   - Use appropriate `top_k` values to reduce processing time
   - For Transformers: GPU greatly improves speed for larger models
   - Consider model size vs. accuracy trade-offs

3. **Quality Improvement**:
   - Combine with good initial retrieval (embedding-based search)
   - Use reranking as a refinement step, not primary retrieval
   - Consider query preprocessing for better results
   - Test different models to find best fit for your data

4. **Privacy & Compliance**:
   - Use Transformers provider for complete data privacy
   - All processing happens locally with no external API calls
   - Models cached locally after first download

5. **Error Handling**:
   - Always validate inputs before reranking
   - Handle missing dependencies gracefully (especially mxbai-rerank)
   - Implement fallback strategies for high-availability systems
   - Test different models in development to ensure compatibility

## Migration Guide

### From Other Libraries

```python
# Old approach with different libraries
# from sentence_transformers import CrossEncoder
# from cohere import Client  
# import torch

# New unified approach with Esperanto
from esperanto.factory import AIFactory

# Single interface for ALL reranking models
models = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",     # sentence-transformers equivalent
    "jinaai/jina-reranker-v2-base-multilingual", # Jina local equivalent
    "Qwen/Qwen3-Reranker-4B",                   # Advanced reasoning model
]

for model in models:
    reranker = AIFactory.create_reranker("transformers", model)
    results = reranker.rerank(query, documents, top_k=5)
    
    # Consistent response format across ALL models
    for result in results.results:
        print(f"Score: {result.relevance_score:.3f}, Doc: {result.document[:60]}...")
```

### Provider Switching

```python
# Easy provider and model switching
configurations = [
    ("transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2"),  # Local, fast
    ("transformers", "BAAI/bge-reranker-base"),               # Local, multilingual
    ("transformers", "Qwen/Qwen3-Reranker-0.6B"),            # Local, high accuracy
    ("jina", "jina-reranker-v2-base-multilingual"),          # Cloud, multilingual
    ("voyage", "rerank-2")                                    # Cloud, fast
]

query = "artificial intelligence applications"
documents = ["AI doc 1", "AI doc 2", "unrelated doc"]

for provider, model in configurations:
    try:
        reranker = AIFactory.create_reranker(provider, model)
        results = reranker.rerank(query, documents, top_k=2)
        print(f"{provider}/{model}: {results.results[0].relevance_score:.3f}")
    except Exception as e:
        print(f"Skipped {provider}/{model}: {e}")
```

This universal interface ensures that your code remains clean and maintainable while giving you unprecedented flexibility to experiment with different reranking providers and models. The Transformers provider alone supports 12+ different model architectures, all accessible through the same simple API!