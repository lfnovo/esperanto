# Transformers

## Overview

The Transformers provider enables **complete privacy and local processing** by running models directly on your machine. No data is sent to external APIs, making it perfect for sensitive data and air-gapped environments.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ❌ | Not available |
| Embeddings | ✅ | 100+ models from HuggingFace, advanced features |
| Reranking | ✅ | Universal support: CrossEncoder, Jina, Qwen, Mixedbread |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://huggingface.co/docs/transformers

## Prerequisites

### Installation

Install Esperanto with transformers extras:

```bash
pip install "esperanto[transformers]"
```

This installs:
- `transformers>=4.40.0` - Core Hugging Face library
- `torch>=2.2.2` - PyTorch framework
- `sentence-transformers>=2.2.0` - Enhanced embedding features
- `scikit-learn>=1.3.0` - PCA dimension reduction
- `numpy>=1.21.0` - Numerical operations

### Optional Dependencies

**For Mixedbread v2 Reranking:**
```bash
pip install mxbai-rerank
```

**For GPU Support:**
```bash
# NVIDIA CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon
# torch with MPS support is included by default
```

### Account Requirements
- **None!** No API keys needed
- Models downloaded from HuggingFace Hub
- Optional: HuggingFace token for private/gated models

## Environment Variables

```bash
# Optional: HuggingFace token for private/gated models
# HF_TOKEN="your-huggingface-token"

# Optional: Custom model cache directory
# TRANSFORMERS_CACHE="/path/to/model/cache"

# Optional: Disable tokenizer parallelism warnings in notebooks
# TOKENIZERS_PARALLELISM="false"
```

**No API key required!** All processing is local.

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Embedding model
embedder = AIFactory.create_embedding("transformers", "sentence-transformers/all-MiniLM-L6-v2")

# Reranker model
reranker = AIFactory.create_reranker("transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### Direct Instantiation

```python
from esperanto.providers.embedding.transformers import TransformersEmbeddingModel
from esperanto.providers.reranker.transformers import TransformersRerankerModel

# Embedding model
embedder = TransformersEmbeddingModel(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    config={
        "device": "auto",           # Auto-detect GPU/MPS/CPU
        "pooling_strategy": "mean"  # Embedding pooling method
    }
)

# Reranker model
reranker = TransformersRerankerModel(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    config={
        "device": "auto",
        "cache_dir": "./models"
    }
)
```

## Capabilities

### Embeddings

**Recommended Models:**

| Model | Size | Dimensions | Best For |
|-------|------|------------|----------|
| **all-MiniLM-L6-v2** | 22MB | 384 | Fast, lightweight, development |
| **all-mpnet-base-v2** | 420MB | 768 | High quality, general purpose |
| **multilingual-e5-large-instruct** | 2.2GB | 1024 | Task optimization, multilingual |
| **Qwen3-Embedding-4B** | 8.5GB | 1024 | Large context (8192 tokens) |
| **bge-large-en-v1.5** | 1.2GB | 1024 | English, high quality |

**Configuration:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",
    config={
        "device": "auto",                    # auto, cuda, mps, cpu
        "pooling_strategy": "mean",          # mean, cls, max
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,  # Task optimization
        "late_chunking": True,               # Intelligent chunking
        "output_dimensions": 128,            # Dimension control
        "truncate_at_max_length": True,
        "model_cache_dir": "./models"
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings - completely offline!
texts = ["Hello, world!", "Another text"]
response = model.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
    print(f"Vector: {embedding_obj.embedding[:5]}...")  # First 5 values
```

**Example - GPU Acceleration:**

```python
# Use CUDA GPU for faster processing
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-mpnet-base-v2",
    config={"device": "cuda"}
)

# Significantly faster on GPU
response = model.embed(large_text_batch)
```

**Example - Apple Silicon (MPS):**

```python
# Use Apple Silicon GPU
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",
    config={"device": "mps"}
)

# Accelerated on M1/M2/M3 Macs
response = model.embed(texts)
```

**Example - Task-Specific Optimization:**

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Optimize for search queries
query_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

# Optimize for documents
doc_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

# Task-specific prefixes are automatically applied
query_embedding = query_model.embed(["search query"])
doc_embeddings = doc_model.embed(["document 1", "document 2"])
```

**Example - Late Chunking (Long Documents):**

```python
# Handle documents longer than model context window
model = AIFactory.create_embedding(
    "transformers",
    "Qwen/Qwen3-Embedding-4B",
    config={
        "late_chunking": True,
        "device": "cuda"
    }
)

# Intelligently chunks and aggregates long documents
long_document = "Very long text..." * 1000
embeddings = model.embed([long_document])  # Returns single embedding
```

**Example - Dimension Control:**

```python
# Reduce dimensions for faster search
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-mpnet-base-v2",  # 768 dims
    config={
        "output_dimensions": 128,  # Reduce to 128 using PCA
        "device": "cuda"
    }
)

embeddings = model.embed(["Text"])
print(len(embeddings[0]))  # 128 dimensions

# Expand dimensions with zero padding
expanded_model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims
    config={"output_dimensions": 512}  # Expand to 512
)

embeddings = expanded_model.embed(["Text"])
print(len(embeddings[0]))  # 512 dimensions
```

**Example - Multilingual Embeddings:**

```python
# Use multilingual model for international content
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

texts = [
    "Hello, world!",           # English
    "Bonjour le monde!",       # French
    "Hola, mundo!",            # Spanish
    "こんにちは世界!"            # Japanese
]

response = model.embed(texts)
print(f"Generated {len(response.data)} multilingual embeddings")
```

### Reranking

**Universal Model Support - 4 Strategies:**

The Transformers reranker **automatically detects** the correct strategy for any model:

| Strategy | Models | Use Case |
|----------|--------|----------|
| **CrossEncoder** | cross-encoder/*, BAAI/bge-reranker-*, mixedbread-ai/*-v1 | General-purpose |
| **Sequence Classification** | jinaai/jina-reranker-* | High accuracy, multilingual |
| **Causal Language Model** | Qwen/Qwen3-Reranker-* | Advanced reasoning |
| **Mixedbread v2** | mixedbread-ai/*-v2 | Latest high-performance |

**Recommended Models:**

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | 90MB | Very Fast | Good | Development, production |
| **BAAI/bge-reranker-base** | 1.1GB | Fast | High | Multilingual, quality |
| **jinaai/jina-reranker-v2-base-multilingual** | 1.3GB | Fast | High | 100+ languages |
| **Qwen/Qwen3-Reranker-0.6B** | 1.2GB | Medium | Very High | Complex queries |
| **mixedbread-ai/mxbai-rerank-base-v2** | 1.2GB | Fast | Very High | Latest performance |

**Configuration:**

```python
from esperanto.factory import AIFactory

reranker = AIFactory.create_reranker(
    "transformers",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    config={
        "device": "auto",
        "cache_dir": "./models"
    }
)
```

**Example - Basic Reranking:**

```python
from esperanto.factory import AIFactory

# Create reranker - works with ANY supported model!
reranker = AIFactory.create_reranker(
    "transformers",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "The weather forecast shows rain tomorrow.",
    "Python is a popular programming language.",
    "Deep learning uses neural networks.",
    "Coffee is best served hot."
]

# Rerank documents by relevance
results = reranker.rerank(query, documents, top_k=3)

# Results sorted by relevance
for i, result in enumerate(results.results):
    print(f"{i+1}. Score: {result.relevance_score:.3f}")
    print(f"   Document: {result.document[:60]}...\n")
```

**Example - Universal Model Switching:**

```python
# Same code works for ALL model types!
models = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",     # CrossEncoder
    "BAAI/bge-reranker-base",                   # CrossEncoder
    "jinaai/jina-reranker-v2-base-multilingual", # Sequence classification
    "Qwen/Qwen3-Reranker-0.6B",                # Causal LM
]

query = "artificial intelligence applications"
documents = ["AI doc 1", "AI doc 2", "unrelated doc"]

for model_name in models:
    reranker = AIFactory.create_reranker("transformers", model_name)
    results = reranker.rerank(query, documents, top_k=2)
    print(f"{model_name}: {results.results[0].relevance_score:.3f}")
```

**Example - GPU-Accelerated Reranking:**

```python
# Use GPU for faster reranking with large models
reranker = AIFactory.create_reranker(
    "transformers",
    "Qwen/Qwen3-Reranker-4B",
    config={"device": "cuda"}
)

# Much faster on GPU for large models
results = reranker.rerank(query, large_document_list, top_k=10)
```

**Example - Multilingual Reranking:**

```python
# Use multilingual model for international queries
reranker = AIFactory.create_reranker(
    "transformers",
    "jinaai/jina-reranker-v2-base-multilingual"
)

# Works with 100+ languages
query = "Was ist künstliche Intelligenz?"  # German
documents = [
    "KI ist die Simulation menschlicher Intelligenz.",
    "Das Wetter wird morgen regnerisch.",
    "Python ist eine beliebte Programmiersprache."
]

results = reranker.rerank(query, documents)
```

**Example - Mixedbread v2 (Latest Performance):**

```bash
# First install the dependency
pip install mxbai-rerank
```

```python
# Use latest high-performance models
reranker = AIFactory.create_reranker(
    "transformers",
    "mixedbread-ai/mxbai-rerank-base-v2"
)

results = reranker.rerank(query, documents, top_k=5)
# State-of-the-art performance
```

## Advanced Features

### Task-Specific Optimization

Advanced prefixes for different embedding use cases:

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Supported task types
task_types = {
    EmbeddingTaskType.RETRIEVAL_QUERY: "Search queries",
    EmbeddingTaskType.RETRIEVAL_DOCUMENT: "Documents for search",
    EmbeddingTaskType.CLASSIFICATION: "Text classification",
    EmbeddingTaskType.CLUSTERING: "Document clustering",
    EmbeddingTaskType.SIMILARITY: "Semantic similarity",
    EmbeddingTaskType.CODE_RETRIEVAL: "Code search",
    EmbeddingTaskType.QUESTION_ANSWERING: "Q&A optimization",
    EmbeddingTaskType.FACT_VERIFICATION: "Fact checking"
}

# Apply task-specific optimization
model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={"task_type": EmbeddingTaskType.CLASSIFICATION}
)
```

### Semantic Late Chunking

Intelligently segment long texts that exceed model context:

```python
model = AIFactory.create_embedding(
    "transformers",
    "Qwen/Qwen3-Embedding-4B",
    config={
        "late_chunking": True,
        "device": "cuda"
    }
)

# Handles documents longer than 8192 tokens
long_document = "..." * 10000
embeddings = model.embed([long_document])  # Single aggregated embedding
```

**How it works:**
- Uses sentence-transformers for semantic boundary detection
- Model-aware chunk sizing (512-8192 tokens based on model)
- Mean aggregation of chunk embeddings
- Fallback to simple sentence chunking if dependencies unavailable

### Output Dimension Control

Control embedding dimensionality:

```python
# Dimension reduction (PCA)
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-mpnet-base-v2",  # 768 dims
    config={"output_dimensions": 256}  # Reduce to 256
)

# Dimension expansion (zero padding)
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims
    config={"output_dimensions": 512}  # Expand to 512
)
```

### Pooling Strategies

Different methods to extract embeddings:

```python
# Mean pooling (default, best for most cases)
mean_model = AIFactory.create_embedding(
    "transformers",
    "bert-base-uncased",
    config={"pooling_strategy": "mean"}
)

# CLS token (good for classification)
cls_model = AIFactory.create_embedding(
    "transformers",
    "bert-base-uncased",
    config={"pooling_strategy": "cls"}
)

# Max pooling (emphasizes key features)
max_model = AIFactory.create_embedding(
    "transformers",
    "bert-base-uncased",
    config={"pooling_strategy": "max"}
)
```

### Model Caching

Models are automatically cached after first download:

```python
# Set custom cache directory
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",
    config={"model_cache_dir": "/path/to/cache"}
)

# Or use environment variable
import os
os.environ["TRANSFORMERS_CACHE"] = "/path/to/cache"
```

### Async Operations

All providers support async operations:

```python
import asyncio

async def embed_async():
    model = AIFactory.create_embedding(
        "transformers",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    results = await model.aembed(texts)
    return results

async def rerank_async():
    reranker = AIFactory.create_reranker(
        "transformers",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    results = await reranker.arerank(query, documents, top_k=3)
    return results

# Run async operations
results = asyncio.run(embed_async())
```

## Device Selection Guide

### Auto Detection (Recommended)

```python
config = {"device": "auto"}
# Automatically uses: GPU > MPS > CPU
```

### Manual Selection

```python
# NVIDIA GPU
config = {"device": "cuda"}

# Apple Silicon GPU
config = {"device": "mps"}

# CPU (always available)
config = {"device": "cpu"}
```

### Device Performance

| Device | Speed | Use Case |
|--------|-------|----------|
| **CUDA (NVIDIA GPU)** | Fastest | Production, large models, high volume |
| **MPS (Apple Silicon)** | Fast | Mac development, medium workloads |
| **CPU** | Baseline | Development, small models, low volume |

## Memory Optimization

### Model Size vs. Memory

| Model Category | Model Size | GPU Memory | Recommendation |
|----------------|-----------|------------|----------------|
| **Tiny** | 22-80MB | <1GB | Development, real-time |
| **Small** | 80-420MB | 1-2GB | Production, balanced |
| **Base** | 420MB-1.2GB | 2-4GB | High quality |
| **Large** | 1.2GB-2.2GB | 4-8GB | Best quality |
| **XL** | 2.2GB+ | 8GB+ | Specialized tasks |

### Tips for Limited Memory

```python
# Use smaller models
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2"  # Only 22MB
)

# Force CPU if GPU memory is limited
config = {"device": "cpu"}

# Process in smaller batches
for batch in batches(texts, batch_size=32):
    response = model.embed(batch)
```

## Performance Benchmarks

### Embedding Speed (approximate)

**Model: all-MiniLM-L6-v2, Batch size: 100**
- CPU: ~2 sec/batch
- MPS (M1): ~0.5 sec/batch
- CUDA (RTX 3080): ~0.1 sec/batch

**Model: all-mpnet-base-v2, Batch size: 100**
- CPU: ~8 sec/batch
- MPS (M1): ~2 sec/batch
- CUDA (RTX 3080): ~0.4 sec/batch

### Reranking Speed (approximate)

**Model: cross-encoder/ms-marco-MiniLM-L-6-v2, 100 docs**
- CPU: ~3 sec
- MPS (M1): ~1 sec
- CUDA (RTX 3080): ~0.3 sec

## Troubleshooting

### Common Errors

**Import Error:**
```
Error: No module named 'transformers'
```
**Solution:** Install transformers extras: `pip install "esperanto[transformers]"`

**CUDA Out of Memory:**
```
Error: CUDA out of memory
```
**Solution:** Use smaller model, reduce batch size, or force CPU: `config={"device": "cpu"}`

**Model Download Fails:**
```
Error: Connection timeout
```
**Solution:** Check internet connection or download model manually to cache directory.

**Slow Performance:**
```
Models loading slowly
```
**Solution:** Models are cached after first load. Subsequent loads are fast.

**Mixedbread v2 Error:**
```
ImportError: No module named 'mxbai_rerank'
```
**Solution:** Install dependency: `pip install mxbai-rerank`

### Best Practices

1. **Start Small:** Begin with lightweight models for development.

2. **Use GPU:** Leverage GPU acceleration for production workloads.

3. **Batch Processing:** Process texts in batches for efficiency.

4. **Cache Models:** Models are cached automatically after first download.

5. **Task Optimization:** Use task types for better embedding quality.

6. **Monitor Memory:** Choose models appropriate for your hardware.

7. **Test Locally:** Validate everything works before deployment.

## Privacy & Security

### Complete Data Privacy

**No External API Calls:**
- All processing happens locally
- No data sent to external servers
- Perfect for sensitive/confidential data

**Air-Gapped Environments:**
- Download models once
- Run completely offline
- No internet connection needed after setup

**Compliance:**
- GDPR compliant (no data leaves your system)
- HIPAA compliant (healthcare data)
- Enterprise compliant (financial, legal data)

### Example - Offline Processing

```python
# 1. Download model once (requires internet)
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Disconnect from internet

# 3. Process sensitive data completely offline
sensitive_texts = ["Confidential document 1", "Confidential document 2"]
embeddings = model.embed(sensitive_texts)
# No data ever leaves your machine!
```

## Use Cases

### Privacy-First Applications

```python
# Process confidential business documents
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-mpnet-base-v2"
)

confidential_docs = [
    "Q4 financial projections...",
    "Strategic acquisition plans...",
    "Proprietary research data..."
]

# Process completely locally
embeddings = model.embed(confidential_docs)
```

### Air-Gapped Systems

```python
# Healthcare system with no internet access
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-mpnet-base-v2",
    config={
        "device": "cuda",
        "model_cache_dir": "/secure/cache"
    }
)

# Process patient data offline
patient_notes = ["Patient note 1", "Patient note 2"]
embeddings = model.embed(patient_notes)
```

### Cost Optimization

```python
# High-volume processing with no per-token costs
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",
    config={"device": "cuda"}
)

# Process millions of documents
# No API costs, one-time download
for batch in large_document_batches:
    embeddings = model.embed(batch)
```

### Research & Experimentation

```python
# Try different models without API costs
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-large-en-v1.5"
]

for model_name in models:
    model = AIFactory.create_embedding("transformers", model_name)
    embeddings = model.embed(test_texts)
    # Evaluate and compare
```

## See Also

- [Embeddings Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/embedding.md)
- [Reranking Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/rerank.md)
- [Transformers Advanced Features](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/TRANSFORMERS_ADVANCED_FEATURES.md)
- [Embedding Providers](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/embedding/providers.md)
- [OpenAI Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/openai.md)
- [Jina Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/jina.md)
