# Embedding Models

## Overview

Embedding models convert text into high-dimensional vectors that capture semantic meaning. These vectors enable similarity search, clustering, classification, and other vector-based operations essential for modern AI applications.

## Common Use Cases

- **Semantic Search**: Find documents similar to a query based on meaning
- **Recommendation Systems**: Match users with relevant content
- **Clustering & Classification**: Group similar texts, categorize documents
- **Retrieval-Augmented Generation (RAG)**: Provide context to LLMs

## Interface

### Creating an Embedding Model

```python
from esperanto.factory import AIFactory

# Basic usage
embedder = AIFactory.create_embedding(
    provider="openai",
    model_name="text-embedding-3-small"
)

# With configuration
embedder = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3",
    config={
        "timeout": 60.0,
        "batch_size": 32
    }
)
```

### Core Methods

#### `embed(texts)`

Synchronous embedding generation.

```python
texts = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks"
]

response = embedder.embed(texts)
vectors = [item.embedding for item in response.data]
```

#### `aembed(texts)`

Asynchronous embedding generation (identical interface to `embed`).

```python
response = await embedder.aembed(texts)
vectors = [item.embedding for item in response.data]
```

## Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | float | 60.0 | Request timeout in seconds |
| `batch_size` | int | Provider default | Number of texts to process per request |

### Input Format

```python
# List of strings
texts = ["First text", "Second text", "Third text"]

# Single string (automatically converted to list)
text = "Single text to embed"
```

## Response Structure

All embedding providers return standardized `EmbeddingResponse` objects:

```python
response = embedder.embed(texts)

# Access embeddings
response.data[0].embedding    # Vector for first text (list of floats)
response.data[0].index         # Index of the text (0)

# Metadata
response.model                 # Model name used
response.usage.total_tokens    # Total tokens processed

# Get all vectors
vectors = [item.embedding for item in response.data]
```

## Task-Aware Embeddings

Esperanto supports task-specific optimization across all embedding providers. Providers either support this natively or emulate it through intelligent text processing.

### Available Task Types

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Task types
EmbeddingTaskType.RETRIEVAL_QUERY        # Search queries
EmbeddingTaskType.RETRIEVAL_DOCUMENT     # Documents to store
EmbeddingTaskType.SIMILARITY             # General text similarity
EmbeddingTaskType.CLASSIFICATION         # Text classification
EmbeddingTaskType.CLUSTERING             # Document clustering
EmbeddingTaskType.CODE_RETRIEVAL         # Code search
EmbeddingTaskType.QUESTION_ANSWERING     # Q&A optimization
EmbeddingTaskType.FACT_VERIFICATION      # Fact checking
```

### Using Task Types

```python
embedder = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY
    }
)

# Or per-request
query_vector = embedder.embed(
    ["What is machine learning?"],
    task_type=EmbeddingTaskType.RETRIEVAL_QUERY
)
```

### Provider Support

| Feature | Native Support | Emulated Support |
|---------|---------------|------------------|
| Task Types | Jina, Google | OpenAI, Transformers, Others |
| Late Chunking | Jina | - |
| Output Dimensions | Jina, OpenAI | - |

→ **See [Task-Aware Embeddings Guide](../advanced/task-aware-embeddings.md)** for detailed information.

## Advanced Features

### Output Dimensions

Control vector size for some providers:

```python
embedder = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3",
    config={"output_dimensions": 512}  # Default: 1024
)
```

**Supported**: Jina, OpenAI (some models)

### Late Chunking

Improve long-context embeddings:

```python
embedder = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3",
    config={"late_chunking": True}
)
```

**Supported**: Jina (native), potentially others via provider-specific APIs

## Provider Selection

→ **See [Provider Comparison](../providers/README.md)** for detailed comparison and selection guide.

### Quick Provider Guide

- **OpenAI**: Industry standard, excellent quality, broad compatibility
- **Jina**: Advanced features (task types, late chunking, dimensions control)
- **Google**: Native task type support, competitive pricing
- **Voyage**: Specialized for retrieval, strong performance
- **Transformers**: Local deployment, privacy-focused, no API costs
- **Azure**: Enterprise compliance, private deployment
- **Ollama**: Local models, simple setup

## Examples

### Basic Embedding

```python
from esperanto.factory import AIFactory

embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")

texts = [
    "Esperanto is a universal AI interface",
    "Machine learning models process data"
]

response = embedder.embed(texts)
vectors = [item.embedding for item in response.data]

print(f"Generated {len(vectors)} vectors")
print(f"Vector dimension: {len(vectors[0])}")
```

### Task-Optimized Search

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Embed documents
doc_embedder = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

documents = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Java is popular for enterprise applications"
]

doc_vectors = doc_embedder.embed(documents)

# Embed query
query_embedder = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

query = ["Which language is best for web development?"]
query_vector = query_embedder.embed(query)

# Compute similarity (cosine similarity, etc.)
# ...
```

### Async Batch Processing

```python
embedder = AIFactory.create_embedding(
    "voyage", "voyage-2",
    config={"batch_size": 50}
)

large_corpus = [...]  # Many documents

# Process asynchronously
response = await embedder.aembed(large_corpus)
vectors = [item.embedding for item in response.data]
```

### Dimensionality Control

```python
# Standard dimensions (1024)
embedder_full = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3"
)

# Reduced dimensions (faster, less storage)
embedder_small = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"output_dimensions": 256}
)

text = ["Sample text for embedding"]

full_vector = embedder_full.embed(text)
small_vector = embedder_small.embed(text)

print(f"Full: {len(full_vector.data[0].embedding)} dims")
print(f"Small: {len(small_vector.data[0].embedding)} dims")
```

## Similarity Computation

Esperanto returns raw vectors. Use standard libraries for similarity:

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Example
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")

texts = ["AI and machine learning", "Artificial intelligence research"]
response = embedder.embed(texts)

vec1 = response.data[0].embedding
vec2 = response.data[1].embedding

similarity = cosine_similarity(vec1, vec2)
print(f"Similarity: {similarity:.4f}")
```

## Advanced Topics

- **Task-Aware Embeddings**: [docs/advanced/task-aware-embeddings.md](../advanced/task-aware-embeddings.md)
- **Transformers Advanced Features**: [docs/advanced/transformers-features.md](../advanced/transformers-features.md)
- **Timeout Configuration**: [docs/advanced/timeout-configuration.md](../advanced/timeout-configuration.md)

## See Also

- [Provider Setup Guides](../providers/README.md)
- [Reranking Models](./reranking.md)
- [Language Models](./llm.md)
