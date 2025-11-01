# Transformers Advanced Features

## Overview

The Esperanto Transformers provider brings advanced embedding features to local, privacy-first processing. These features, previously only available through cloud providers, now work completely offline with HuggingFace transformer models.

This guide covers the Transformers provider's advanced capabilities:

- **Task-Specific Optimization**: Advanced prefixes optimized for different use cases
- **Semantic Late Chunking**: Intelligent text segmentation for long documents
- **Output Dimension Control**: PCA-based reduction and zero-padding expansion
- **Model-Aware Configuration**: Optimized settings for different transformer models
- **Graceful Degradation**: Fallback behavior when optional dependencies are missing

For general Transformers provider usage, see the [Transformers Provider Guide](/docs/providers/transformers.md).

## Installation

To use advanced features, install the transformers extras:

```bash
pip install esperanto[transformers]
```

This installs:
- `transformers>=4.40.0` - Core transformer models
- `torch>=2.2.2` - PyTorch backend
- `sentence-transformers>=2.2.0` - Semantic chunking support
- `scikit-learn>=1.3.0` - PCA dimension reduction
- `numpy>=1.21.0` - Numerical operations

## Quick Start

```python
from esperanto import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Create model with advanced features
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="Qwen/Qwen3-Embedding-4B",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "late_chunking": True,
        "output_dimensions": 512,
    }
)

# Generate embeddings with all advanced features
embeddings = model.embed(["Your text here"])
```

## Task-Specific Optimization

Task optimization applies sophisticated prefixes that help local models understand the intended use case, similar to how cloud providers optimize embeddings.

### Supported Task Types

| Task Type | Description | Prefix Applied |
|-----------|-------------|----------------|
| `RETRIEVAL_QUERY` | Search queries | "Represent this query for retrieving relevant documents: " |
| `RETRIEVAL_DOCUMENT` | Documents for search | "Represent this document for retrieval: " |
| `CLASSIFICATION` | Text classification | "Represent this text for classification: " |
| `CLUSTERING` | Document clustering | "Represent this text for clustering: " |
| `SIMILARITY` | Semantic similarity | "Represent this text for semantic similarity: " |
| `CODE_RETRIEVAL` | Code search | "Represent this code for search: " |
| `QUESTION_ANSWERING` | Q&A optimization | "Represent this question for answering: " |
| `FACT_VERIFICATION` | Fact checking | "Represent this claim for verification: " |

### Usage

```python
# For search queries
query_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

# For documents to be searched
doc_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

# The model automatically applies appropriate prefixes
queries = ["How do transformers work?"]
documents = ["Transformers are a deep learning architecture..."]

query_embeddings = query_model.embed(queries)
doc_embeddings = doc_model.embed(documents)
```

### How It Works

Task-specific prefixes are intelligently applied to your text before processing:

```python
# Input text
text = "machine learning tutorial"

# With RETRIEVAL_QUERY task type
# Actual text sent to model:
# "Represent this query for retrieving relevant documents: machine learning tutorial"

# With CLASSIFICATION task type
# Actual text sent to model:
# "Represent this text for classification: machine learning tutorial"
```

## Late Chunking

Late chunking intelligently segments long texts that exceed the model's context window, then aggregates the results for a single comprehensive embedding.

### Features

- **Semantic Boundary Detection**: Uses sentence-transformers to find natural break points
- **Model-Aware Limits**: Automatically configures chunk sizes based on the model
- **Mean Aggregation**: Combines chunk embeddings using mean pooling
- **Fallback Support**: Simple sentence-based chunking when advanced features unavailable

### Model-Specific Limits

The provider automatically detects optimal chunk sizes based on the model:

| Model Pattern | Max Chunk Tokens |
|---------------|-------------------|
| Qwen3/Qwen-3 | 8192 |
| E5/multilingual | 1024 |
| Default/BERT-like | 512 |

### Usage

```python
model = AIFactory.create_embedding(
    "transformers",
    "Qwen/Qwen3-Embedding-4B",
    config={"late_chunking": True}
)

# Handles long documents automatically
long_document = """
This is a very long document that exceeds the model's context window.
It contains multiple paragraphs and sections that need to be processed.
""" * 100  # Simulating a very long document

embeddings = model.embed([long_document])  # Returns single aggregated embedding
```

### How It Works

1. **Text Analysis**: Document is analyzed for length
2. **Semantic Segmentation**: If too long, split at sentence boundaries
3. **Chunk Processing**: Each chunk is embedded independently
4. **Aggregation**: Chunk embeddings are averaged into a single vector
5. **Final Embedding**: Single vector representing entire document

### Advanced Configuration

Combine late chunking with task optimization:

```python
model = AIFactory.create_embedding(
    "transformers",
    "Qwen/Qwen3-Embedding-4B",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True
    }
)

# Task prefix is applied before chunking, preserving task context
long_docs = ["Very long research paper text..." * 500]
embeddings = model.embed(long_docs)
```

### When to Use Late Chunking

```python
# Perfect for long documents
use_cases = [
    "Research papers (5-50 pages)",
    "Legal documents",
    "Technical documentation",
    "News articles",
    "Book chapters",
    "Product manuals"
]

# Enable for documents > 512 tokens
document_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={
        "late_chunking": True,
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)

embeddings = document_model.embed(use_cases)
```

## Output Dimension Control

Control the dimensionality of output embeddings through PCA-based reduction or zero-padding expansion.

### Dimension Reduction (PCA)

Uses Principal Component Analysis to reduce embedding dimensions while preserving the most important information.

```python
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims
    config={"output_dimensions": 128}  # Reduce to 128
)

embeddings = model.embed(["Text"])
print(len(embeddings.data[0].embedding))  # 128
```

#### How PCA Reduction Works

1. **Training Phase**: First embedding call fits PCA on the data
2. **Transformation**: Subsequent calls use fitted PCA to reduce dimensions
3. **Preservation**: Keeps most important variance in fewer dimensions
4. **Caching**: PCA model is cached for consistent transformations

#### Benefits

- **Faster similarity search**: Fewer dimensions = faster computations
- **Lower memory usage**: Smaller vectors reduce storage requirements
- **Maintained quality**: PCA preserves most important features

### Dimension Expansion (Zero Padding)

Expands embeddings by adding zero-valued dimensions.

```python
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims
    config={"output_dimensions": 512}  # Expand to 512
)

embeddings = model.embed(["Text"])
print(len(embeddings.data[0].embedding))  # 512
```

#### When to Use Expansion

- **Compatibility**: Match dimensions with other models in your pipeline
- **Future-proofing**: Prepare for potential model upgrades
- **Standardization**: Use consistent dimensions across different models

### Trade-offs

| Dimensions | Use Case | Quality | Speed | Memory |
|------------|----------|---------|-------|---------|
| **1536+** | Research, critical accuracy | ⭐⭐⭐⭐⭐ | 🐌 | 💾💾💾💾 |
| **1024** | Production, high quality | ⭐⭐⭐⭐ | ⚡ | 💾💾💾 |
| **512** | Real-time, good quality | ⭐⭐⭐ | ⚡⚡ | 💾💾 |
| **256** | Mobile, fast search | ⭐⭐ | ⚡⚡⚡ | 💾 |

### Production Example

```python
# Development - prioritize quality
dev_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={"output_dimensions": 1024}
)

# Production - balance quality and speed
prod_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={
        "output_dimensions": 512,  # 2x faster similarity search
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True
    }
)

# Mobile/Edge - prioritize speed
mobile_model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",
    config={"output_dimensions": 256}  # 4x faster, 75% less memory
)
```

## Recommended Models

### For General Use
- `intfloat/multilingual-e5-large-instruct` - Excellent general-purpose model
- `sentence-transformers/all-MiniLM-L6-v2` - Fast and lightweight
- `sentence-transformers/all-mpnet-base-v2` - Good balance of speed and quality

### For Advanced Features
- `Qwen/Qwen3-Embedding-4B` - Large context window (8192 tokens), best for late chunking
- `BAAI/bge-large-en-v1.5` - High-quality English embeddings
- `intfloat/e5-large-v2` - Strong performance across tasks

### Model Selection Guide

```python
# For long documents (late chunking)
long_doc_model = AIFactory.create_embedding(
    "transformers",
    "Qwen/Qwen3-Embedding-4B",  # 8192 token context
    config={"late_chunking": True}
)

# For multilingual support
multilingual_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct"
)

# For speed-critical applications
fast_model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",
    config={"output_dimensions": 128}  # Reduced dims for speed
)
```

## Configuration Options

Complete configuration reference for the Transformers provider:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | `EmbeddingTaskType` | `None` | Task-specific optimization |
| `late_chunking` | `bool` | `False` | Enable intelligent chunking |
| `output_dimensions` | `int` | `None` | Target embedding dimensions |
| `truncate_at_max_length` | `bool` | `True` | Truncate inputs at model limit |
| `device` | `str` | `"auto"` | Computation device (cpu/cuda/mps) |
| `pooling_strategy` | `str` | `"mean"` | Embedding pooling method |

### Example: Complete Configuration

```python
model = AIFactory.create_embedding(
    "transformers",
    "intfloat/multilingual-e5-large-instruct",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,
        "output_dimensions": 512,
        "device": "cuda",  # Use GPU if available
        "truncate_at_max_length": True,
        "pooling_strategy": "mean"
    }
)
```

## Error Handling

The provider gracefully handles missing optional dependencies:

### Missing sentence-transformers

```python
# When sentence-transformers is not available:
# - Late chunking falls back to simple sentence splitting
# - Task optimization still works with base class prefixes
# - Warning is logged but processing continues
```

### Missing scikit-learn

```python
# When scikit-learn is not available:
# - Dimension reduction falls back to simple truncation
# - Dimension expansion still works (zero padding)
# - Warning is logged for the user
```

### Handling Gracefully

```python
try:
    model = AIFactory.create_embedding(
        "transformers",
        "intfloat/e5-large-v2",
        config={
            "late_chunking": True,
            "output_dimensions": 256
        }
    )
    embeddings = model.embed(["text"])
except ImportError as e:
    print(f"Missing dependency: {e}")
    # Fall back to basic configuration
    model = AIFactory.create_embedding(
        "transformers",
        "intfloat/e5-large-v2"
    )
    embeddings = model.embed(["text"])
```

## Performance Considerations

### Memory Usage
- Large models like Qwen3-Embedding-4B require significant GPU memory
- Use CPU device for development and testing
- Consider smaller models for memory-constrained environments

```python
# For limited memory environments
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
    config={
        "device": "cpu",  # Force CPU usage
        "output_dimensions": 256  # Reduce dimensions
    }
)
```

### Processing Speed
- Late chunking adds processing overhead for long documents
- PCA dimension reduction has one-time fitting cost
- Batch processing is more efficient for multiple texts

```python
# Efficient batch processing
texts = ["text1", "text2", "text3", ...]
embeddings = model.embed(texts)  # Much faster than individual calls
```

### Model Loading
- Models are cached after first load using HuggingFace cache
- Set `HF_HOME` environment variable to control cache location
- Cold starts can be slow for large models

```bash
# Set custom cache directory
export HF_HOME=/path/to/cache
```

## Complete Example

Here's a complete example combining all advanced features:

```python
from esperanto import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Create advanced embedding model
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="Qwen/Qwen3-Embedding-4B",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,
        "output_dimensions": 512,
        "device": "cuda",  # Use GPU
    }
)

# Process documents
documents = [
    "Short document",
    "Very long document that exceeds the model context window..." * 100,
    "Another document with technical content"
]

# Generate optimized embeddings
# - Task prefix applied
# - Long documents automatically chunked
# - Dimensions reduced to 512
embeddings = model.embed(documents)

print(f"Generated {len(embeddings.data)} embeddings")
print(f"Dimensions: {len(embeddings.data[0].embedding)}")
```

## Troubleshooting

### Common Issues

**Import Error**: Missing optional dependencies
```bash
pip install esperanto[transformers]
```

**CUDA Out of Memory**: Large model on GPU
```python
config={"device": "cpu"}  # Force CPU usage
```

**Slow Performance**: Model loading time
- Models are cached after first load
- Use smaller models for development
- Consider using faster models like MiniLM

**Dimension Mismatch**: PCA fitting issues
- Ensure sufficient training data for PCA
- Check input embedding dimensions match model
- Verify output dimension targets are reasonable

### Debug Logging

Enable debug logging to see feature usage:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Will show task optimization, chunking, and dimension control logs
model = AIFactory.create_embedding(
    "transformers",
    "intfloat/e5-large-v2",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "late_chunking": True,
        "output_dimensions": 256
    }
)

embeddings = model.embed(["Debug this"])
```

## Integration with Universal Interface

All advanced features work seamlessly with Esperanto's universal interface:

```python
# Same config works across providers
config = {
    "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
    "late_chunking": True,
    "output_dimensions": 512
}

# Transformers (local emulation)
local_model = AIFactory.create_embedding(
    "transformers",
    "intfloat/e5-large-v2",
    config=config
)

# Jina (native support)
cloud_model = AIFactory.create_embedding(
    "jina",
    "jina-embeddings-v3",
    config=config
)

# Same interface, different implementation
texts = ["sample text"]
local_embeddings = local_model.embed(texts)
cloud_embeddings = cloud_model.embed(texts)
```

This ensures provider-agnostic development while maintaining the privacy and cost benefits of local processing.

## See Also

- [Transformers Provider Guide](/docs/providers/transformers.md) - Basic provider usage
- [Task-Aware Embeddings](/docs/advanced/task-aware-embeddings.md) - Deep dive on task types
- [Embedding Capabilities](/docs/capabilities/embeddings.md) - Overview of embedding features
- [Jina Provider](/docs/providers/jina.md) - Cloud alternative with native features
