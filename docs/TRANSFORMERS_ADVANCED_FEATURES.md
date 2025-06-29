# Transformers Advanced Features

The Esperanto Transformers provider now supports advanced embedding features that were previously only available through cloud providers. These features provide privacy-first alternatives with complete local processing.

## Overview

The enhanced Transformers provider includes:

- **Task-Specific Optimization**: Advanced prefixes optimized for different use cases
- **Semantic Late Chunking**: Intelligent text segmentation for long documents  
- **Output Dimension Control**: PCA-based reduction and zero-padding expansion
- **Model-Aware Configuration**: Optimized settings for different transformer models
- **Graceful Degradation**: Fallback behavior when optional dependencies are missing

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

Task optimization applies sophisticated prefixes that help models understand the intended use case.

### Supported Task Types

| Task Type | Description | Prefix |
|-----------|-------------|--------|
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
```

## Late Chunking

Late chunking intelligently segments long texts that exceed the model's context window, then aggregates the results.

### Features

- **Semantic Boundary Detection**: Uses sentence-transformers to find natural break points
- **Model-Aware Limits**: Automatically configures chunk sizes based on the model
- **Mean Aggregation**: Combines chunk embeddings using mean pooling
- **Fallback Support**: Simple sentence-based chunking when advanced features unavailable

### Model-Specific Limits

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
long_document = "Very long text..." * 1000
embeddings = model.embed([long_document])  # Returns single embedding
```

## Output Dimension Control

Control the dimensionality of output embeddings through reduction or expansion.

### Dimension Reduction (PCA)

Uses Principal Component Analysis to reduce embedding dimensions while preserving the most important information.

```python
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims
    config={"output_dimensions": 128}  # Reduce to 128
)

embeddings = model.embed(["Text"])
print(len(embeddings[0]))  # 128
```

### Dimension Expansion (Zero Padding)

Expands embeddings by adding zero-valued dimensions.

```python
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims  
    config={"output_dimensions": 512}  # Expand to 512
)

embeddings = model.embed(["Text"])
print(len(embeddings[0]))  # 512
```

## Recommended Models

### For General Use
- `intfloat/multilingual-e5-large-instruct` - Excellent general-purpose model
- `sentence-transformers/all-MiniLM-L6-v2` - Fast and lightweight

### For Advanced Features
- `Qwen/Qwen3-Embedding-4B` - Large context window (8192 tokens)
- `BAAI/bge-large-en-v1.5` - High-quality English embeddings
- `intfloat/e5-large-v2` - Strong performance across tasks

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | `EmbeddingTaskType` | `None` | Task-specific optimization |
| `late_chunking` | `bool` | `False` | Enable intelligent chunking |
| `output_dimensions` | `int` | `None` | Target embedding dimensions |
| `truncate_at_max_length` | `bool` | `True` | Truncate inputs at model limit |
| `device` | `str` | `"auto"` | Computation device |
| `pooling_strategy` | `str` | `"mean"` | Embedding pooling method |

## Error Handling

The provider gracefully handles missing optional dependencies:

```python
# When sentence-transformers is not available:
# - Late chunking falls back to simple sentence splitting  
# - Task optimization still works with base class prefixes

# When scikit-learn is not available:
# - Dimension reduction falls back to simple truncation
# - Warnings are logged for the user
```

## Performance Considerations

### Memory Usage
- Large models like Qwen3-Embedding-4B require significant GPU memory
- Use CPU device for development and testing
- Consider quantization for memory-constrained environments

### Processing Speed  
- Late chunking adds processing overhead for long documents
- PCA dimension reduction has one-time fitting cost
- Batch processing is more efficient for multiple texts

### Model Loading
- Models are cached after first load using HuggingFace cache
- Set `model_cache_dir` to control cache location
- Cold starts can be slow for large models

## Examples

See `examples/transformers_advanced_features.py` for comprehensive usage examples including:

1. Task-specific optimization for different use cases
2. Late chunking with long documents
3. Dimension control (reduction and expansion)  
4. Combined advanced features
5. Qwen3-Embedding-4B large context handling

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
- Consider quantization options

**Dimension Mismatch**: PCA fitting issues
- Ensure sufficient training data for PCA
- Check input embedding dimensions
- Verify output dimension targets

### Debug Logging

Enable debug logging to see feature usage:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Will show task optimization, chunking, and dimension control logs
model.embed(["Text"])
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
local_model = AIFactory.create_embedding("transformers", model, config=config)

# Jina (native support)  
cloud_model = AIFactory.create_embedding("jina", model, config=config)

# Same interface, different implementation
embeddings1 = local_model.embed(texts)
embeddings2 = cloud_model.embed(texts)
```

This ensures provider-agnostic development while maintaining the privacy and cost benefits of local processing.