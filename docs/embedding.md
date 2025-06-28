# Embedding Models

Embedding models convert text into high-dimensional vector representations that capture semantic meaning. These vectors can be used for tasks like semantic search, similarity comparison, clustering, and recommendation systems. Esperanto provides a unified interface for working with various embedding providers.

## Supported Providers

- **Azure OpenAI** (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- **Google** (Gemini embedding models)
- **Jina** (jina-embeddings-v4, jina-embeddings-v3, with advanced task optimization)
- **Mistral** (mistral-embed)
- **OpenAI** (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- **Ollama** (Local deployment with various models)
- **Transformers** (Local Hugging Face models)
- **Vertex AI** (textembedding-gecko)
- **Voyage** (voyage-3, voyage-code-2)

## Available Methods

All embedding model providers implement the following methods:

- **`embed(texts)`**: Generate embeddings for text(s) - accepts single string or list of strings
- **`aembed(texts)`**: Async version of embed
- **`embed_query(text)`**: Generate embedding for a single query (alias for embed with single text)
- **`aembed_query(text)`**: Async version of embed_query

### Parameters:

- `texts`: Single string or list of strings to embed
- Returns: `EmbeddingResponse` object with embeddings and metadata

## Common Interface

All embedding models return standardized response objects:

### EmbeddingResponse

```python
response = model.embed(["Hello, world!", "Another text"])
# Access attributes:
response.data[0].embedding      # Vector for first text (list of floats)
response.data[0].index          # Index of the text (0)
response.data[1].embedding      # Vector for second text
response.model                  # Model used
response.provider               # Provider name
response.usage.total_tokens     # Token usage information
```

## Examples

### Basic Embedding

```python
from esperanto.factory import AIFactory

# Create an embedding model
model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Single text embedding
response = model.embed("Hello, world!")
vector = response.data[0].embedding  # List of floats

# Multiple texts
texts = ["Hello, world!", "How are you?", "Machine learning is fascinating"]
response = model.embed(texts)

for i, embedding_data in enumerate(response.data):
    print(f"Text {i}: {texts[i]}")
    print(f"Vector dimension: {len(embedding_data.embedding)}")
    print(f"First 5 values: {embedding_data.embedding[:5]}")
```

### Async Embedding

```python
async def embed_async():
    model = AIFactory.create_embedding("google", "text-embedding-004")

    texts = ["Document 1 content", "Document 2 content"]
    response = await model.aembed(texts)

    for data in response.data:
        print(f"Embedding dimension: {len(data.embedding)}")
```

### Semantic Search Example

```python
import numpy as np
from esperanto.factory import AIFactory

model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Documents to search
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms to learn patterns",
    "The weather is sunny today",
    "Neural networks are inspired by biological neurons"
]

# Create embeddings for documents
doc_response = model.embed(documents)
doc_embeddings = [data.embedding for data in doc_response.data]

# Query
query = "What is artificial intelligence?"
query_response = model.embed(query)
query_embedding = query_response.data[0].embedding

# Calculate similarity (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find most similar document
similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
most_similar_idx = np.argmax(similarities)

print(f"Most similar document: {documents[most_similar_idx]}")
print(f"Similarity score: {similarities[most_similar_idx]:.3f}")
```

### Batch Processing

```python
async def process_large_dataset():
    model = AIFactory.create_embedding("voyage", "voyage-3")

    # Process in batches to handle rate limits
    texts = ["text " + str(i) for i in range(1000)]
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await model.aembed(batch)
        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)

    print(f"Generated {len(all_embeddings)} embeddings")
```

## Task-Aware Embeddings

Esperanto supports advanced task-aware embeddings that optimize vector representations for specific use cases. This feature works across **all embedding providers** through a universal interface.

### Universal Task Types

- **`RETRIEVAL_QUERY`** - Optimize embeddings for search queries
- **`RETRIEVAL_DOCUMENT`** - Optimize embeddings for document storage
- **`SIMILARITY`** - General text similarity tasks
- **`CLASSIFICATION`** - Text classification tasks  
- **`CLUSTERING`** - Document clustering tasks
- **`CODE_RETRIEVAL`** - Code search optimization
- **`DEFAULT`** - No specific optimization

### Basic Task-Aware Usage

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Create task-optimized model
model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "output_dimensions": 512
    }
)

# Generate task-optimized embeddings
query = "What is machine learning?"
embeddings = model.embed([query])
```

### Advanced Features

```python
# Advanced configuration with all features
advanced_model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,           # Better long-context handling
        "output_dimensions": 1024,       # Control embedding size
        "truncate_at_max_length": True   # Handle long texts gracefully
    }
)

# Works with long documents
long_document = """
Large amounts of text that benefit from late chunking...
The model will handle context preservation automatically.
"""

embeddings = advanced_model.embed([long_document])
```

### Universal Interface Across Providers

The same configuration works with **any** embedding provider:

```python
# These all work with identical configuration!
providers_config = {
    "task_type": EmbeddingTaskType.CLASSIFICATION,
    "output_dimensions": 512
}

# Jina - Native API support
jina_model = AIFactory.create_embedding("jina", "jina-embeddings-v3", config=providers_config)

# OpenAI - Task prefixes emulation  
openai_model = AIFactory.create_embedding("openai", "text-embedding-3-small", config=providers_config)

# Google - Task translation
google_model = AIFactory.create_embedding("google", "text-embedding-004", config=providers_config)

# Transformers - Local emulation
local_model = AIFactory.create_embedding("transformers", "all-MiniLM-L6-v2", config=providers_config)

# All generate optimized embeddings for classification!
texts = ["This is a positive review", "This is a negative review"]
for model in [jina_model, openai_model, google_model, local_model]:
    embeddings = model.embed(texts)
    print(f"Generated {len(embeddings)} classification-optimized embeddings")
```

### RAG-Optimized Example

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Create specialized models for RAG pipeline
query_model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3", 
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "output_dimensions": 512
    }
)

document_model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,  # Better for long documents
        "output_dimensions": 512
    }
)

# Index documents with document-optimized embeddings
documents = [
    "Machine learning is a subset of artificial intelligence...",
    "Deep learning uses neural networks with multiple layers...",
    "Natural language processing deals with text understanding..."
]

doc_embeddings = document_model.embed(documents)

# Search with query-optimized embeddings
user_query = "What is deep learning?"
query_embedding = query_model.embed([user_query])[0]

# Calculate similarities for retrieval
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
best_match_idx = np.argmax(similarities)

print(f"Best match: {documents[best_match_idx]}")
print(f"Similarity: {similarities[best_match_idx]:.3f}")
```

### Provider-Specific Behavior

Different providers handle task-aware features differently:

| Provider | Task Types | Late Chunking | Output Dimensions | Implementation |
|----------|------------|---------------|-------------------|----------------|
| **Jina** | ✅ Native API | ✅ Native API | ✅ Native API | Full native support |
| **Google** | ✅ Translated | ❌ Emulated locally | ✅ Native API | API translation + emulation |
| **OpenAI** | ✅ Prefixes | ❌ Emulated locally | ✅ Native API | Prefix emulation + API |
| **Transformers** | ✅ Prefixes | ✅ Local algorithm | ❌ Model-dependent | Full local emulation |
| **Others** | ✅ Prefixes | ❌ Graceful skip | ❌ Graceful skip | Prefix emulation only |

### String Task Types

For convenience, you can also use strings instead of enums:

```python
# These are equivalent
model1 = AIFactory.create_embedding("jina", "jina-embeddings-v3", 
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY})

model2 = AIFactory.create_embedding("jina", "jina-embeddings-v3",
    config={"task_type": "retrieval.query"})  # String automatically converted
```

## Provider-Specific Information

### Jina Provider

The Jina provider offers the most advanced embedding capabilities with full native support for task-aware features.

**Configuration:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Basic usage
model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v3"  # or "jina-embeddings-v4"
)

# Advanced configuration with all features
model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v4",
    api_key="your-jina-api-key",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "late_chunking": True,
        "output_dimensions": 1024,
        "truncate_at_max_length": True
    }
)
```

**Environment Variables:**

Set this environment variable for automatic configuration:

```bash
JINA_API_KEY=your-jina-api-key
```

**Supported Models:**

- **jina-embeddings-v4** - Latest multimodal model (text + images)
- **jina-embeddings-v3** - High-performance multilingual model
- **jina-embeddings-v2-base-en** - English-optimized base model
- **jina-embeddings-v2-small-en** - Lightweight English model
- **jina-embeddings-v2-base-multilingual** - Multilingual base model
- **jina-clip-v1** - Multimodal text/image model
- **jina-clip-v2** - Latest multimodal model

**Native Features:**

- ✅ **Task Optimization** - Native API support for all task types
- ✅ **Late Chunking** - Advanced context preservation for long texts
- ✅ **Output Dimensions** - Configurable embedding dimensions
- ✅ **Multilingual** - Superior performance across 100+ languages
- ✅ **8K Context** - Long context window support

**Task Type Mapping:**

```python
# Universal → Jina API mapping
EmbeddingTaskType.RETRIEVAL_QUERY → "retrieval.query"
EmbeddingTaskType.RETRIEVAL_DOCUMENT → "retrieval.passage"
EmbeddingTaskType.SIMILARITY → "text-matching"
EmbeddingTaskType.CLASSIFICATION → "classification"
EmbeddingTaskType.CLUSTERING → "separation"
EmbeddingTaskType.CODE_RETRIEVAL → "code.query"
```

**Example:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Create Jina model with advanced features
model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v4",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,
        "output_dimensions": 512
    }
)

# Generate optimized embeddings
documents = [
    "Long document that benefits from late chunking context preservation...",
    "Another document with important semantic content..."
]
embeddings = model.embed(documents)

print(f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
```

### Azure OpenAI Provider

The Azure OpenAI provider supports Azure's embedding models through your Azure OpenAI deployment.

**Configuration:**

```python
from esperanto.factory import AIFactory

# Using environment variables
model = AIFactory.create_embedding(
    provider="azure",
    model_name="your-deployment-name"  # Your Azure deployment name
)

# Or with explicit configuration
model = AIFactory.create_embedding(
    provider="azure",
    model_name="your-deployment-name",
    api_key="your-azure-api-key",
    base_url="https://your-resource.openai.azure.com",
    api_version="2024-12-01-preview"
)
```

**Environment Variables:**

Set these environment variables for automatic configuration:

```bash
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

**Supported Models:**

- text-embedding-3-small
- text-embedding-3-large
- text-embedding-ada-002

**Features:**

- Enhanced text preprocessing (based on Microsoft recommendations)
- Custom dimension support
- Async support
- Automatic text cleaning and normalization

**Example:**

```python
from esperanto.factory import AIFactory

# Create Azure embedding model
model = AIFactory.create_embedding("azure", "text-embedding-3-small")

# Generate embeddings with custom dimensions
texts = ["Hello, world!", "How are you?"]
embeddings = model.embed(texts, dimensions=1024)

# The Azure provider automatically cleans and normalizes text for optimal results
```

### Transformers Provider

The Transformers provider requires the transformers extra to be installed:

```bash
pip install "esperanto[transformers]"
```

This installs:

- `transformers`
- `torch`
- `tokenizers`

**Advanced Configuration:**

```python
from esperanto.factory import AIFactory

# Basic usage
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Advanced configuration
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="auto",  # 'auto', 'cpu', 'cuda', or 'mps'
    pooling_strategy="mean",  # 'mean', 'max', or 'cls'
    quantize="8bit",  # optional: '4bit' or '8bit' for memory efficiency
    tokenizer_config={
        "max_length": 512,
        "padding": True,
        "truncation": True
    }
)

# Example with multilingual model
multilingual_model = AIFactory.create_embedding(
    provider="transformers",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    tokenizer_config={
        "max_length": 256,  # Shorter for memory efficiency
        "padding": True,
        "truncation": True
    }
)

# Pooling strategies:
# - "mean": Average of all token embeddings (default, good for semantic similarity)
# - "max": Maximum value across token embeddings (good for key feature extraction)
# - "cls": Use the [CLS] token embedding (good for sentence classification)
```

**GPU and Quantization:**

```python
# Use GPU if available
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cuda"  # or "mps" for Apple Silicon
)

# Use quantization for large models
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="BAAI/bge-large-en-v1.5",
    quantize="8bit",  # Reduces memory usage
    device="cuda"
)
```
