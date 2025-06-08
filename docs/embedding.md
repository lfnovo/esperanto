# Embedding Models

Embedding models convert text into high-dimensional vector representations that capture semantic meaning. These vectors can be used for tasks like semantic search, similarity comparison, clustering, and recommendation systems. Esperanto provides a unified interface for working with various embedding providers.

## Supported Providers

- **OpenAI** (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- **Google** (Gemini embedding models)
- **Vertex AI** (textembedding-gecko)
- **Ollama** (Local deployment with various models)
- **Transformers** (Local Hugging Face models)
- **Voyage** (voyage-3, voyage-code-2)
- **Mistral** (mistral-embed)

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

## Provider-Specific Information

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