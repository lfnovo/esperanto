# Embedding Models

Esperanto provides a unified interface for text embedding services across different providers. This guide explains how to use embedding models to convert text into vector representations.

## Interface

All embedding models implement the following interface:

```python
async def embed_query(self, text: str) -> List[float]:
    """Generate embeddings for a single piece of text."""

async def embed_documents(self, documents: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple documents."""

def to_langchain(self) -> Embeddings:
    """Convert to a LangChain embeddings model (requires langchain extra)."""
```

## Basic Usage

```python
from esperanto.factory import AIFactory

# Create an embedding model instance
embeddings = AIFactory.create_embeddings(
    provider="openai",  # Choose your provider
    model_name="text-embedding-3-small",  # Model name specific to the provider
)

# Generate embeddings for a single text
query_vector = await embeddings.embed_query("What is machine learning?")

# Generate embeddings for multiple documents
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural Language Processing deals with text"
]
document_vectors = await embeddings.embed_documents(documents)
```

## Supported Providers

### OpenAI
```python
embeddings = AIFactory.create_embeddings(
    provider="openai",
    model_name="text-embedding-3-small",  # or text-embedding-3-large
    config={
        "api_key": "your-api-key",  # Optional: defaults to OPENAI_API_KEY env var
        "organization": "org-id",    # Optional: your OpenAI organization ID
    }
)
```

### Cohere
```python
embeddings = AIFactory.create_embeddings(
    provider="cohere",
    model_name="embed-english-v3.0",  # or other Cohere embedding models
    config={
        "api_key": "your-api-key",  # Optional: defaults to COHERE_API_KEY env var
    }
)
```

### Google (Vertex AI)
```python
embeddings = AIFactory.create_embeddings(
    provider="vertex",
    model_name="textembedding-gecko",  # or other Vertex AI embedding models
    config={
        "project": "your-project",    # Optional: defaults to VERTEX_PROJECT env var
        "location": "us-central1",    # Optional: defaults to VERTEX_LOCATION env var
    }
)
```

## Vector Operations

Here are some common operations you might want to perform with embeddings:

### Cosine Similarity
```python
import numpy as np

def cosine_similarity(a: List[float], b: List[float]) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare two texts
text1 = "Machine learning is fascinating"
text2 = "AI is amazing"

vec1 = await embeddings.embed_query(text1)
vec2 = await embeddings.embed_query(text2)

similarity = cosine_similarity(vec1, vec2)
```

### Semantic Search
```python
from typing import List, Tuple
import numpy as np

def semantic_search(
    query: str,
    documents: List[str],
    embeddings_model,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """Search documents by semantic similarity."""
    # Generate embeddings
    query_embedding = await embeddings_model.embed_query(query)
    doc_embeddings = await embeddings_model.embed_documents(documents)
    
    # Calculate similarities
    similarities = [
        cosine_similarity(query_embedding, doc_embedding)
        for doc_embedding in doc_embeddings
    ]
    
    # Get top k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [
        (documents[i], similarities[i])
        for i in top_indices
    ]

# Example usage
documents = [
    "Machine learning models learn from data",
    "Neural networks are inspired by the brain",
    "Data science involves statistical analysis",
    "Python is a popular programming language"
]

query = "How do computers learn?"
results = await semantic_search(query, documents, embeddings)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc}")
```

## Vector Databases

Embeddings are commonly used with vector databases for efficient similarity search:

### Using with Chroma
```python
from chromadb import Client
import chromadb

# Initialize Chroma
client = chromadb.Client()
collection = client.create_collection("my_collection")

# Add documents with embeddings
documents = [
    "Machine learning is fascinating",
    "Neural networks are amazing",
    "Data science is important"
]
embeddings_list = await embeddings.embed_documents(documents)

# Add to collection
collection.add(
    embeddings=embeddings_list,
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Query
query = "How do computers learn?"
query_embedding = await embeddings.embed_query(query)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)
```

### Using with FAISS
```python
import faiss
import numpy as np

# Initialize FAISS index
dimension = 1536  # Depends on the embedding model
index = faiss.IndexFlatL2(dimension)

# Add vectors
documents = [
    "Machine learning is fascinating",
    "Neural networks are amazing",
    "Data science is important"
]
embeddings_list = await embeddings.embed_documents(documents)
index.add(np.array(embeddings_list))

# Search
query = "How do computers learn?"
query_embedding = await embeddings.embed_query(query)
D, I = index.search(np.array([query_embedding]), k=2)
```

## Best Practices

1. **Model Selection**:
   - Choose models based on your language and performance requirements
   - Consider dimensionality vs. accuracy trade-offs
   - Be aware of token limits and costs

2. **Performance**:
   - Batch documents when possible
   - Cache embeddings for frequently used texts
   - Use appropriate vector databases for large-scale operations

3. **Data Preparation**:
   - Clean and preprocess text before embedding
   - Consider text length limitations
   - Handle multilingual content appropriately

4. **Error Handling**:
   - Implement proper error handling for API failures
   - Handle rate limits and quotas
   - Consider fallback options

## Common Use Cases

1. **Semantic Search**:
   - Document retrieval
   - Question answering
   - Content recommendation

2. **Text Classification**:
   - Topic modeling
   - Sentiment analysis
   - Content categorization

3. **Similarity Analysis**:
   - Duplicate detection
   - Content clustering
   - Recommendation systems

4. **Information Retrieval**:
   - Document ranking
   - Relevance scoring
   - Knowledge base search
