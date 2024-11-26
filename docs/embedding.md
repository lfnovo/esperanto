# Embedding Providers

Esperanto supports multiple embedding providers for converting text into vector representations, useful for semantic search, text similarity, and other NLP tasks.

## Supported Providers

### OpenAI
- Models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- Features:
  - High-quality embeddings
  - Efficient dimensionality
  - Batch processing support

### Google Vertex AI (Gemini)
- Models: textembedding-gecko
- Features:
  - Enterprise-grade embeddings
  - Scalable API
  - Regional availability options

### Google Gemini
- Direct access to Gemini embedding models
- Features:
  - Fast inference
  - Competitive performance
  - Simple API

### Ollama
- Local embedding model deployment
- Support for various open-source models
- Features:
  - No API key required
  - Custom model configuration
  - Low latency for local deployments

## Usage Examples

### Basic Usage
```python
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel

model = OpenAIEmbeddingModel(
    api_key="your-api-key",
    model_name="text-embedding-3-small"  # optional
)

# Get embeddings for a single text
embedding = model.embed("Hello, world!")

# Get embeddings for multiple texts
embeddings = model.embed_many(["Hello, world!", "How are you?"])
```

### Local Deployment with Ollama
```python
from esperanto.providers.embedding.ollama import OllamaEmbeddingModel

model = OllamaEmbeddingModel(
    model_name="llama2",  # or any other supported model
    base_url="http://localhost:11434"  # default Ollama server
)

embedding = model.embed("Hello, world!")
```

### Google Vertex AI
```python
from esperanto.providers.embedding.vertex import VertexEmbeddingModel

model = VertexEmbeddingModel(
    project_id="your-project-id",
    location="us-central1"  # or your preferred region
)

embedding = model.embed("Hello, world!")
```

## Provider-Specific Configuration

Each provider may have specific configuration options. Here are some examples:

### OpenAI
```python
model = OpenAIEmbeddingModel(
    api_key="your-api-key",
    model_name="text-embedding-3-small",
    organization=None  # Optional, for org-specific API
)
```

### Gemini
```python
from esperanto.providers.embedding.gemini import GeminiEmbeddingModel

model = GeminiEmbeddingModel(
    api_key="your-api-key"
)
```
