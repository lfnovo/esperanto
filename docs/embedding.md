# Embedding Providers

Esperanto supports multiple embedding providers for converting text into vector representations.

## Supported Providers

- OpenAI (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- Google Vertex AI (textembedding-gecko)
- Google GenAI
- Ollama (Local deployment)

## Usage Examples

### Using AI Factory

```python
from esperanto.factory import AIFactory

# Create an embedding instance
model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Synchronous usage
texts = ["Hello, world!", "Another text"]
embeddings = model.embed(texts)

# Asynchronous usage
async def get_embeddings():
    texts = ["Hello, world!", "Another text"]
    embeddings = await model.aembed(texts)
```

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
    model_name="mxbai-embed-large",  # or any other supported model
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
    api_key="your-api-key", # or use ENV
    model_name="text-embedding-3-small",
    organization=None  # Optional, for org-specific API
)
```

### Google GenAI
```python
from esperanto.providers.embedding.google import GoogleEmbeddingModel

model = GoogleEmbeddingModel(
    api_key="your-api-key" # or use ENV
)
```
