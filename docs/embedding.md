# Embedding Providers

Esperanto supports multiple embedding providers for converting text into vector representations.

## Supported Providers

- OpenAI (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- Google Vertex AI (textembedding-gecko)
- Google GenAI
- Ollama (Local deployment)
- Transformers (Local deployment with Hugging Face models)
- Voyage AI (voyage-large-2, voyage-code-2)
- Mistral (mistral-embed, etc.) [NEW]

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

### Local Deployment with Transformers
```python
from esperanto.factory import AIFactory

# Basic usage with defaults
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="bert-base-uncased",  # or any other Hugging Face model
)

# Advanced configuration
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="bert-base-uncased",  # or any other Hugging Face model
    config={
        "device": "auto",  # 'auto', 'cpu', 'cuda', or 'mps'
        "pooling_strategy": "mean",  # 'mean', 'max', or 'cls'
        "quantize": "8bit",  # optional: '4bit' or '8bit'
        "tokenizer_config": {  # optional tokenizer configuration
            "max_length": 512,  # maximum sequence length (default for BERT)
            "padding": True,
            "truncation": True
        }
    }
)

# Example with a multilingual model
model = AIFactory.create_embedding(
    provider="transformers",
    model_name="neuralmind/bert-base-portuguese-cased",  # Portuguese BERT
    config={
        "tokenizer_config": {
            "max_length": 256,  # shorter for memory efficiency
            "padding": True,
            "truncation": True
        }
    }
)

embeddings = model.embed(["Hello, world!"])

# Pooling Strategies:
# - "mean": Average of all token embeddings (default, good for semantic similarity)
# - "max": Maximum value across token embeddings (good for key feature extraction)
# - "cls": Use the [CLS] token embedding (good for sentence classification)
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

### Voyage AI
```python
from esperanto.factory import AIFactory

# Basic usage
model = AIFactory.create_embedding(
    provider="voyage",
    model_name="voyage-3",  # or voyage-code-2 for code embeddings
    api_key="your-api-key"  # or set VOYAGE_API_KEY env var
)

# Get embeddings
texts = ["Hello, world!", "Another text"]
embeddings = model.embed(texts)

# Async usage
async def get_embeddings():
    embeddings = await model.aembed(texts)
```
