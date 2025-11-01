# Configuration Guide

This guide covers all configuration options for Esperanto, including environment variables, parameters, and best practices.

## Environment Variables

Esperanto uses environment variables for API keys and provider configuration. The complete reference is in `.env.example` at the project root.

### Quick Setup

```bash
# Copy example file
cp .env.example .env

# Edit with your credentials
nano .env
```

### Using Environment Variables

**Option 1: Export in shell**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Option 2: .env file with python-dotenv**
```python
from dotenv import load_dotenv
load_dotenv()

# Now Esperanto can access the variables
from esperanto.factory import AIFactory
model = AIFactory.create_language("openai", "gpt-4")
```

**Option 3: Direct configuration (not recommended for production)**
```python
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"api_key": "sk-..."}
)
```

## Provider Configuration

### Cloud API Providers

#### OpenAI
```bash
OPENAI_API_KEY=sk-...
```

```python
model = AIFactory.create_language("openai", "gpt-4", config={
    "api_key": "sk-...",  # Or from env var
    "organization": "org-...",  # Optional
    "base_url": "https://api.openai.com/v1",  # Optional custom endpoint
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 60.0
})
```

→ **[Full OpenAI Setup Guide](./providers/openai.md)**

#### Anthropic
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

```python
model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022", config={
    "api_key": "sk-ant-...",  # Or from env var
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 60.0
})
```

→ **[Full Anthropic Setup Guide](./providers/anthropic.md)**

#### Google (GenAI)
```bash
GOOGLE_API_KEY=...
# Optional: Override base URL
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com
```

```python
model = AIFactory.create_language("google", "gemini-pro", config={
    "api_key": "...",  # Or from env var
    "timeout": 60.0
})
```

→ **[Full Google Setup Guide](./providers/google.md)**

#### Groq
```bash
GROQ_API_KEY=...
```

→ **[Full Groq Setup Guide](./providers/groq.md)**

#### Mistral
```bash
MISTRAL_API_KEY=...
```

→ **[Full Mistral Setup Guide](./providers/mistral.md)**

#### DeepSeek
```bash
DEEPSEEK_API_KEY=...
```

→ **[Full DeepSeek Setup Guide](./providers/deepseek.md)**

#### Perplexity
```bash
PERPLEXITY_API_KEY=...
```

→ **[Full Perplexity Setup Guide](./providers/perplexity.md)**

#### xAI
```bash
XAI_API_KEY=...
```

→ **[Full xAI Setup Guide](./providers/xai.md)**

#### OpenRouter
```bash
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

→ **[Full OpenRouter Setup Guide](./providers/openrouter.md)**

#### Jina
```bash
JINA_API_KEY=...
```

→ **[Full Jina Setup Guide](./providers/jina.md)**

#### Voyage
```bash
VOYAGE_API_KEY=...
```

→ **[Full Voyage Setup Guide](./providers/voyage.md)**

#### ElevenLabs
```bash
ELEVENLABS_API_KEY=...
```

→ **[Full ElevenLabs Setup Guide](./providers/elevenlabs.md)**

### Enterprise Providers

#### Azure OpenAI

**Generic configuration (works for all modalities):**
```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-01
```

**Modality-specific configuration (takes precedence):**
```bash
# For LLM
AZURE_OPENAI_API_KEY_LLM=...
AZURE_OPENAI_ENDPOINT_LLM=...
AZURE_OPENAI_API_VERSION_LLM=...

# For Embeddings
AZURE_OPENAI_API_KEY_EMBEDDING=...
AZURE_OPENAI_ENDPOINT_EMBEDDING=...
AZURE_OPENAI_API_VERSION_EMBEDDING=...

# For Speech-to-Text
AZURE_OPENAI_API_KEY_STT=...
AZURE_OPENAI_ENDPOINT_STT=...
AZURE_OPENAI_API_VERSION_STT=...

# For Text-to-Speech
AZURE_OPENAI_API_KEY_TTS=...
AZURE_OPENAI_ENDPOINT_TTS=...
AZURE_OPENAI_API_VERSION_TTS=...
```

**Priority order:**
1. Modality-specific variables (highest)
2. Generic variables
3. Legacy variables (backward compatibility)

→ **[Full Azure Setup Guide](./providers/azure.md)**

#### Vertex AI

```bash
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-east5
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

→ **[Full Vertex AI Setup Guide](./providers/vertex.md)**

### Local/Self-Hosted Providers

#### Ollama

```bash
OLLAMA_BASE_URL=http://localhost:11434
```

No API key needed. Requires Ollama installed locally.

→ **[Full Ollama Setup Guide](./providers/ollama.md)**

#### Transformers

```bash
# Optional: HuggingFace token for private/gated models
HF_TOKEN=...
```

No API key needed. Models downloaded from HuggingFace.

→ **[Full Transformers Setup Guide](./providers/transformers.md)**

#### OpenAI-Compatible

**Generic configuration (works for all modalities):**
```bash
OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1
OPENAI_COMPATIBLE_API_KEY=...  # Optional, depends on endpoint
```

**Modality-specific configuration:**
```bash
# For LLM
OPENAI_COMPATIBLE_BASE_URL_LLM=http://localhost:1234/v1
OPENAI_COMPATIBLE_API_KEY_LLM=...

# For Embeddings
OPENAI_COMPATIBLE_BASE_URL_EMBEDDING=http://localhost:8080/v1
OPENAI_COMPATIBLE_API_KEY_EMBEDDING=...

# For Speech-to-Text
OPENAI_COMPATIBLE_BASE_URL_STT=http://localhost:9000/v1
OPENAI_COMPATIBLE_API_KEY_STT=...

# For Text-to-Speech
OPENAI_COMPATIBLE_BASE_URL_TTS=http://localhost:7000/v1
OPENAI_COMPATIBLE_API_KEY_TTS=...
```

**Use cases:**
- LM Studio (local LLM server)
- vLLM (high-performance inference)
- Local Ollama with OpenAI compatibility
- Custom OpenAI-compatible endpoints

→ **[Full OpenAI-Compatible Setup Guide](./providers/openai-compatible.md)**

## Timeout Configuration

Control request timeouts globally or per-provider-type.

### Default Timeouts

- **LLM, Embedding, Reranking:** 60 seconds
- **Speech-to-Text, Text-to-Speech:** 300 seconds (5 minutes)

### Global Timeout Configuration

Set via environment variables:

```bash
# Override defaults for all providers
ESPERANTO_LLM_TIMEOUT=90           # 90 seconds for LLMs
ESPERANTO_EMBEDDING_TIMEOUT=120    # 2 minutes for embeddings
ESPERANTO_RERANKER_TIMEOUT=75      # 75 seconds for rerankers
ESPERANTO_STT_TIMEOUT=600          # 10 minutes for STT
ESPERANTO_TTS_TIMEOUT=400          # 6.5 minutes for TTS
```

### Per-Instance Configuration

Override via config parameter:

```python
# Via config dictionary (highest priority)
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 120.0}  # 2 minutes
)

# For STT/TTS, also via direct parameter
transcriber = AIFactory.create_speech_to_text(
    "openai",
    timeout=600.0  # 10 minutes
)
```

### Priority Order

1. **Config parameter** (highest priority)
2. **Environment variable** (`ESPERANTO_*_TIMEOUT`)
3. **Provider type default** (60s or 300s)

→ **[Full Timeout Configuration Guide](./advanced/timeout-configuration.md)**

## Common Parameters

### Language Models (LLM)

```python
model = AIFactory.create_language(
    provider="openai",
    model_name="gpt-4",
    config={
        # Sampling parameters
        "temperature": 0.7,      # 0.0-2.0, creativity
        "top_p": 0.9,           # Nucleus sampling
        "max_tokens": 1000,     # Response length limit

        # Output format
        "streaming": False,      # Enable token-by-token streaming
        "structured": {"type": "json"},  # JSON output mode

        # Performance
        "timeout": 60.0,        # Request timeout in seconds

        # Authentication (if not using env vars)
        "api_key": "...",

        # Provider-specific
        "organization": "...",  # OpenAI only
        "base_url": "...",      # Custom endpoints
    }
)
```

### Embeddings

```python
embedder = AIFactory.create_embedding(
    provider="openai",
    model_name="text-embedding-3-small",
    config={
        # Performance
        "timeout": 60.0,
        "batch_size": 32,       # Texts per request

        # Advanced (provider-specific)
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,  # Jina, Google
        "late_chunking": True,  # Jina only
        "output_dimensions": 512,  # Jina, OpenAI (some models)

        # Authentication
        "api_key": "...",
    }
)
```

### Reranking

```python
reranker = AIFactory.create_reranker(
    provider="jina",
    model_name="jina-reranker-v2-base-multilingual",
    config={
        "timeout": 60.0,
        "api_key": "...",
    }
)
```

### Speech-to-Text

```python
transcriber = AIFactory.create_speech_to_text(
    provider="openai",
    model_name="whisper-1",
    config={
        "timeout": 300.0,       # Longer for audio processing
        "language": "en",       # ISO-639-1 code
        "response_format": "json",  # "json", "text", "srt", "vtt", "verbose_json"
        "temperature": 0.0,     # Sampling (0.0 = deterministic)
        "api_key": "...",
    }
)
```

### Text-to-Speech

```python
speaker = AIFactory.create_text_to_speech(
    provider="openai",
    model_name="tts-1",
    config={
        "timeout": 300.0,       # Longer for audio generation
        "voice": "nova",        # Default voice
        "speed": 1.0,           # 0.25-4.0, speech rate
        "response_format": "mp3",  # "mp3", "opus", "aac", "flac", "wav", "pcm"
        "api_key": "...",
    }
)
```

## Configuration Patterns

### Development vs Production

**Development:**
```python
# .env.development
OPENAI_API_KEY=...
ESPERANTO_LLM_TIMEOUT=30  # Faster timeouts for testing
```

**Production:**
```python
# .env.production
OPENAI_API_KEY=...
ESPERANTO_LLM_TIMEOUT=120  # Longer timeouts for reliability
```

Load environment-specific config:

```python
import os
from dotenv import load_dotenv

env = os.getenv("ENV", "development")
load_dotenv(f".env.{env}")
```

### Multi-Environment Setup

```python
import os

def get_model():
    """Get model based on environment."""
    if os.getenv("ENV") == "production":
        # Production: OpenAI for quality
        return AIFactory.create_language("openai", "gpt-4")
    else:
        # Development: Ollama for cost savings
        return AIFactory.create_language("ollama", "llama3.2")

model = get_model()
```

### Provider Fallback

```python
def create_llm_with_fallback():
    """Try primary provider, fall back to secondary."""
    try:
        return AIFactory.create_language("openai", "gpt-4", config={"timeout": 30.0})
    except Exception as e:
        print(f"Primary failed: {e}, falling back to Groq")
        return AIFactory.create_language("groq", "mixtral-8x7b-32768")

model = create_llm_with_fallback()
```

### Multi-Provider Configuration

```python
# .env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
JINA_API_KEY=...
ELEVENLABS_API_KEY=...

# Use best provider for each task
llm = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
embedder = AIFactory.create_embedding("jina", "jina-embeddings-v3")
speaker = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")
```

## Best Practices

### Security

**DO:**
- ✅ Use environment variables for API keys
- ✅ Use `.env` file in development (add to `.gitignore`)
- ✅ Use secret management in production (AWS Secrets Manager, etc.)
- ✅ Rotate API keys regularly
- ✅ Use least-privilege keys when possible

**DON'T:**
- ❌ Hard-code API keys in source code
- ❌ Commit `.env` files to version control
- ❌ Share API keys in logs or error messages
- ❌ Use production keys in development

### Performance

- **Timeouts:** Set appropriate timeouts based on expected operation duration
- **Batch Size:** Increase for embeddings when processing many texts
- **Async:** Use async methods for concurrent requests
- **Caching:** Cache model instances when possible (AIFactory does this automatically)

### Error Handling

```python
from esperanto.factory import AIFactory

try:
    model = AIFactory.create_language("openai", "gpt-4")
    response = model.chat_complete(messages)
except ValueError as e:
    # Configuration errors (invalid parameters, missing API key)
    print(f"Configuration error: {e}")
except TimeoutError as e:
    # Request timeout
    print(f"Request timed out: {e}")
except Exception as e:
    # Other errors (network, API errors, etc.)
    print(f"Error: {e}")
```

### Testing

**Use different providers for test vs production:**

```python
# conftest.py
import pytest
import os

@pytest.fixture
def llm():
    if os.getenv("CI"):
        # CI: Use mock or free provider
        return MockLLM()
    else:
        # Local: Use real provider
        return AIFactory.create_language("openai", "gpt-4")
```

## Validation

Esperanto validates configuration parameters:

- **Timeout:** Must be 1-3600 seconds
- **Temperature:** Must be 0.0-2.0 (LLM)
- **API Keys:** Must be provided (via env var or config)
- **Model Names:** Validated against available models (where possible)

Invalid configuration raises `ValueError` with descriptive message.

## Complete .env Example

See `.env.example` in project root for the complete, up-to-date reference with all providers and options.

```bash
# Copy to get started
cp .env.example .env
```

## See Also

- **[Provider Comparison](./providers/README.md)** - Choose your providers
- **[Provider Setup Guides](./providers/)** - Detailed setup per provider
- **[Timeout Configuration](./advanced/timeout-configuration.md)** - Advanced timeout options
- **[Quick Start Guide](./quickstart.md)** - Get started quickly

---

**Need help with a specific provider?** → Check [Provider Setup Guides](./providers/)
