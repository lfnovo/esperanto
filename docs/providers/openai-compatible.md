# OpenAI-Compatible

## Overview

The OpenAI-Compatible provider enables you to use any service that implements the OpenAI API format. This includes local deployments, custom endpoints, and third-party services, giving you maximum flexibility while maintaining Esperanto's unified interface.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Any OpenAI-compatible chat completions endpoint |
| Embeddings | ✅ | Any OpenAI-compatible embeddings endpoint |
| Reranking | ❌ | Not available |
| Speech-to-Text | ✅ | Any OpenAI-compatible transcription endpoint |
| Text-to-Speech | ✅ | Any OpenAI-compatible TTS endpoint |

**Official Documentation:** Varies by implementation

## Prerequisites

### Supported Endpoints

**Local Deployments:**
- [LM Studio](https://lmstudio.ai/) - User-friendly local model server with GUI
- [Ollama](https://ollama.ai/) - Simple local model deployment (via OpenAI-compatible mode)
- [vLLM](https://docs.vllm.ai/) - High-performance inference server
- [LocalAI](https://localai.io/) - Self-hosted OpenAI alternative
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) - Popular WebUI for local LLMs

**Speech Services:**
- [Speaches](https://github.com/speaches-ai/speaches/) - OpenAI-compatible server for faster-whisper (STT) and Piper/Kokoro (TTS)
- [Shabdabhav](https://github.com/Hardik94/shabdabhav) - OpenAI-compatible server for Piper and Parler TTS models

**Cloud Services:**
- Custom OpenAI-format APIs
- Self-hosted embedding services
- Edge computing deployments

### Installation Requirements

Requirements vary by endpoint. For LM Studio:
1. Download from https://lmstudio.ai
2. Load a model (LLM or embedding)
3. Start the local server (default: `http://localhost:1234`)

## Environment Variables

```bash
# Generic (works for all OpenAI-compatible providers)
OPENAI_COMPATIBLE_BASE_URL="http://localhost:1234/v1"
OPENAI_COMPATIBLE_API_KEY="your-api-key"  # Optional for local endpoints

# Provider-specific (takes precedence, new in v2.7.0)
# Use these when you want different endpoints for different AI capabilities
OPENAI_COMPATIBLE_BASE_URL_LLM="http://localhost:1234/v1"
OPENAI_COMPATIBLE_API_KEY_LLM="your-key"

OPENAI_COMPATIBLE_BASE_URL_EMBEDDING="http://localhost:8080/v1"
OPENAI_COMPATIBLE_API_KEY_EMBEDDING="your-key"

OPENAI_COMPATIBLE_BASE_URL_STT="http://localhost:9000/v1"
OPENAI_COMPATIBLE_API_KEY_STT="your-key"

OPENAI_COMPATIBLE_BASE_URL_TTS="http://localhost:7000/v1"
OPENAI_COMPATIBLE_API_KEY_TTS="your-key"
```

**Configuration Precedence** (highest to lowest):
1. Direct parameters in config dictionary (`base_url=`, `api_key=`)
2. Provider-specific environment variables (`OPENAI_COMPATIBLE_BASE_URL_LLM`, etc.)
3. Generic environment variables (`OPENAI_COMPATIBLE_BASE_URL`, etc.)
4. Default values

This allows you to use different OpenAI-compatible endpoints for different AI capabilities (LLM, Embedding, STT, TTS) without code changes.

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Language model
model = AIFactory.create_language(
    "openai-compatible",
    "your-model-name",
    config={"base_url": "http://localhost:1234/v1"}
)

# Embedding model
embedder = AIFactory.create_embedding(
    "openai-compatible",
    "nomic-embed-text",
    config={"base_url": "http://localhost:1234/v1"}
)

# Speech-to-text
transcriber = AIFactory.create_speech_to_text(
    "openai-compatible",
    "faster-whisper-large-v3",
    config={"base_url": "http://localhost:8000"}
)

# Text-to-speech
speaker = AIFactory.create_text_to_speech(
    "openai-compatible",
    "piper-tts",
    config={"base_url": "http://localhost:8000"}
)
```

### Direct Instantiation

```python
from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel
from esperanto.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingModel
from esperanto.providers.speech_to_text.openai_compatible import OpenAICompatibleSpeechToText
from esperanto.providers.text_to_speech.openai_compatible import OpenAICompatibleTextToSpeech

# Language model
llm = OpenAICompatibleLanguageModel(
    base_url="http://localhost:1234/v1",
    api_key="not-required",  # Often not needed
    model_name="your-model-name"
)

# Embedding model
embedder = OpenAICompatibleEmbeddingModel(
    base_url="http://localhost:1234/v1",
    model_name="nomic-embed-text"
)

# Speech-to-text
stt = OpenAICompatibleSpeechToText(
    base_url="http://localhost:8000",
    model_name="faster-whisper-large-v3"
)

# Text-to-speech
tts = OpenAICompatibleTextToSpeech(
    base_url="http://localhost:8000",
    model_name="piper-tts"
)
```

## Capabilities

### Language Models (LLM)

**Common Endpoints:**
- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`
- vLLM: `http://localhost:8000/v1`
- Custom: `https://your-endpoint.com/v1`

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "openai-compatible",
    "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    config={
        "base_url": "http://localhost:1234/v1",  # Required
        "api_key": "lm-studio",                   # Optional
        "temperature": 0.7,                       # Standard OpenAI params
        "max_tokens": 1000,
        "streaming": True
    }
)
```

**Example - LM Studio:**

```python
from esperanto.factory import AIFactory

# Connect to LM Studio running locally
model = AIFactory.create_language(
    "openai-compatible",
    "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    config={
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio"
    }
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

# Regular completion
response = model.chat_complete(messages)
print(response.choices[0].message.content)

# Streaming completion
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Example - vLLM:**

```python
# Connect to vLLM server
model = AIFactory.create_language(
    "openai-compatible",
    "facebook/opt-125m",
    config={
        "base_url": "http://localhost:8000/v1",
        "temperature": 0.8
    }
)

response = model.chat_complete(messages)
```

**Example - Custom Endpoint:**

```python
# Use any OpenAI-compatible endpoint
model = AIFactory.create_language(
    "openai-compatible",
    "custom-model",
    config={
        "base_url": "https://api.yourcompany.com/v1",
        "api_key": "your-service-token"
    }
)

response = model.chat_complete(messages)
```

**Example - JSON Mode:**

```python
# Note: JSON mode support depends on endpoint implementation
model = AIFactory.create_language(
    "openai-compatible",
    "your-model",
    config={
        "base_url": "http://localhost:1234/v1",
        "structured": {"type": "json"}
    }
)

messages = [{
    "role": "user",
    "content": "List three programming languages as JSON"
}]

response = model.chat_complete(messages)
```

### Embeddings

**Common Models (via LM Studio/LocalAI):**
- nomic-embed-text (768 dimensions)
- all-MiniLM-L6-v2 (384 dimensions)
- e5-large-v2 (1024 dimensions)
- bge-large-en-v1.5 (1024 dimensions)

**Configuration:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

embedder = AIFactory.create_embedding(
    "openai-compatible",
    "nomic-embed-text",
    config={
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-required",              # Often not needed
        "timeout": 120,                          # Default: 120 seconds
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT  # Task optimization via prefixes
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
embedder = AIFactory.create_embedding(
    "openai-compatible",
    "nomic-embed-text",
    config={"base_url": "http://localhost:1234/v1"}
)

# Generate embeddings
texts = ["Hello, world!", "Local embeddings are great"]
response = embedder.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
```

**Example - Large Batch Processing:**

```python
# Increase timeout for large batches
embedder = AIFactory.create_embedding(
    "openai-compatible",
    "nomic-embed-text",
    config={
        "base_url": "http://localhost:1234/v1",
        "timeout": 300  # 5 minutes for large batches
    }
)

# Process many documents
large_batch = [f"Document {i}" for i in range(1000)]
response = embedder.embed(large_batch)
```

**Example - Task-Optimized:**

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Query embeddings
query_model = AIFactory.create_embedding(
    "openai-compatible",
    "nomic-embed-text",
    config={
        "base_url": "http://localhost:1234/v1",
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY
    }
)

# Document embeddings
doc_model = AIFactory.create_embedding(
    "openai-compatible",
    "nomic-embed-text",
    config={
        "base_url": "http://localhost:1234/v1",
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)
```

### Speech-to-Text

**Supported Implementations:**
- [Speaches](https://github.com/speaches-ai/speaches/) - faster-whisper models
- Custom faster-whisper deployments
- OpenAI Whisper servers
- Custom STT endpoints

**Configuration:**

```python
from esperanto.factory import AIFactory

transcriber = AIFactory.create_speech_to_text(
    "openai-compatible",
    "faster-whisper-large-v3",
    config={
        "base_url": "http://localhost:8000",
        "api_key": "your-api-key",  # Optional
        "timeout": 600               # 10 minutes for large files
    }
)
```

**Example - Basic Transcription:**

```python
from esperanto.factory import AIFactory

# Create STT model
stt = AIFactory.create_speech_to_text(
    "openai-compatible",
    "faster-whisper-large-v3",
    config={"base_url": "http://localhost:8000"}
)

# Transcribe audio
response = stt.transcribe("meeting.mp3")
print(response.text)
```

**Example - With Language and Context:**

```python
# Improve accuracy with language and context
response = stt.transcribe(
    "podcast.wav",
    language="en",
    prompt="This is a technical discussion about AI and machine learning"
)
print(f"Transcription: {response.text}")
print(f"Language: {response.language}")
```

**Example - Async Transcription:**

```python
async def transcribe_batch():
    stt = AIFactory.create_speech_to_text(
        "openai-compatible",
        "faster-whisper-large-v3",
        config={"base_url": "http://localhost:8000"}
    )

    files = ["audio1.mp3", "audio2.wav", "audio3.m4a"]
    for audio_file in files:
        response = await stt.atranscribe(audio_file)
        print(f"{audio_file}: {response.text[:100]}...")

# Run async
# await transcribe_batch()
```

### Text-to-Speech

**Supported Implementations:**
- [Speaches](https://github.com/speaches-ai/speaches/) - Piper, Kokoro models
- [Shabdabhav](https://github.com/Hardik94/shabdabhav) - Piper and Parler models
- Custom Piper-TTS deployments
- Custom TTS endpoints

**Configuration:**

```python
from esperanto.factory import AIFactory

speaker = AIFactory.create_text_to_speech(
    "openai-compatible",
    "piper-tts",
    config={
        "base_url": "http://localhost:8000",  # Required
        "api_key": "your-api-key"             # Optional
    }
)
```

**Example - Basic TTS:**

```python
from esperanto.factory import AIFactory

# Create TTS model
tts = AIFactory.create_text_to_speech(
    "openai-compatible",
    "speaches-ai/Kokoro-82M-v1.0-ONNX",
    config={"base_url": "http://localhost:8000"}
)

# Generate speech
response = tts.generate_speech(
    text="Hello from OpenAI-compatible TTS!",
    voice="af_heart",  # Use voice supported by your endpoint
    output_file="output.mp3"
)

print(f"Generated {len(response.audio_data)} bytes of audio")
```

**Example - Async TTS:**

```python
async def generate_speech_async():
    tts = AIFactory.create_text_to_speech(
        "openai-compatible",
        "piper-tts",
        config={"base_url": "http://localhost:8000"}
    )

    response = await tts.agenerate_speech(
        text="Async speech generation example",
        voice="en_US-amy-medium"
    )
    return response.audio_data

# Run async
# audio = await generate_speech_async()
```

**Example - Custom Parameters:**

```python
# Pass additional parameters supported by your endpoint
response = tts.generate_speech(
    text="Custom speech generation",
    voice="en_US-amy-medium",
    speed=1.2,              # Custom parameter
    response_format="wav",  # Audio format: mp3, opus, aac, flac, wav, pcm
    quality="high"          # Custom parameter
)
```

## Advanced Features

### Multi-Endpoint Configuration

Use different endpoints for different capabilities:

```bash
# Set up environment for multiple endpoints
export OPENAI_COMPATIBLE_BASE_URL_LLM="http://localhost:1234/v1"
export OPENAI_COMPATIBLE_BASE_URL_EMBEDDING="http://localhost:8080/v1"
export OPENAI_COMPATIBLE_BASE_URL_STT="http://192.168.1.100:9000"
export OPENAI_COMPATIBLE_BASE_URL_TTS="http://192.168.1.100:7000"
```

```python
# Each capability automatically uses its specific endpoint
llm = AIFactory.create_language("openai-compatible", "llama-3")
embedder = AIFactory.create_embedding("openai-compatible", "nomic-embed")
stt = AIFactory.create_speech_to_text("openai-compatible", "whisper")
tts = AIFactory.create_text_to_speech("openai-compatible", "piper")
```

### Model Discovery

Some endpoints support listing available models:

```python
# If endpoint supports /v1/models
model = AIFactory.create_language(
    "openai-compatible",
    "any-model",
    config={"base_url": "http://localhost:1234/v1"}
)

# The provider will query available models automatically
```

### Graceful Degradation

The provider handles varying feature support:

```python
try:
    # Attempt JSON mode
    model = AIFactory.create_language(
        "openai-compatible",
        "model-name",
        config={
            "base_url": "http://localhost:1234/v1",
            "structured": {"type": "json"}
        }
    )
    response = model.chat_complete(messages)
except Exception as e:
    # Falls back gracefully if not supported
    print(f"JSON mode not supported: {e}")
```

### LangChain Integration

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "openai-compatible",
    "your-model",
    config={"base_url": "http://localhost:1234/v1"}
)

langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Troubleshooting

### Common Errors

**Connection Error:**
```
Error: Failed to connect to endpoint
```
**Solution:**
1. Ensure your endpoint is running and accessible
2. Verify the base URL is correct (include `/v1` if required)
3. Check firewall/network settings

**Authentication Error:**
```
Error: Unauthorized
```
**Solution:**
1. Check if your endpoint requires an API key
2. Verify the API key is correct
3. Some local endpoints don't require authentication - try removing the api_key

**Model Not Found:**
```
Error: Model 'xxx' not found
```
**Solution:**
1. Verify your model name matches what's loaded in your endpoint
2. Check the endpoint's model list (if available)
3. For LM Studio: ensure the model is loaded in the GUI

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase timeout for large requests:
```python
config={
    "base_url": "http://localhost:1234/v1",
    "timeout": 600  # 10 minutes
}
```

### LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a model from the "Discover" tab
3. Load the model in the "Chat" tab
4. Click the "Local Server" tab and start the server
5. Use base URL: `http://localhost:1234/v1`

**LM Studio Example:**
```python
model = AIFactory.create_language(
    "openai-compatible",
    "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    config={
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-required"
    }
)
```

### Endpoint Compatibility

**Required Endpoints:**
- **LLM**: `POST /v1/chat/completions`
- **Embedding**: `POST /v1/embeddings`
- **STT**: `POST /audio/transcriptions`
- **TTS**: `POST /audio/speech`

**Optional Endpoints:**
- `GET /v1/models` - List available models
- `GET /audio/voices` - List available voices (TTS)

## Use Cases

### When to Choose OpenAI-Compatible

**Perfect for:**
- Privacy-sensitive applications (local processing)
- Local/on-premise deployments
- Custom or specialized models
- Cost optimization with high volume
- Development and testing environments
- Air-gapped/offline environments
- Integration with existing local infrastructure

**Consider alternatives if:**
- Want zero-setup cloud solution
- Need guaranteed enterprise SLA
- Prefer not to manage infrastructure
- Want cutting-edge cloud features immediately

### Deployment Scenarios

**1. Local Development:**
```python
# LM Studio for easy local testing
model = AIFactory.create_language(
    "openai-compatible",
    "local-model",
    config={"base_url": "http://localhost:1234/v1"}
)
```

**2. Production Self-Hosted:**
```python
# vLLM for high-performance production
model = AIFactory.create_language(
    "openai-compatible",
    "production-model",
    config={
        "base_url": "https://llm.yourcompany.com/v1",
        "api_key": "your-service-token"
    }
)
```

**3. Edge Computing:**
```python
# Local inference on edge devices
model = AIFactory.create_language(
    "openai-compatible",
    "edge-model",
    config={"base_url": "http://edge-device.local:8000/v1"}
)
```

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [Embeddings Guide](../capabilities/embedding.md)
- [Speech-to-Text Guide](../capabilities/speech-to-text.md)
- [Text-to-Speech Guide](../capabilities/text-to-speech.md)
- [Ollama Provider](./ollama.md)
- [OpenAI Provider](./openai.md)
