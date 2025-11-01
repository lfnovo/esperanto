# Azure OpenAI

## Overview

Azure OpenAI Service provides access to OpenAI models through Microsoft Azure's enterprise-grade infrastructure with additional security, compliance, and regional deployment options.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | GPT-4, GPT-3.5 via Azure deployments |
| Embeddings | ✅ | text-embedding-3-small/large, ada-002 |
| Reranking | ❌ | Not available |
| Speech-to-Text | ✅ | Whisper via Azure deployments |
| Text-to-Speech | ✅ | TTS-1, TTS-1-HD via Azure deployments |

**Official Documentation:** https://learn.microsoft.com/azure/ai-services/openai/

## Prerequisites

### Account Requirements
- Azure subscription
- Azure OpenAI resource created in Azure Portal
- Deployments created for models you want to use

### Setting Up Azure OpenAI
1. Create Azure OpenAI resource in Azure Portal
2. Create deployments for your desired models (e.g., "gpt-4-deployment")
3. Note your endpoint URL (e.g., `https://your-resource.openai.azure.com/`)
4. Get your API key from the resource's "Keys and Endpoint" section

### Important: Deployment Names
In Azure OpenAI, you use **deployment names**, not model names. The `model_name` parameter in Esperanto corresponds to your Azure deployment name.

Example: If you created a deployment called "my-gpt4-deployment" for GPT-4, use:
```python
model = AIFactory.create_language("azure", "my-gpt4-deployment")
```

## Environment Variables

```bash
# Generic (works for all modalities)
AZURE_OPENAI_API_KEY="your-azure-api-key"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-02-01"

# Modality-specific (takes precedence over generic)
# Use these when you want different deployments for different AI capabilities

# Language Models
# AZURE_OPENAI_API_KEY_LLM="your-llm-key"
# AZURE_OPENAI_ENDPOINT_LLM="https://your-llm-resource.openai.azure.com/"
# AZURE_OPENAI_API_VERSION_LLM="2024-02-01"

# Embeddings
# AZURE_OPENAI_API_KEY_EMBEDDING="your-embedding-key"
# AZURE_OPENAI_ENDPOINT_EMBEDDING="https://your-embedding-resource.openai.azure.com/"
# AZURE_OPENAI_API_VERSION_EMBEDDING="2024-12-01-preview"

# Speech-to-Text
# AZURE_OPENAI_API_KEY_STT="your-stt-key"
# AZURE_OPENAI_ENDPOINT_STT="https://your-stt-resource.openai.azure.com/"
# AZURE_OPENAI_API_VERSION_STT="2024-02-01"

# Text-to-Speech
# AZURE_OPENAI_API_KEY_TTS="your-tts-key"
# AZURE_OPENAI_ENDPOINT_TTS="https://your-tts-resource.openai.azure.com/"
# AZURE_OPENAI_API_VERSION_TTS="2024-02-01"
```

**Variable Priority:**
1. Direct parameters in code (`api_key=`, `azure_endpoint=`, `api_version=`)
2. Modality-specific env vars (`AZURE_OPENAI_*_LLM`, `AZURE_OPENAI_*_EMBEDDING`, etc.)
3. Generic env vars (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, etc.)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Language model (using deployment name)
model = AIFactory.create_language("azure", "your-gpt4-deployment")

# Embedding model (using deployment name)
embedder = AIFactory.create_embedding("azure", "your-embedding-deployment")

# Speech-to-text (using deployment name)
transcriber = AIFactory.create_speech_to_text("azure", "your-whisper-deployment")

# Text-to-speech (using deployment name)
speaker = AIFactory.create_text_to_speech("azure", "your-tts-deployment")
```

### Direct Instantiation

```python
from esperanto.providers.llm.azure import AzureLanguageModel
from esperanto.providers.embedding.azure import AzureEmbeddingModel
from esperanto.providers.speech_to_text.azure import AzureSpeechToText
from esperanto.providers.text_to_speech.azure import AzureTextToSpeech

# Language model
llm = AzureLanguageModel(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01",
    model_name="your-deployment-name"
)

# Embedding model
embedder = AzureEmbeddingModel(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-12-01-preview",
    model_name="your-embedding-deployment"
)

# Speech-to-text
stt = AzureSpeechToText(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01",
    model_name="your-whisper-deployment"
)

# Text-to-speech
tts = AzureTextToSpeech(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01",
    model_name="your-tts-deployment"
)
```

## Capabilities

### Language Models (LLM)

**Available Models:**
Same models as OpenAI, deployed through Azure:
- **GPT-4 series** - gpt-4, gpt-4-turbo, gpt-4o
- **GPT-3.5 series** - gpt-3.5-turbo

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "azure",
    "your-deployment-name",  # Your Azure deployment name
    config={
        "temperature": 0.7,           # Randomness (0.0 - 2.0)
        "max_tokens": 1000,           # Maximum response length
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode
        "api_key": "your-key",        # Optional if using env vars
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-01"
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create model using environment variables
model = AIFactory.create_language("azure", "my-gpt4-deployment")

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Explicit Configuration:**

```python
# Without environment variables
model = AIFactory.create_language(
    "azure",
    "my-gpt4-deployment",
    config={
        "api_key": "your-azure-key",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-01",
        "temperature": 0.7,
        "structured": {"type": "json"}  # Azure supports JSON mode
    }
)

messages = [{"role": "user", "content": "Translate 'hello' to Spanish."}]
response = model.chat_complete(messages)
```

**Example - Streaming:**

```python
# Synchronous streaming
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Embeddings

**Available Models:**
Same as OpenAI, deployed through Azure:
- **text-embedding-3-small** (1536 dimensions)
- **text-embedding-3-large** (3072 dimensions)
- **text-embedding-ada-002** (1536 dimensions)

**Configuration:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

model = AIFactory.create_embedding(
    "azure",
    "your-embedding-deployment",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,  # Task optimization
        "output_dimensions": 1024,  # Reduce dimensions (3-large only)
        "api_key": "your-key",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "api_version": "2024-12-01-preview"
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
embedder = AIFactory.create_embedding("azure", "my-embedding-deployment")

# Generate embeddings
texts = ["Hello, world!", "Another text"]
response = embedder.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
```

**Example - Task-Optimized:**

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Query embeddings
query_model = AIFactory.create_embedding(
    "azure",
    "my-embedding-deployment",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

# Document embeddings
document_model = AIFactory.create_embedding(
    "azure",
    "my-embedding-deployment",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

query_emb = query_model.embed(["search query"])
doc_emb = document_model.embed(["document 1", "document 2"])
```

### Speech-to-Text

**Available Models:**
- **whisper** - Whisper model via Azure deployment

**Configuration:**

```python
from esperanto.factory import AIFactory

transcriber = AIFactory.create_speech_to_text(
    "azure",
    "your-whisper-deployment",
    config={
        "timeout": 300.0,  # 5 minutes for large files
        "api_key": "your-key",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-01"
    }
)
```

**Example - Basic Transcription:**

```python
from esperanto.factory import AIFactory

# Create speech-to-text model
model = AIFactory.create_speech_to_text("azure", "my-whisper-deployment")

# Transcribe audio
response = model.transcribe("meeting.mp3")
print(response.text)
```

**Example - With Language and Context:**

```python
# Improve accuracy
response = model.transcribe(
    "podcast.wav",
    language="en",
    prompt="This is a technical discussion about AI and machine learning"
)
print(f"Transcription: {response.text}")
```

**Example - Async Transcription:**

```python
async def transcribe_batch():
    model = AIFactory.create_speech_to_text("azure", "my-whisper-deployment")

    files = ["audio1.mp3", "audio2.wav", "audio3.m4a"]
    for audio_file in files:
        response = await model.atranscribe(audio_file)
        print(f"{audio_file}: {response.text[:100]}...")
```

### Text-to-Speech

**Available Models:**
- **tts-1** - Standard quality
- **tts-1-hd** - High definition quality

**Available Voices:**
Same as OpenAI:
- **alloy** - Balanced, neutral voice
- **echo** - Male voice with clarity
- **fable** - British accent, storytelling tone
- **onyx** - Deep, authoritative male voice
- **nova** - Young, energetic female voice
- **shimmer** - Warm, expressive female voice

**Configuration:**

```python
from esperanto.factory import AIFactory

speaker = AIFactory.create_text_to_speech(
    "azure",
    "your-tts-deployment",
    config={
        "timeout": 300.0,
        "api_key": "your-key",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-01"
    }
)
```

**Example - Basic Speech Generation:**

```python
from esperanto.factory import AIFactory

# Create text-to-speech model
model = AIFactory.create_text_to_speech("azure", "my-tts-deployment")

# Generate speech
response = model.generate_speech(
    text="Hello from Azure Text-to-Speech!",
    voice="alloy",
    output_file="greeting.mp3"
)

print(f"Generated {len(response.audio_data)} bytes of audio")
```

**Example - Voice Customization:**

```python
# HD quality with speed control
response = model.generate_speech(
    text="This speech has custom speed settings",
    voice="nova",
    speed=1.2,  # Faster speech
    output_file="fast_speech.mp3"
)
```

**Example - Async Generation:**

```python
async def generate_audio_async():
    model = AIFactory.create_text_to_speech("azure", "my-tts-deployment")

    response = await model.agenerate_speech(
        text="Async speech generation",
        voice="shimmer",
        output_file="async_output.mp3"
    )
    return response
```

## Advanced Features

### Modality-Specific Configuration
Use different Azure resources for different capabilities:

```bash
# Set environment variables for different resources
export AZURE_OPENAI_API_KEY_LLM="key-for-llm"
export AZURE_OPENAI_ENDPOINT_LLM="https://llm-resource.openai.azure.com/"

export AZURE_OPENAI_API_KEY_EMBEDDING="key-for-embeddings"
export AZURE_OPENAI_ENDPOINT_EMBEDDING="https://embedding-resource.openai.azure.com/"

export AZURE_OPENAI_API_KEY_STT="key-for-stt"
export AZURE_OPENAI_ENDPOINT_STT="https://stt-resource.openai.azure.com/"

export AZURE_OPENAI_API_KEY_TTS="key-for-tts"
export AZURE_OPENAI_ENDPOINT_TTS="https://tts-resource.openai.azure.com/"
```

```python
# Each capability uses its specific configuration
llm = AIFactory.create_language("azure", "gpt4-deployment")  # Uses LLM vars
embedder = AIFactory.create_embedding("azure", "embed-deployment")  # Uses EMBEDDING vars
stt = AIFactory.create_speech_to_text("azure", "whisper-deployment")  # Uses STT vars
tts = AIFactory.create_text_to_speech("azure", "tts-deployment")  # Uses TTS vars
```

### Enterprise Features
- **Private Networks**: VNet integration for secure communication
- **Compliance**: SOC 2, HIPAA, and other certifications
- **Regional Control**: Deploy in specific Azure regions for data residency
- **Integrated Billing**: Usage appears on your Azure subscription
- **Enterprise Support**: SLA guarantees and Microsoft support

### Timeout Configuration
Customize request timeouts:

```python
# LLM with custom timeout
model = AIFactory.create_language(
    "azure",
    "my-deployment",
    config={"timeout": 120.0}  # 2 minutes
)

# STT with longer timeout for large files
transcriber = AIFactory.create_speech_to_text(
    "azure",
    "whisper-deployment",
    config={"timeout": 600.0}  # 10 minutes
)
```

### LangChain Integration
Convert to LangChain models:

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("azure", "my-gpt4-deployment")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key is correct in the Azure Portal under "Keys and Endpoint".

**Deployment Not Found:**
```
Error: The API deployment for this resource does not exist
```
**Solution:** Ensure you're using the correct deployment name (not model name). Check your deployments in Azure Portal.

**Incorrect Endpoint:**
```
Error: Failed to connect
```
**Solution:** Verify your endpoint URL is correct. It should be in the format:
`https://your-resource-name.openai.azure.com/`

**API Version Error:**
```
Error: Invalid API version
```
**Solution:** Use a valid API version. Common versions:
- `2024-02-01` - General use
- `2024-12-01-preview` - Embeddings with custom dimensions

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Check your Azure OpenAI quota and request an increase if needed.

### Best Practices

1. **Use Deployment Names:** Always use your Azure deployment names, not OpenAI model names.

2. **Environment Variables:** Use modality-specific env vars when using multiple Azure resources.

3. **API Versions:** Keep API versions updated. Check Azure documentation for the latest.

4. **Private Endpoints:** Use VNet integration for production workloads requiring private connectivity.

5. **Regional Deployment:** Deploy in regions close to your users for better latency.

6. **Monitoring:** Use Azure Monitor to track usage and performance.

7. **Cost Management:** Set up Azure cost alerts to monitor OpenAI usage.

## Azure vs OpenAI

**Choose Azure OpenAI if you:**
- Need enterprise compliance (HIPAA, SOC 2, etc.)
- Require regional data residency
- Want integrated Azure billing
- Need private network connectivity
- Already use Azure infrastructure
- Require Microsoft support SLAs

**Choose OpenAI if you:**
- Want latest features immediately
- Prefer simpler setup
- Don't need enterprise features
- Want direct OpenAI billing

## See Also

- [Language Models Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/llm.md)
- [Embeddings Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/embedding.md)
- [Speech-to-Text Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/speech_to_text.md)
- [Text-to-Speech Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/text_to_speech.md)
- [OpenAI Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/openai.md)
- [Google Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/google.md)
