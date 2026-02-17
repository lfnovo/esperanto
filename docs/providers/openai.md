# OpenAI

## Overview

OpenAI provides access to GPT models, Whisper speech recognition, text-to-speech, and embeddings through their comprehensive API platform.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | GPT-4o, o1, o3, GPT-3.5 |
| Embeddings | ✅ | text-embedding-3-small/large, ada-002 |
| Reranking | ❌ | Not available |
| Speech-to-Text | ✅ | Whisper models |
| Text-to-Speech | ✅ | TTS-1, TTS-1-HD |

**Official Documentation:** https://platform.openai.com/docs

## Prerequisites

### Account Requirements
- OpenAI account (sign up at https://platform.openai.com)
- API key with billing enabled

### Getting API Keys
1. Visit https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy and store the key securely (it won't be shown again)

## Environment Variables

```bash
# OpenAI API key (required)
OPENAI_API_KEY="sk-..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`OPENAI_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Language model
model = AIFactory.create_language("openai", "gpt-4")

# Embedding model
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Speech-to-text
transcriber = AIFactory.create_speech_to_text("openai", "whisper-1")

# Text-to-speech
speaker = AIFactory.create_text_to_speech("openai", "tts-1")
```

### Direct Instantiation

```python
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
from esperanto.providers.speech_to_text.openai import OpenAISpeechToText
from esperanto.providers.text_to_speech.openai import OpenAITextToSpeech

# Language model
llm = OpenAILanguageModel(
    api_key="your-api-key",
    model_name="gpt-4"
)

# Embedding model
embedder = OpenAIEmbeddingModel(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

# Speech-to-text
stt = OpenAISpeechToText(
    api_key="your-api-key",
    model_name="whisper-1"
)

# Text-to-speech
tts = OpenAITextToSpeech(
    api_key="your-api-key",
    model_name="tts-1"
)
```

## Capabilities

### Language Models (LLM)

**Available Models:**
- **gpt-4o** - Latest GPT-4 Optimized model with vision capabilities
- **gpt-4-turbo** - Fast GPT-4 with 128K context window
- **gpt-4** - Original GPT-4 model
- **gpt-3.5-turbo** - Fast and cost-effective model
- **o1-preview** - Advanced reasoning model (o1 series)
- **o1-mini** - Lighter o1 model
- **o3-mini** - Latest o3 series model

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 2.0)
        "max_tokens": 1000,           # Maximum response length
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode
        "base_url": None,             # Custom endpoint (optional)
        "organization": None          # Organization ID (optional)
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create model
model = AIFactory.create_language("openai", "gpt-4")

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
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

**Example - JSON Mode:**

```python
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three European capitals as JSON"
}]

response = model.chat_complete(messages)
# Response will be valid JSON
```

### Embeddings

**Available Models:**

| Model | Dimensions | Cost (per 1M tokens) | Best For |
|-------|------------|---------------------|----------|
| **text-embedding-3-small** | 1536 | $0.02 | General use, cost-effective |
| **text-embedding-3-large** | 3072 | $0.13 | Highest quality, complex tasks |
| **text-embedding-ada-002** | 1536 | $0.10 | Legacy, compatibility |

**Configuration:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

model = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-large",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,  # Task optimization via prefixes
        "output_dimensions": 1024,  # Reduce from default 3072
        "timeout": 60.0  # Request timeout in seconds
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Generate embeddings
texts = ["Hello, world!", "Another text"]
response = embedder.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
    print(f"First 5 values: {embedding_obj.embedding[:5]}")
```

**Example - Task-Optimized Embeddings:**

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Optimize for search queries
query_model = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-large",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

# Optimize for document storage
document_model = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-large",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

# Generate embeddings
query_embedding = query_model.embed(["search query"])
doc_embeddings = document_model.embed(["document 1", "document 2"])
```

**Example - Custom Dimensions:**

```python
# Reduce embedding size for performance
model = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-large",
    config={"output_dimensions": 512}  # Down from 3072
)

embeddings = model.embed(["text"])
print(f"Dimensions: {len(embeddings.data[0].embedding)}")  # 512
```

### Speech-to-Text

**Available Models:**
- **whisper-1** - Latest Whisper model for speech recognition

**Configuration:**

```python
from esperanto.factory import AIFactory

transcriber = AIFactory.create_speech_to_text(
    "openai",
    "whisper-1",
    config={
        "timeout": 300.0  # 5 minutes for large files
    }
)
```

**Example - Basic Transcription:**

```python
from esperanto.factory import AIFactory

# Create speech-to-text model
model = AIFactory.create_speech_to_text("openai", "whisper-1")

# Transcribe from file path
response = model.transcribe("audio.mp3")
print(response.text)

# Transcribe from file object
with open("audio.mp3", "rb") as f:
    response = model.transcribe(f)
    print(response.text)
```

**Example - With Language and Context:**

```python
# Improve accuracy with language and context
response = model.transcribe(
    "podcast.mp3",
    language="en",  # Specify language
    prompt="This is a technical podcast about machine learning and AI"
)
print(response.text)
print(f"Language detected: {response.language}")
```

**Example - Async Transcription:**

```python
async def transcribe_async():
    model = AIFactory.create_speech_to_text("openai", "whisper-1")

    response = await model.atranscribe("meeting.wav")
    print(f"Transcription: {response.text}")
    print(f"Language: {response.language}")
```

### Text-to-Speech

**Available Models:**
- **tts-1** - Standard quality, faster generation
- **tts-1-hd** - High quality, more natural sounding

**Available Voices:**
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
    "openai",
    "tts-1-hd",
    config={
        "timeout": 300.0  # 5 minutes timeout
    }
)
```

**Example - Basic Speech Generation:**

```python
from esperanto.factory import AIFactory

# Create text-to-speech model
model = AIFactory.create_text_to_speech("openai", "tts-1")

# Generate speech
response = model.generate_speech(
    text="Hello, world!",
    voice="alloy",
    output_file="greeting.mp3"
)

print(f"Generated {len(response.audio_data)} bytes of audio")
print(f"Content type: {response.content_type}")
```

**Example - Voice Customization:**

```python
# HD quality with speed control
model = AIFactory.create_text_to_speech("openai", "tts-1-hd")

response = model.generate_speech(
    text="This speech has custom speed settings",
    voice="nova",
    speed=1.2,  # Faster speech (0.25 to 4.0)
    output_file="fast_speech.mp3"
)
```

**Example - Multiple Voices:**

```python
# Generate greetings in different voices
voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

for voice in voices:
    response = model.generate_speech(
        text=f"Hello from {voice}!",
        voice=voice,
        output_file=f"greeting_{voice}.mp3"
    )
    print(f"Generated {voice} greeting")
```

**Example - Async Generation:**

```python
async def generate_audio_async():
    model = AIFactory.create_text_to_speech("openai", "tts-1")

    response = await model.agenerate_speech(
        text="This is an async speech generation example",
        voice="shimmer",
        output_file="async_output.mp3"
    )
    print(f"Audio saved to async_output.mp3")
```

## Advanced Features

### Custom Base URL
Use a custom endpoint for OpenAI-compatible services:

```python
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={
        "base_url": "https://custom-endpoint.com/v1",
        "api_key": "your-api-key"
    }
)
```

### Organization Support
For users with multiple organizations:

```python
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={
        "organization": "org-xxxxxxxxxxxxx"
    }
)
```

### Timeout Configuration
Customize request timeouts:

```python
# LLM with custom timeout
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"timeout": 120.0}  # 2 minutes
)

# Embedding with custom timeout
embedder = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-small",
    config={"timeout": 90.0}  # 1.5 minutes
)

# Speech-to-text with longer timeout
transcriber = AIFactory.create_speech_to_text(
    "openai",
    "whisper-1",
    config={"timeout": 600.0}  # 10 minutes for large files
)
```

### LangChain Integration
Convert to LangChain models:

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("openai", "gpt-4")
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
**Solution:** Verify your API key is correct and has billing enabled.

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic with exponential backoff or upgrade your plan.

**Context Length Exceeded:**
```
Error: This model's maximum context length is X tokens
```
**Solution:** Reduce the message history or use a model with a larger context window.

**Invalid Model Name:**
```
Error: The model 'xxx' does not exist
```
**Solution:** Check the model name spelling and ensure you have access to that model.

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase the timeout configuration or check your network connection.

### Audio Format Issues

**Supported Audio Formats (Speech-to-Text):**
- MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM
- Maximum file size: 25 MB

**TTS Output Format:**
- MP3 format by default
- Configurable via `response_format` kwarg: `"mp3"`, `"opus"`, `"aac"`, `"flac"`, `"wav"`, `"pcm"`
- Stereo output
- 24kHz sample rate (tts-1) or 44.1kHz (tts-1-hd)

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [Embeddings Guide](../capabilities/embedding.md)
- [Speech-to-Text Guide](../capabilities/speech-to-text.md)
- [Text-to-Speech Guide](../capabilities/text-to-speech.md)
- [Anthropic Provider](./anthropic.md)
- [Google Provider](./google.md)
