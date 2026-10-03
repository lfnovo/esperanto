# Groq

## Overview

Groq provides ultra-fast inference for open-source language models and Whisper speech recognition through their custom LPU (Language Processing Unit) hardware.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | GPT-OSS and Qwen models |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ✅ | Whisper models with faster inference |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://console.groq.com/docs

## Prerequisites

### Account Requirements
- Groq account (sign up at https://console.groq.com)
- API key with credits

### Getting API Keys
1. Visit https://console.groq.com/keys
2. Click "Create API Key"
3. Copy and store the key securely

## Environment Variables

```bash
# Groq API key (required)
GROQ_API_KEY="gsk_..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`GROQ_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Language model
model = AIFactory.create_language("groq", "openai/gpt-oss-120b")

# Speech-to-text
transcriber = AIFactory.create_speech_to_text("groq", "whisper-large-v3")
```

### Direct Instantiation

```python
from esperanto.providers.llm.groq import GroqLanguageModel
from esperanto.providers.speech_to_text.groq import GroqSpeechToText

# Language model
llm = GroqLanguageModel(
    api_key="your-api-key",
    model_name="openai/gpt-oss-120b"
)

# Speech-to-text
stt = GroqSpeechToText(
    api_key="your-api-key",
    model_name="whisper-large-v3"
)
```

## Capabilities

### Language Models (LLM)

**Available Models:**

| Model | Context Window | Best For |
|-------|----------------|----------|
| **openai/gpt-oss-120b** | 128K tokens | Default. Strong reasoning, tool use and structured output |
| **openai/gpt-oss-20b** | 128K tokens | Faster and cheaper, good for simple tasks |
| **qwen/qwen3.8-27b** | 128K tokens | Multilingual, general-purpose |

Groq's catalog changes often; list the current models with
`AIFactory.get_provider_models("groq")` or see https://console.groq.com/docs/models.

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "groq",
    "openai/gpt-oss-120b",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 2.0)
        "max_tokens": 1000,           # Maximum response length
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode
        "timeout": 60.0               # Request timeout
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create model
model = AIFactory.create_language("groq", "openai/gpt-oss-120b")

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain machine learning briefly."}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Fast Inference:**

```python
# Use the smaller GPT-OSS model for fast responses
model = AIFactory.create_language("groq", "openai/gpt-oss-20b")

messages = [{"role": "user", "content": "What is Python?"}]
response = model.chat_complete(messages)
# Extremely fast response time thanks to Groq's LPU
```

**Example - Long Context:**

```python
# 128K context window
model = AIFactory.create_language("groq", "openai/gpt-oss-120b")

# Handle long documents
long_doc = "..." * 10000  # Large document

messages = [{
    "role": "user",
    "content": f"Summarize this document:\n\n{long_doc}"
}]

response = model.chat_complete(messages)
```

**Example - Streaming:**

```python
# Synchronous streaming - extremely fast token generation
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
import asyncio

async def stream_async():
    async for chunk in await model.achat_complete(messages, stream=True):
        print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(stream_async())
```

**Example - JSON Mode:**

```python
model = AIFactory.create_language(
    "groq",
    "openai/gpt-oss-120b",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three countries with their capitals as JSON"
}]

response = model.chat_complete(messages)
# Response will be valid JSON
```

**Example - Async Chat:**

```python
async def chat_async():
    model = AIFactory.create_language("groq", "openai/gpt-oss-120b")

    messages = [{"role": "user", "content": "Explain quantum computing"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)
```

### Speech-to-Text

**Available Models:**

| Model | Best For |
|-------|----------|
| **whisper-large-v3** | Highest accuracy, multiple languages |
| **whisper-large-v3-turbo** | Faster inference, good accuracy |

**Configuration:**

```python
from esperanto.factory import AIFactory

transcriber = AIFactory.create_speech_to_text(
    "groq",
    "whisper-large-v3",
    config={
        "timeout": 300.0  # 5 minutes for large files
    }
)
```

**Example - Basic Transcription:**

```python
from esperanto.factory import AIFactory

# Create speech-to-text model
model = AIFactory.create_speech_to_text("groq", "whisper-large-v3")

# Transcribe audio - ultra-fast with Groq's LPU
response = model.transcribe("audio.mp3")
print(response.text)

# Transcribe from file object
with open("audio.mp3", "rb") as f:
    response = model.transcribe(f)
    print(response.text)
```

**Example - Fast Transcription:**

```python
# Use the turbo model for faster processing
model = AIFactory.create_speech_to_text("groq", "whisper-large-v3-turbo")

response = model.transcribe("english_audio.mp3")
print(response.text)
```

**Example - With Language and Context:**

```python
# Improve accuracy with language and prompt
response = model.transcribe(
    "podcast.mp3",
    language="en",
    prompt="This is a technical podcast about machine learning and AI"
)
print(f"Transcription: {response.text}")
print(f"Language: {response.language}")
```

**Example - Async Transcription:**

```python
async def transcribe_async():
    model = AIFactory.create_speech_to_text("groq", "whisper-large-v3")

    response = await model.atranscribe("meeting.wav")
    print(f"Transcription: {response.text}")
    print(f"Language: {response.language}")
```

**Example - Segments and Duration:**

Groq inherits OpenAI's `verbose_json` behavior, so segments and duration come
back automatically:

```python
response = model.transcribe("audio.mp3")

print(f"Duration: {response.duration:.2f}s")

if response.segments:
    for segment in response.segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
        # Per-segment Whisper extras (avg_logprob, compression_ratio, etc.)
        # live in segment.metadata.
```

**Example - Batch Processing:**

```python
import os
from esperanto.factory import AIFactory

model = AIFactory.create_speech_to_text("groq", "whisper-large-v3-turbo")

# Process multiple audio files quickly
audio_files = ["file1.mp3", "file2.wav", "file3.m4a"]
transcriptions = []

for file_path in audio_files:
    if os.path.exists(file_path):
        response = model.transcribe(file_path)
        transcriptions.append({
            "file": file_path,
            "text": response.text,
            "language": response.language
        })
        print(f"Transcribed {file_path}: {len(response.text)} characters")

# Save all transcriptions
for transcript in transcriptions:
    output_file = transcript["file"].replace(".mp3", ".txt").replace(".wav", ".txt")
    with open(output_file, "w") as f:
        f.write(transcript["text"])
```

**Example - Real-time Processing:**

```python
async def process_audio_stream():
    model = AIFactory.create_speech_to_text("groq", "whisper-large-v3-turbo")

    # Process audio files as they become available
    audio_queue = ["chunk1.wav", "chunk2.wav", "chunk3.wav"]

    for audio_chunk in audio_queue:
        response = await model.atranscribe(audio_chunk)
        print(f"Chunk transcription: {response.text}")

        # Process immediately
        if "urgent" in response.text.lower():
            print("🚨 Urgent content detected!")
```

## Advanced Features

### Ultra-Fast Inference
Groq's LPU (Language Processing Unit) provides exceptional inference speed:

```python
import time

model = AIFactory.create_language("groq", "openai/gpt-oss-20b")

messages = [{"role": "user", "content": "What is the speed of light?"}]

start = time.time()
response = model.chat_complete(messages)
end = time.time()

print(f"Response in {end - start:.2f} seconds")
# Typically sub-second for short responses
```

### Streaming Performance
Groq excels at streaming with high token generation speeds:

```python
import time

model = AIFactory.create_language("groq", "openai/gpt-oss-120b")

messages = [{"role": "user", "content": "Write a short story about AI."}]

start = time.time()
token_count = 0

for chunk in model.chat_complete(messages, stream=True):
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
        token_count += len(content.split())

end = time.time()
print(f"\n\nGenerated {token_count} tokens in {end - start:.2f}s")
print(f"Speed: {token_count / (end - start):.0f} tokens/second")
```

### Timeout Configuration
Customize request timeouts:

```python
# LLM with custom timeout
model = AIFactory.create_language(
    "groq",
    "openai/gpt-oss-120b",
    config={"timeout": 120.0}  # 2 minutes
)

# STT with longer timeout for large files
transcriber = AIFactory.create_speech_to_text(
    "groq",
    "whisper-large-v3",
    config={"timeout": 600.0}  # 10 minutes
)
```

### LangChain Integration
Convert to LangChain models:

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("groq", "openai/gpt-oss-120b")
langchain_model = model.to_langchain()

# Use with LangChain
response = langchain_model.invoke("Hello!")
print(response.content)
```

## Model Selection Guide

### GPT-OSS 120B (Default)
**Best for:** Reasoning, tool use and structured output
- OpenAI's open-weight model, served on Groq's LPU
- 128K context window
- Supports `json_schema` structured output

```python
model = AIFactory.create_language("groq", "openai/gpt-oss-120b")
```

### GPT-OSS 20B
**Best for:** Speed and cost-efficiency
- Faster and cheaper than the 120B model
- 128K context window
- Good for simple tasks

```python
model = AIFactory.create_language("groq", "openai/gpt-oss-20b")
```

### Qwen 3.8 27B
**Best for:** Multilingual tasks
- 128K context window

```python
model = AIFactory.create_language("groq", "qwen/qwen3.8-27b")
```

### Whisper Large V3
**Best for:** High-accuracy transcription
- Best accuracy
- Multiple languages
- Slower than turbo

```python
transcriber = AIFactory.create_speech_to_text("groq", "whisper-large-v3")
```

### Whisper Large V3 Turbo
**Best for:** Fast transcription
- Faster than standard
- Good accuracy
- Multiple languages

```python
transcriber = AIFactory.create_speech_to_text("groq", "whisper-large-v3-turbo")
```

## Performance Characteristics

### LLM Inference Speed
Groq's LPU serves models at hundreds of tokens per second; smaller models such
as `openai/gpt-oss-20b` are the fastest. Actual speed varies by model and load.

### Speech-to-Text Speed
- **Whisper Large V3 Turbo**: faster than Large V3, with slightly lower accuracy

### Context Windows
Current Groq chat models offer 128K tokens.
`AIFactory.get_provider_models("groq")` reports each model's context window.

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key is correct in the Groq console.

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Groq has generous rate limits. If exceeded, wait or contact support for increases.

**Model Not Available:**
```
Error: Model not found
```
**Solution:** Ensure you're using a valid model name. Check the Groq documentation for available models.

**Audio Format Issues:**
```
Error: Unsupported audio format
```
**Solution:** Groq supports MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM formats. Maximum file size: 25 MB.

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase the timeout configuration, especially for long audio files.

### Best Practices

1. **Leverage Speed:** Take advantage of Groq's ultra-fast inference for real-time applications.

2. **Choose Right Model:** Use `openai/gpt-oss-20b` for speed and `openai/gpt-oss-120b` for quality.

3. **Streaming:** Always use streaming for better user experience with Groq's high token generation speed.

4. **Context Windows:** Utilize large context windows (128K) for long documents.

5. **Fast Audio:** Use `whisper-large-v3-turbo` when speed matters more than maximum accuracy.

6. **Batch Processing:** Process multiple requests efficiently thanks to fast inference.

## Use Cases

### Real-time Chat Applications
```python
# Fast responses for interactive chat
model = AIFactory.create_language("groq", "openai/gpt-oss-20b")

# Sub-second response times
response = model.chat_complete(messages)
```

### Live Transcription
```python
# Fast transcription for live events
transcriber = AIFactory.create_speech_to_text("groq", "whisper-large-v3-turbo")

# Process audio chunks quickly
response = transcriber.transcribe("live_chunk.wav")
```

### High-Volume Processing
```python
# Process many requests quickly
model = AIFactory.create_language("groq", "openai/gpt-oss-120b")

# Fast inference allows high throughput
for item in large_dataset:
    response = model.chat_complete([{"role": "user", "content": item}])
```

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [Speech-to-Text Guide](../capabilities/speech-to-text.md)
- [OpenAI Provider](./openai.md)
- [Anthropic Provider](./anthropic.md)
- [Google Provider](./google.md)
