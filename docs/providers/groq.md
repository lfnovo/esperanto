# Groq

## Overview

Groq provides ultra-fast inference for open-source language models and Whisper speech recognition through their custom LPU (Language Processing Unit) hardware.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Mixtral, Llama, Gemma models |
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
model = AIFactory.create_language("groq", "mixtral-8x7b-32768")

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
    model_name="mixtral-8x7b-32768"
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
| **mixtral-8x7b-32768** | 32K tokens | Balanced performance, high quality |
| **llama-3.3-70b-versatile** | 128K tokens | Latest Llama, versatile tasks |
| **llama-3.1-70b-versatile** | 128K tokens | Previous Llama version |
| **llama-3.1-8b-instant** | 128K tokens | Fast responses, cost-effective |
| **gemma2-9b-it** | 8K tokens | Google's Gemma, instruction-tuned |
| **gemma-7b-it** | 8K tokens | Smaller Gemma variant |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "groq",
    "mixtral-8x7b-32768",
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
model = AIFactory.create_language("groq", "mixtral-8x7b-32768")

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain machine learning briefly."}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Fast Inference with Llama:**

```python
# Use Llama 3.1 8B for ultra-fast responses
model = AIFactory.create_language("groq", "llama-3.1-8b-instant")

messages = [{"role": "user", "content": "What is Python?"}]
response = model.chat_complete(messages)
# Extremely fast response time thanks to Groq's LPU
```

**Example - Large Context with Llama 70B:**

```python
# Llama 3.3 70B with 128K context window
model = AIFactory.create_language("groq", "llama-3.3-70b-versatile")

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
async for chunk in model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Example - JSON Mode:**

```python
model = AIFactory.create_language(
    "groq",
    "mixtral-8x7b-32768",
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
    model = AIFactory.create_language("groq", "mixtral-8x7b-32768")

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
| **distil-whisper-large-v3-en** | English-only, optimized for speed |

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

**Example - Fast English Transcription:**

```python
# Use distil-whisper for English-only, faster processing
model = AIFactory.create_speech_to_text("groq", "distil-whisper-large-v3-en")

response = model.transcribe("english_audio.mp3")
print(response.text)
# Extremely fast transcription for English content
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

model = AIFactory.create_language("groq", "llama-3.1-8b-instant")

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

model = AIFactory.create_language("groq", "mixtral-8x7b-32768")

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
    "mixtral-8x7b-32768",
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

model = AIFactory.create_language("groq", "mixtral-8x7b-32768")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Model Selection Guide

### Mixtral 8x7B
**Best for:** Balanced performance and quality
- Excellent reasoning capabilities
- Good for complex tasks
- 32K context window
- Fast inference on Groq LPU

```python
model = AIFactory.create_language("groq", "mixtral-8x7b-32768")
```

### Llama 3.3 70B Versatile
**Best for:** Latest capabilities, long context
- Latest Llama model
- 128K context window
- Versatile for various tasks
- Strong performance

```python
model = AIFactory.create_language("groq", "llama-3.3-70b-versatile")
```

### Llama 3.1 8B Instant
**Best for:** Speed and cost-efficiency
- Ultra-fast responses
- Cost-effective
- 128K context window
- Good for simple tasks

```python
model = AIFactory.create_language("groq", "llama-3.1-8b-instant")
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

### Distil-Whisper Large V3 EN
**Best for:** English-only, maximum speed
- English-only
- Fastest transcription
- Optimized model

```python
transcriber = AIFactory.create_speech_to_text("groq", "distil-whisper-large-v3-en")
```

## Performance Characteristics

### LLM Inference Speed
Groq's LPU provides exceptional speed:
- **Llama 3.1 8B**: 500+ tokens/second
- **Mixtral 8x7B**: 300+ tokens/second
- **Llama 3.3 70B**: 200+ tokens/second

### Speech-to-Text Speed
Faster than traditional Whisper implementations:
- **Whisper Large V3**: 5-10x faster than CPU
- **Whisper Turbo**: 2-3x faster than large v3
- **Distil-Whisper**: Fastest for English

### Context Windows
- **Mixtral**: 32K tokens
- **Llama 3.x**: 128K tokens
- **Gemma**: 8K tokens

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

2. **Choose Right Model:** Use 8B models for speed, 70B models for quality, Mixtral for balance.

3. **Streaming:** Always use streaming for better user experience with Groq's high token generation speed.

4. **Context Windows:** Utilize large context windows (128K) for long documents.

5. **English-Only Audio:** Use distil-whisper for English content for maximum speed.

6. **Batch Processing:** Process multiple requests efficiently thanks to fast inference.

## Use Cases

### Real-time Chat Applications
```python
# Ultra-fast responses for interactive chat
model = AIFactory.create_language("groq", "llama-3.1-8b-instant")

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
model = AIFactory.create_language("groq", "mixtral-8x7b-32768")

# Fast inference allows high throughput
for item in large_dataset:
    response = model.chat_complete([{"role": "user", "content": item}])
```

## See Also

- [Language Models Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/llm.md)
- [Speech-to-Text Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/speech_to_text.md)
- [OpenAI Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/openai.md)
- [Anthropic Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/anthropic.md)
- [Google Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/google.md)
