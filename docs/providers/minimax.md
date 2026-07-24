# MiniMax

## Overview

MiniMax provides `MiniMax-M3` with a 1M-token context window through an
OpenAI-compatible API, plus a native Text-to-Speech API.

> [!IMPORTANT]
> MiniMax API keys are region-specific. Mainland China keys must use
> `https://api.minimaxi.com/v1`; international keys use
> `https://api.minimax.io/v1`. Using the wrong endpoint returns an
> `invalid api key` error even when the key is valid.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | MiniMax-M3 and M2 series |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ✅ | speech-2.8, speech-2.6, and speech-02 series |

**Official Documentation:** https://platform.minimaxi.com/

## Prerequisites

### Account Requirements
- MiniMax account at https://platform.minimaxi.com/
- API key with credits

### Getting API Keys
1. Visit https://platform.minimaxi.com/
2. Navigate to API Keys
3. Create and copy your API key

## Environment Variables

```bash
MINIMAX_API_KEY="your-api-key"

# International endpoint (default)
MINIMAX_BASE_URL="https://api.minimax.io/v1"

# Mainland China endpoint
MINIMAX_BASE_URL="https://api.minimaxi.com/v1"
```

`MINIMAX_BASE_URL` is used for both LLM and TTS requests. The TTS provider
normalizes the optional `/v1` suffix when building its native endpoints.

## Quick Start

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("minimax", "MiniMax-M3")

messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

MiniMax language models use Esperanto's standard OpenAI-compatible path.

## Available Models

| Model | Context Window | Best For |
|-------|---------------|----------|
| `MiniMax-M3` | 1,000,000 | Latest coding and agent workloads |
| `MiniMax-M2.7` | 204,800 | General-purpose workloads |
| `MiniMax-M2.7-highspeed` | 204,800 | Latency-sensitive workloads |
| `MiniMax-M2.5` | 204,800 | Legacy workloads |
| `MiniMax-M2.5-highspeed` | 204,800 | Legacy latency-sensitive workloads |

The account model catalog can also be queried dynamically:

```python
models = AIFactory.get_provider_models("minimax", model_type="language")
```

## Features

### Streaming

```python
model = AIFactory.create_language("minimax", "MiniMax-M3")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "minimax", "MiniMax-M3",
    config={"structured": {"type": "json_object"}}
)
```

### Async Support

```python
response = await model.achat_complete(messages)
```

## Text-to-Speech

```python
speaker = AIFactory.create_text_to_speech(
    "minimax",
    "speech-2.8-hd",
)

audio = speaker.generate_speech(
    text="Hello from MiniMax.",
    voice="English_Graceful_Lady",
    language_boost="English",
)

with open("minimax.mp3", "wb") as file:
    file.write(audio.audio_data)
```

### TTS Models

| Model |
|-------|
| `speech-2.8-hd` |
| `speech-2.8-turbo` |
| `speech-2.6-hd` |
| `speech-2.6-turbo` |
| `speech-02-hd` |
| `speech-02-turbo` |

### Voice Discovery

```python
for voice_id, voice in speaker.available_voices.items():
    print(voice_id, voice.name)
```

### Voice and Audio Controls

```python
audio = speaker.generate_speech(
    text="Welcome to the show!",
    voice="English_Graceful_Lady",
    response_format="wav",
    speed=1.1,
    volume=1.5,
    pitch=1,
    emotion="happy",
    sample_rate=32000,
    bitrate=128000,
    channels=1,
)
```

Provider-native options such as `language_boost`, `pronunciation_dict`,
`voice_modify`, `voice_setting`, and `audio_setting` can be passed through as
keyword arguments.

### Async TTS

```python
audio = await speaker.agenerate_speech(
    text="Generated asynchronously.",
    voice="English_Graceful_Lady",
)
```

## Configuration

```python
model = AIFactory.create_language(
    "minimax", "MiniMax-M3",
    config={"api_key": "your-key"}
)
```
