# Mistral Text-to-Speech (Voxtral)

## Overview

Mistral's Voxtral model provides high-quality text-to-speech generation with multiple voices.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | See [mistral.md](mistral.md) |
| Embeddings | ✅ | See [mistral.md](mistral.md) |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ✅ | Voxtral model |

**Official Documentation:** https://docs.mistral.ai/capabilities/audio/

## Prerequisites

### Account Requirements
- Mistral AI account
- API key with access to Voxtral

### Getting API Keys
1. Visit https://console.mistral.ai/api-keys
2. Create a new API key
3. Store the key securely

## Environment Variables

```bash
# Mistral API key (required) — same key used for LLM and embedding providers
MISTRAL_API_KEY="your-api-key"
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`MISTRAL_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

tts = AIFactory.create_text_to_speech(
    "mistral",
    "voxtral-mini-tts-2603",
    config={"api_key": "your-api-key"},
)

response = tts.generate_speech(
    text="Hello, world!",
    voice="gb_jane_neutral",
)
```

### Direct Instantiation

```python
from esperanto.providers.tts.mistral import MistralTextToSpeechModel

tts = MistralTextToSpeechModel(
    model_name="voxtral-mini-tts-2603",
    api_key="your-api-key",
)

response = tts.generate_speech(
    text="Hello, world!",
    voice="gb_jane_neutral",
)
```

## Available Models

| Model | Description |
|-------|-------------|
| `voxtral-mini-tts-2603` | Voxtral Mini — default TTS model |

## Voice Selection

Mistral provides preset voices accessible via the `available_voices` property:

```python
voices = tts.available_voices
for voice_id, voice in voices.items():
    print(f"{voice_id}: {voice.name} ({voice.gender})")
```

**Known preset voices:**

| Voice ID | Description |
|----------|-------------|
| `gb_jane_neutral` | Jane — neutral British English voice |
| `en_paul_neutral` | Paul — neutral English voice |

To discover all available voices dynamically:

```python
voices = tts.available_voices
```

## Response Formats

Mistral supports the following audio formats:

| Format | Content-Type | Notes |
|--------|-------------|-------|
| `mp3` | `audio/mp3` | Default format |
| `wav` | `audio/wav` | Uncompressed audio |
| `flac` | `audio/flac` | Lossless compression |
| `opus` | `audio/opus` | Efficient compression |
| `pcm` | `audio/pcm` | Raw PCM audio |

```python
response = tts.generate_speech(
    text="Hello",
    voice="gb_jane_neutral",
    response_format="wav",
)
```

## Saving to File

```python
response = tts.generate_speech(
    text="Hello, world!",
    voice="gb_jane_neutral",
    output_file="output.mp3",
)
```

## Async Usage

```python
import asyncio
from esperanto.factory import AIFactory

async def main():
    tts = AIFactory.create_text_to_speech(
        "mistral",
        "voxtral-mini-tts-2603",
        config={"api_key": "your-api-key"},
    )
    response = await tts.agenerate_speech(
        text="Hello, world!",
        voice="gb_jane_neutral",
    )
    with open("output.mp3", "wb") as f:
        f.write(response.audio_data)

asyncio.run(main())
```

## Response Object

`generate_speech` and `agenerate_speech` return an `AudioResponse`:

| Field | Type | Description |
|-------|------|-------------|
| `audio_data` | `bytes` | Raw audio bytes (base64-decoded from API) |
| `content_type` | `str` | MIME type (e.g., `audio/mp3`) |
| `model` | `str` | Model used |
| `voice` | `str` | Voice ID used |
| `provider` | `str` | Always `"mistral"` |
| `metadata` | `dict` | Contains the original `text` |

## Error Handling

```python
try:
    response = tts.generate_speech(text="Hello", voice="gb_jane_neutral")
except RuntimeError as e:
    print(f"TTS failed: {e}")
```

Errors raised by the Mistral API are wrapped in `RuntimeError` with the message prefix `"Mistral API error: ..."`.
