# Mistral Speech-to-Text

## Overview

Mistral provides speech-to-text transcription via its Voxtral models.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Speech-to-Text | ✅ | Voxtral models |
| Language auto-detection | ✅ | Returned in response |

**Official Documentation:** https://docs.mistral.ai/capabilities/speech-to-text/

## Prerequisites

### Account Requirements

- Mistral account (sign up at https://console.mistral.ai)
- API key with access to audio endpoints

### Getting API Keys

1. Visit https://console.mistral.ai/api-keys
2. Create a new API key
3. Copy and store it securely

## Environment Variables

```bash
MISTRAL_API_KEY="your-api-key"
```

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

transcriber = AIFactory.create_speech_to_text("mistral")
response = transcriber.transcribe("audio.mp3")
print(response.text)
print(response.language)  # auto-detected by Mistral
```

### Direct Instantiation

```python
from esperanto.providers.stt.mistral import MistralSpeechToTextModel

transcriber = MistralSpeechToTextModel(api_key="your-key")
response = transcriber.transcribe("audio.mp3")
```

## Models

| Model | Description |
|-------|-------------|
| `voxtral-mini-latest` | Default; fast, lightweight |
| `voxtral-small-latest` | Higher accuracy |

```python
transcriber = AIFactory.create_speech_to_text("mistral", "voxtral-small-latest")
```

## Options

```python
response = transcriber.transcribe(
    "audio.mp3",
    language="fr",              # optional ISO 639-1 hint
    prompt="Podcast sur l'IA",  # optional context
)
```

## Async Usage

```python
import asyncio
from esperanto.factory import AIFactory

async def main():
    transcriber = AIFactory.create_speech_to_text("mistral")
    response = await transcriber.atranscribe("audio.mp3")
    print(response.text)

asyncio.run(main())
```

## Response Format

```python
TranscriptionResponse(
    text="Transcribed text here",
    language="en",       # detected language from Mistral
    duration=12.4,        # audio duration in seconds (when provided)
    model="voxtral-mini-latest",
    provider="mistral",
    segments=[
        TranscriptionSegment(
            text="Transcribed text here",
            start=0.0,
            end=12.4,
            metadata={
                "id": 0,
                "confidence": 0.95,
                "speaker": "speaker_0",
                "language": "en",
            },
        ),
        # ... more segments
    ],
    usage=TranscriptionUsage(
        input_seconds=12.4,   # audio seconds billed (Mistral-specific)
        input_tokens=120,
        output_tokens=80,
        total_tokens=200,
    ),
)
```

### Iterating over segments

Mistral natively returns timestamped segments. Esperanto exposes them on
`response.segments` so you can render captions, build subtitles, or extract
per-speaker turns:

```python
response = transcriber.transcribe("interview.mp3")

if response.segments:
    for segment in response.segments:
        speaker = segment.metadata.get("speaker") if segment.metadata else None
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {speaker}: {segment.text}")

if response.usage and response.usage.input_seconds:
    print(f"Billed for {response.usage.input_seconds:.2f}s of audio")
```

## Error Handling

```python
try:
    response = transcriber.transcribe("audio.mp3")
except RuntimeError as e:
    print(f"API error: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
```
