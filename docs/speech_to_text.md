# Speech-to-Text (STT)

Esperanto provides a unified interface for Speech-to-Text services across different providers. This guide explains how to use STT services.

## Interface

All Speech-to-Text models implement the following interface:

```python
async def transcribe(self, audio_path: str) -> str:
    """Transcribe audio file to text."""

async def transcribe_bytes(self, audio_bytes: bytes, mime_type: str) -> str:
    """Transcribe audio bytes to text."""
```

## Basic Usage

```python
from esperanto.factory import AIFactory

# Create a speech-to-text model instance
stt = AIFactory.create_stt(
    provider="openai",  # Choose your provider
    model_name="whisper-1",  # Model name specific to the provider
    config={
        "language": "en",  # Optional: Specify language
        "prompt": "This is a technical discussion",  # Optional: Guide transcription
    }
)

# Transcribe from file
transcript = await stt.transcribe("path/to/audio.mp3")

# Transcribe from bytes
with open("path/to/audio.mp3", "rb") as f:
    audio_bytes = f.read()
transcript = await stt.transcribe_bytes(audio_bytes, "audio/mp3")
```

## Supported Providers

### OpenAI (Whisper)
```python
stt = AIFactory.create_stt(
    provider="openai",
    model_name="whisper-1",
    config={
        "api_key": "your-api-key",  # Optional: defaults to OPENAI_API_KEY env var
        "language": "en",           # Optional: Specify language
        "prompt": "",              # Optional: Guide transcription
        "temperature": 0,          # Optional: Model temperature
    }
)
```

### Groq
```python
stt = AIFactory.create_stt(
    provider="groq",
    model_name="whisper-large-v3",  # or other supported models
    config={
        "api_key": "your-api-key",  # Optional: defaults to GROQ_API_KEY env var
        "language": "en",           # Optional: Specify language
    }
)
```

## Supported Audio Formats

- MP3
- MP4
- MPEG
- MPGA
- M4A
- WAV
- WEBM

## Error Handling

```python
try:
    stt = AIFactory.create_stt("openai", "whisper-1")
    transcript = await stt.transcribe("audio.mp3")
except ImportError as e:
    print("Provider dependencies not installed:", e)
except ValueError as e:
    print("Invalid configuration:", e)
except FileNotFoundError as e:
    print("Audio file not found:", e)
except Exception as e:
    print("Error during transcription:", e)
```

## Best Practices

1. **Audio Quality**:
   - Use high-quality audio recordings
   - Minimize background noise
   - Ensure clear speech

2. **File Size**:
   - Keep files under 25MB
   - For longer audio, consider splitting into chunks

3. **Language Hints**:
   - When the language is known, specify it in the config
   - Use prompts to guide transcription for domain-specific content

4. **Error Handling**:
   - Always implement proper error handling
   - Check for file existence and format support
   - Handle API rate limits and quotas
