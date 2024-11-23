# Text-to-Speech (TTS)

Esperanto provides a unified interface for Text-to-Speech services across different providers. This guide explains how to use TTS services.

## Interface

All Text-to-Speech models implement the following interface:

```python
async def synthesize(
    self, 
    text: str,
    voice: Optional[str] = None,
    output_path: Optional[str] = None
) -> Union[bytes, str]:
    """Synthesize text to speech."""
```

## Basic Usage

```python
from esperanto.factory import AIFactory

# Create a text-to-speech model instance
tts = AIFactory.create_tts(
    provider="openai",  # Choose your provider
    config={
        "voice": "alloy",  # Optional: Choose voice
        "model": "tts-1",  # Optional: Choose model
        "speed": 1.0,     # Optional: Adjust speed
    }
)

# Synthesize to bytes
audio_bytes = await tts.synthesize(
    text="Hello, world!",
    voice="alloy"  # Optional: Override config voice
)

# Synthesize to file
output_path = await tts.synthesize(
    text="Hello, world!",
    voice="alloy",
    output_path="output.mp3"
)
```

## Supported Providers

### OpenAI
```python
tts = AIFactory.create_tts(
    provider="openai",
    config={
        "api_key": "your-api-key",  # Optional: defaults to OPENAI_API_KEY env var
        "model": "tts-1",           # Optional: TTS model to use
        "voice": "alloy",           # Optional: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        "speed": 1.0,               # Optional: Speech speed (0.25 to 4.0)
    }
)
```

### ElevenLabs
```python
tts = AIFactory.create_tts(
    provider="elevenlabs",
    config={
        "api_key": "your-api-key",  # Optional: defaults to ELEVENLABS_API_KEY env var
        "voice": "Adam",            # Optional: Voice ID or name
        "model": "eleven_multilingual_v2",  # Optional: Model to use
        "stability": 0.5,           # Optional: Voice stability (0.0 to 1.0)
        "similarity_boost": 0.75,   # Optional: Voice similarity boost (0.0 to 1.0)
    }
)
```

### Google (Gemini)
```python
tts = AIFactory.create_tts(
    provider="gemini",
    config={
        "api_key": "your-api-key",  # Optional: defaults to GEMINI_API_KEY env var
        "voice": "en-US-Neural2-A",  # Optional: Voice name
        "language_code": "en-US",    # Optional: Language code
        "speaking_rate": 1.0,        # Optional: Speech speed
        "pitch": 0.0,               # Optional: Voice pitch
    }
)
```

## Voice Selection

Each provider offers different voices with unique characteristics:

### OpenAI
- alloy: Neutral and balanced
- echo: Warm and rounded
- fable: British accent
- onyx: Deep and authoritative
- nova: Warm and energetic
- shimmer: Clear and expressive

### ElevenLabs
- Extensive voice library
- Custom voice cloning available
- Multiple languages supported
- Voice customization options

### Google
- Wide range of natural-sounding voices
- Multiple languages and accents
- Neural2 voices for highest quality
- Custom voice options available

## Output Formats

The synthesized audio is provided in the following formats:

- OpenAI: MP3
- ElevenLabs: MP3
- Google: MP3, WAV, OGG

## Error Handling

```python
try:
    tts = AIFactory.create_tts("openai")
    audio = await tts.synthesize("Hello, world!")
except ImportError as e:
    print("Provider dependencies not installed:", e)
except ValueError as e:
    print("Invalid configuration:", e)
except Exception as e:
    print("Error during synthesis:", e)
```

## Best Practices

1. **Text Preparation**:
   - Use proper punctuation for natural pauses
   - Break long text into sentences
   - Use SSML for fine-grained control (where supported)

2. **Voice Selection**:
   - Choose voices appropriate for your content
   - Test different voices for best results
   - Consider language and accent requirements

3. **Audio Quality**:
   - Adjust speed and pitch for clarity
   - Use stability settings for consistent output
   - Consider file size vs. quality trade-offs

4. **Performance**:
   - Cache frequently used audio
   - Use streaming for long text
   - Handle rate limits appropriately
