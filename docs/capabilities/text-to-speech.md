# Text-to-Speech (TTS)

## Overview

Text-to-Speech models convert written text into natural-sounding audio. They support multiple voices, languages, and audio formats, enabling applications from voice assistants to audiobook generation.

## Common Use Cases

- **Voice Interfaces**: Virtual assistants, smart speakers, IVR systems
- **Content Accessibility**: Audiobooks, screen readers, narration
- **Media Production**: Voice-overs, podcast intro/outros, video narration
- **Notifications**: Voice alerts, announcements, voice-enabled apps

## Interface

### Creating a Text-to-Speech Model

```python
from esperanto.factory import AIFactory

# Basic usage
speaker = AIFactory.create_text_to_speech(
    provider="openai",
    model_name="tts-1"
)

# With configuration
speaker = AIFactory.create_text_to_speech(
    provider="elevenlabs",
    model_name="eleven_multilingual_v2",
    config={
        "timeout": 300.0,
        "voice": "default"
    }
)
```

### Core Methods

#### `generate_speech(text, voice=None, **kwargs)`

Synchronous speech generation returning audio bytes.

```python
# Generate audio bytes
audio_bytes = speaker.generate_speech(
    text="Hello, this is a test of text to speech.",
    voice="alloy"  # Provider-specific voice
)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

#### `agenerate_speech(text, voice=None, **kwargs)`

Asynchronous speech generation (identical interface to `generate_speech`).

```python
audio_bytes = await speaker.agenerate_speech(
    text="Async speech generation",
    voice="shimmer"
)
```

## Parameters

### Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | float | 300.0 | Request timeout in seconds (TTS can take longer) |
| `voice` | str | Provider default | Default voice for speech generation |
| `speed` | float | 1.0 | Speech rate (0.25 to 4.0, 1.0 = normal) |
| `response_format` | str | "mp3" | Audio format: "mp3", "opus", "aac", "flac", "wav", "pcm" |

### Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | Required | Text to convert to speech (max length varies by provider) |
| `voice` | str | Config default | Voice identifier (overrides config) |

## Available Voices

Voices vary significantly by provider. Common patterns:

### OpenAI Voices
- `alloy`: Neutral, balanced
- `echo`: Male, clear
- `fable`: Expressive, storytelling
- `onyx`: Deep, authoritative
- `nova`: Energetic, friendly
- `shimmer`: Warm, conversational

### ElevenLabs Voices
- Custom voices via voice ID
- Pre-made voices library
- Voice cloning available

### Google/Vertex Voices
- Language and gender variants
- Neural voices: `en-US-Neural2-A`, `en-US-Neural2-C`, etc.
- Standard voices: `en-US-Standard-A`, etc.

→ **See individual [Provider Setup Guides](../providers/README.md)** for complete voice lists.

## Response Structure

All TTS providers return audio bytes directly:

```python
speaker = AIFactory.create_text_to_speech("openai", "tts-1")

# Returns bytes object containing audio data
audio_bytes = speaker.generate_speech("Hello world", voice="alloy")

# Save directly to file
with open("speech.mp3", "wb") as f:
    f.write(audio_bytes)

# Or process in memory
# ... audio processing libraries can use audio_bytes directly
```

## Audio Formats

### Supported Formats (Provider-Dependent)

| Format | Use Case | Quality | Size |
|--------|----------|---------|------|
| `mp3` | General purpose | Good | Medium |
| `opus` | Streaming, VoIP | Good | Small |
| `aac` | Mobile, streaming | Good | Medium |
| `flac` | Lossless archival | Excellent | Large |
| `wav` | Uncompressed editing | Excellent | Very large |
| `pcm` | Raw audio processing | Excellent | Largest |

```python
# Specify format in config
speaker = AIFactory.create_text_to_speech(
    "openai", "tts-1",
    config={"response_format": "opus"}
)

audio = speaker.generate_speech("Test audio", voice="alloy")
```

## Provider Selection

→ **See [Provider Comparison](../providers/README.md)** for detailed comparison and selection guide.

### Quick Provider Guide

- **OpenAI**: High quality, natural voices, good pricing
- **ElevenLabs**: Best voice quality, voice cloning, emotional control
- **Google**: Wide language support, neural voices, good quality
- **Azure**: Enterprise compliance, custom neural voices
- **OpenAI-Compatible**: Local deployment options

## Examples

### Basic Speech Generation

```python
from esperanto.factory import AIFactory

speaker = AIFactory.create_text_to_speech("openai", "tts-1")

text = "Welcome to Esperanto text to speech."
audio = speaker.generate_speech(text, voice="nova")

with open("welcome.mp3", "wb") as f:
    f.write(audio)

print("Audio saved to welcome.mp3")
```

### Multiple Voices Comparison

```python
speaker = AIFactory.create_text_to_speech("openai", "tts-1")

text = "This is how I sound."
voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

for voice in voices:
    audio = speaker.generate_speech(text, voice=voice)
    with open(f"sample_{voice}.mp3", "wb") as f:
        f.write(audio)

print(f"Generated {len(voices)} audio samples")
```

### Adjust Speech Speed

```python
speaker = AIFactory.create_text_to_speech(
    "openai", "tts-1",
    config={"speed": 1.5}  # 1.5x faster
)

text = "This speech will be generated at 1.5 times normal speed."
audio = speaker.generate_speech(text, voice="alloy")

with open("fast_speech.mp3", "wb") as f:
    f.write(audio)
```

### Different Audio Formats

```python
from esperanto.factory import AIFactory

text = "Testing different audio formats."

# High quality for editing
speaker_flac = AIFactory.create_text_to_speech(
    "openai", "tts-1-hd",
    config={"response_format": "flac"}
)
audio_flac = speaker_flac.generate_speech(text, voice="alloy")

# Compressed for streaming
speaker_opus = AIFactory.create_text_to_speech(
    "openai", "tts-1",
    config={"response_format": "opus"}
)
audio_opus = speaker_opus.generate_speech(text, voice="alloy")

# Save both
with open("high_quality.flac", "wb") as f:
    f.write(audio_flac)

with open("compressed.opus", "wb") as f:
    f.write(audio_opus)
```

### Async Batch Generation

```python
speaker = AIFactory.create_text_to_speech("openai", "tts-1")

texts = [
    "Chapter 1: Introduction",
    "Chapter 2: Getting Started",
    "Chapter 3: Advanced Topics"
]

async def generate_all():
    tasks = [
        speaker.agenerate_speech(text, voice="fable")
        for text in texts
    ]
    return await asyncio.gather(*tasks)

audio_files = await generate_all()

# Save each chapter
for i, audio in enumerate(audio_files, 1):
    with open(f"chapter_{i}.mp3", "wb") as f:
        f.write(audio)
```

### Long Text Processing

```python
speaker = AIFactory.create_text_to_speech(
    "openai", "tts-1",
    config={"timeout": 600.0}  # 10 minutes for long texts
)

# Split long text into chunks (most providers have length limits)
def split_text(text, max_length=4000):
    """Split text into chunks at sentence boundaries."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

long_text = "..."  # Your long text here
chunks = split_text(long_text)

audio_chunks = []
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}")
    audio = speaker.generate_speech(chunk, voice="alloy")
    audio_chunks.append(audio)

# Combine audio chunks (requires audio processing library)
# Example with pydub:
# from pydub import AudioSegment
# combined = AudioSegment.empty()
# for audio_bytes in audio_chunks:
#     segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
#     combined += segment
# combined.export("complete.mp3", format="mp3")
```

### Stream and Play Audio

```python
import io
from esperanto.factory import AIFactory

speaker = AIFactory.create_text_to_speech("openai", "tts-1")

text = "This audio will be played directly."
audio_bytes = speaker.generate_speech(text, voice="nova")

# Play using pygame (example)
# import pygame
# pygame.mixer.init()
# sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
# sound.play()

# Or use another audio library
# import sounddevice as sd
# import soundfile as sf
# audio, samplerate = sf.read(io.BytesIO(audio_bytes))
# sd.play(audio, samplerate)
```

### Error Handling

```python
from esperanto.factory import AIFactory

speaker = AIFactory.create_text_to_speech("openai", "tts-1")

try:
    audio = speaker.generate_speech(
        "Test text",
        voice="alloy"
    )
    with open("output.mp3", "wb") as f:
        f.write(audio)
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Generation error: {e}")
```

## Best Practices

### Text Length Limits

Most providers limit input text length (typically 4096 characters for OpenAI). For longer content:

1. **Split at sentence boundaries**: Maintain natural speech flow
2. **Process in chunks**: Generate multiple audio files
3. **Combine audio**: Use audio processing library to merge

### Voice Selection

- **Consistency**: Use same voice for related content
- **Audience**: Match voice characteristics to target audience
- **Context**: Formal voices for professional content, friendly for casual
- **Test voices**: Generate samples to find best fit

### Audio Quality vs File Size

```python
# Production/archival: Use HD models and lossless formats
speaker = AIFactory.create_text_to_speech(
    "openai", "tts-1-hd",
    config={"response_format": "flac"}
)

# Streaming/mobile: Use standard models and compressed formats
speaker = AIFactory.create_text_to_speech(
    "openai", "tts-1",
    config={"response_format": "opus"}
)
```

### Performance Considerations

- **Timeout**: Allow 1-2 seconds per sentence minimum
- **Async processing**: Use `agenerate_speech()` for multiple texts
- **Caching**: Cache generated audio for repeated text
- **Batch processing**: Process multiple texts in parallel

## Advanced Topics

- **Timeout Configuration**: [docs/advanced/timeout-configuration.md](../advanced/timeout-configuration.md)

## See Also

- [Provider Setup Guides](../providers/README.md)
- [Speech-to-Text](./speech-to-text.md)
- [Language Models](./llm.md)
