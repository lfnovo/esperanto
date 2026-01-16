# Speech-to-Text (STT)

## Overview

Speech-to-Text models transcribe audio files into text. They support various audio formats, languages, and can include features like timestamp generation, speaker identification, and translation.

## Common Use Cases

- **Transcription**: Meeting notes, podcast transcripts, interview documentation
- **Voice Interfaces**: Voice commands, dictation, voice search
- **Content Accessibility**: Captions for videos, subtitles for media
- **Audio Analysis**: Call center analytics, compliance monitoring

## Interface

### Creating a Speech-to-Text Model

```python
from esperanto.factory import AIFactory

# Basic usage
transcriber = AIFactory.create_speech_to_text(
    provider="openai",
    model_name="whisper-1"
)

# With configuration
transcriber = AIFactory.create_speech_to_text(
    provider="groq",
    model_name="whisper-large-v3",
    config={
        "timeout": 300.0,  # Longer timeout for large files
        "language": "en",
        "response_format": "json"
    }
)
```

### Core Methods

#### `transcribe(audio_file, **kwargs)`

Synchronous audio transcription from file path.

```python
# From file path
transcript = transcriber.transcribe("/path/to/audio.mp3")
print(transcript)

# From file object
with open("/path/to/audio.mp3", "rb") as f:
    transcript = transcriber.transcribe(f)
print(transcript)
```

#### `atranscribe(audio_file, **kwargs)`

Asynchronous transcription (identical interface to `transcribe`).

```python
transcript = await transcriber.atranscribe("/path/to/audio.mp3")
print(transcript)
```

## Parameters

### Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | float | 300.0 | Request timeout in seconds (STT operations take longer) |
| `language` | str | None | Source language (ISO-639-1 code, e.g., "en", "es") |
| `response_format` | str | "json" | Output format: "json", "text", "srt", "vtt", "verbose_json" |
| `temperature` | float | 0.0 | Sampling temperature (0.0 = deterministic) |

### Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_file` | str or file | Required | File path or file object to transcribe |
| `prompt` | str | None | Optional context to guide transcription |

### Supported Audio Formats

Common supported formats (varies by provider):
- **MP3**: Most widely supported
- **WAV**: Uncompressed audio
- **M4A**: Apple audio format
- **FLAC**: Lossless compression
- **OGG**: Open format
- **WebM**: Web multimedia

## Response Structure

### Text Format (Default)

Most providers return plain text:

```python
transcriber = AIFactory.create_speech_to_text("openai", "whisper-1")
transcript = transcriber.transcribe("audio.mp3")
print(transcript)
# Output: "This is the transcribed text from the audio file."
```

### JSON Format

Request structured output:

```python
transcriber = AIFactory.create_speech_to_text(
    "openai", "whisper-1",
    config={"response_format": "json"}
)

result = transcriber.transcribe("audio.mp3")
# Returns: {"text": "Transcribed content..."}
```

### Verbose JSON Format

Get detailed information including timestamps:

```python
transcriber = AIFactory.create_speech_to_text(
    "openai", "whisper-1",
    config={"response_format": "verbose_json"}
)

result = transcriber.transcribe("audio.mp3")
# Returns:
# {
#   "text": "Full transcription...",
#   "language": "en",
#   "duration": 125.5,
#   "segments": [
#     {"start": 0.0, "end": 3.2, "text": "First segment"},
#     ...
#   ]
# }
```

### Subtitle Formats

Generate subtitle files directly:

```python
# SRT format (SubRip)
transcriber = AIFactory.create_speech_to_text(
    "openai", "whisper-1",
    config={"response_format": "srt"}
)

srt_content = transcriber.transcribe("audio.mp3")
# Returns SRT format string

# VTT format (WebVTT)
transcriber = AIFactory.create_speech_to_text(
    "openai", "whisper-1",
    config={"response_format": "vtt"}
)

vtt_content = transcriber.transcribe("audio.mp3")
# Returns VTT format string
```

## Provider Selection

→ **See [Provider Comparison](../providers/README.md)** for detailed comparison and selection guide.

### Quick Provider Guide

- **OpenAI**: Industry standard Whisper model, excellent accuracy
- **Groq**: Fastest transcription, Whisper-based, low latency
- **Google**: Gemini audio transcription, supports multiple formats, simple API key auth
- **Azure**: Enterprise compliance, private deployment
- **ElevenLabs**: Specialized for voice, good multilingual support
- **OpenAI-Compatible**: Local Whisper deployment (faster-whisper, etc.)

## Examples

### Basic Transcription

```python
from esperanto.factory import AIFactory

transcriber = AIFactory.create_speech_to_text("openai", "whisper-1")

# Transcribe audio file
transcript = transcriber.transcribe("meeting_recording.mp3")
print(transcript)
```

### With Language Specification

```python
transcriber = AIFactory.create_speech_to_text(
    "openai", "whisper-1",
    config={"language": "es"}  # Spanish
)

transcript = transcriber.transcribe("spanish_audio.mp3")
print(transcript)
```

### Generate Subtitles

```python
transcriber = AIFactory.create_speech_to_text(
    "openai", "whisper-1",
    config={"response_format": "srt"}
)

# Get SRT subtitle content
srt_content = transcriber.transcribe("video_audio.mp3")

# Save to file
with open("subtitles.srt", "w") as f:
    f.write(srt_content)
```

### With Context Prompt

```python
transcriber = AIFactory.create_speech_to_text("groq", "whisper-large-v3")

# Provide context for better accuracy
prompt = "This is a technical discussion about machine learning and neural networks."

transcript = transcriber.transcribe(
    "tech_talk.mp3",
    prompt=prompt
)
print(transcript)
```

### Google (Gemini) Transcription

```python
# Basic Gemini transcription
transcriber = AIFactory.create_speech_to_text("google", "gemini-2.5-flash")

transcript = transcriber.transcribe("audio.mp3")
print(transcript.text)

# With language hint for better accuracy
transcript = transcriber.transcribe(
    "portuguese_audio.mp3",
    language="pt"
)
print(transcript.text)

# With custom prompt for specialized content
transcript = transcriber.transcribe(
    "medical_consultation.mp3",
    prompt="This is a medical consultation. Focus on medical terminology and patient symptoms."
)
print(transcript.text)
```

> **Note**: Esperanto's Google STT provider uses Gemini API's audio transcription capabilities, not Cloud Speech-to-Text API v2 (Chirp 3). This provides simpler authentication (API key only) and consistent integration with other Google GenAI features. Supported formats: MP3, WAV, AIFF, AAC, OGG, FLAC.

### Async Batch Processing

```python
transcriber = AIFactory.create_speech_to_text(
    "openai", "whisper-1",
    config={"timeout": 600.0}  # 10 minutes for large files
)

audio_files = [
    "episode_01.mp3",
    "episode_02.mp3",
    "episode_03.mp3"
]

async def transcribe_all():
    tasks = [
        transcriber.atranscribe(file)
        for file in audio_files
    ]
    return await asyncio.gather(*tasks)

transcripts = await transcribe_all()
for i, transcript in enumerate(transcripts, 1):
    print(f"\n=== Episode {i} ===")
    print(transcript[:200], "...")
```

### Detailed Transcription with Timestamps

```python
transcriber = AIFactory.create_speech_to_text(
    "openai", "whisper-1",
    config={"response_format": "verbose_json"}
)

result = transcriber.transcribe("interview.mp3")

print(f"Language detected: {result['language']}")
print(f"Duration: {result['duration']:.2f} seconds")
print("\nSegments:")

for segment in result['segments']:
    start = segment['start']
    end = segment['end']
    text = segment['text']
    print(f"[{start:.2f}s - {end:.2f}s] {text}")
```

### Error Handling

```python
from esperanto.factory import AIFactory

transcriber = AIFactory.create_speech_to_text("openai", "whisper-1")

try:
    transcript = transcriber.transcribe("audio.mp3")
    print(transcript)
except FileNotFoundError:
    print("Audio file not found")
except Exception as e:
    print(f"Transcription error: {e}")
```

## Best Practices

### File Size Limits

Most providers have file size limits (typically 25MB). For larger files:

```python
# Option 1: Split audio file before transcription
# Use ffmpeg or similar tool to split into chunks

# Option 2: Use streaming if supported by provider

# Option 3: Compress audio file
# ffmpeg -i input.wav -ar 16000 -ac 1 output.mp3
```

### Audio Quality Recommendations

- **Sample rate**: 16kHz minimum for speech
- **Format**: MP3 or M4A for good compression
- **Mono vs Stereo**: Mono sufficient for speech, saves bandwidth
- **Bitrate**: 64-128 kbps adequate for voice

### Accuracy Tips

1. **Clear audio**: Reduce background noise before transcription
2. **Specify language**: Better accuracy than auto-detection
3. **Use prompts**: Provide context for technical terms or names
4. **Choose right model**: Larger models (e.g., whisper-large) for difficult audio

### Performance Considerations

- **Timeout**: Allow 1-2 minutes per minute of audio minimum
- **Async processing**: Use `atranscribe()` for multiple files
- **Local deployment**: OpenAI-Compatible with faster-whisper for best speed

## Advanced Topics

- **Timeout Configuration**: [docs/advanced/timeout-configuration.md](../advanced/timeout-configuration.md)
- **Resource Management**: [docs/advanced/connection-resource-management.md](../advanced/connection-resource-management.md)

## See Also

- [Provider Setup Guides](../providers/README.md)
- [Text-to-Speech](./text-to-speech.md)
- [Language Models](./llm.md)
