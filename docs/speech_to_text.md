# Speech-to-Text Models

Speech-to-text models convert audio recordings into written text through automatic speech recognition (ASR). These models can transcribe speech from various audio formats and languages, making them useful for creating transcripts, voice assistants, accessibility tools, and content analysis.

## Supported Providers

- **OpenAI** (Whisper models)
- **Groq** (Whisper models with faster inference)
- **ElevenLabs** (Multilingual speech recognition)
- **OpenAI-Compatible** (Any OpenAI-compatible STT endpoint)

## Available Methods

All speech-to-text model providers implement the following methods:

- **`transcribe(audio_file, language=None, prompt=None)`**: Transcribe audio file to text
- **`atranscribe(audio_file, language=None, prompt=None)`**: Async version of transcribe

### Parameters:
- `audio_file`: Audio file path (string) or file object to transcribe
- `language`: Optional language code to improve accuracy (e.g., "en", "es", "fr")
- `prompt`: Optional text to guide the transcription context

## Common Interface

All speech-to-text models return standardized response objects:

### TranscriptionResponse
```python
response = model.transcribe("audio.mp3")
# Access attributes:
response.text           # The transcribed text
response.language       # Detected or specified language
response.model          # Model used for transcription
response.provider       # Provider name
```

## Examples

### Basic Transcription
```python
from esperanto.factory import AIFactory

# Create a speech-to-text model
model = AIFactory.create_speech_to_text("openai", "whisper-1")

# Transcribe from file path
response = model.transcribe("audio.mp3")
print(response.text)

# Transcribe from file object
with open("audio.mp3", "rb") as f:
    response = model.transcribe(f)
    print(response.text)
```

### Async Transcription
```python
async def transcribe_async():
    model = AIFactory.create_speech_to_text("groq", "whisper-large-v3")
    
    response = await model.atranscribe("meeting.wav")
    print(f"Transcription: {response.text}")
    print(f"Language: {response.language}")
```

### Transcription with Context
```python
model = AIFactory.create_speech_to_text("openai", "whisper-1")

# Provide language and context for better accuracy
response = model.transcribe(
    "podcast.mp3",
    language="en",  # Specify language
    prompt="This is a technical podcast about machine learning and AI"  # Context
)
print(response.text)
```

### Batch Processing
```python
import os
from esperanto.factory import AIFactory

model = AIFactory.create_speech_to_text("groq", "whisper-large-v3")

# Process multiple audio files
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
    output_file = transcript["file"].replace(".mp3", ".txt").replace(".wav", ".txt").replace(".m4a", ".txt")
    with open(output_file, "w") as f:
        f.write(transcript["text"])
```

### Real-time Processing
```python
async def process_audio_stream():
    model = AIFactory.create_speech_to_text("elevenlabs", "speech-to-text-1")
    
    # Process audio files as they become available
    audio_queue = ["chunk1.wav", "chunk2.wav", "chunk3.wav"]
    
    for audio_chunk in audio_queue:
        response = await model.atranscribe(audio_chunk)
        print(f"Chunk transcription: {response.text}")
        
        # Process the transcription immediately
        if "urgent" in response.text.lower():
            print("🚨 Urgent content detected!")
```

## OpenAI-Compatible Provider

The OpenAI-compatible provider allows you to use any speech-to-text endpoint that follows the OpenAI API format. This includes local deployments, custom servers, and third-party services that implement OpenAI's `/audio/transcriptions` endpoint.

### Configuration

You can configure the provider using direct parameters, configuration dictionary, or environment variables:

```python
from esperanto.factory import AIFactory

# Using config dictionary
model = AIFactory.create_speech_to_text(
    "openai-compatible",
    model_name="faster-whisper-large-v3",
    config={
        "base_url": "http://localhost:8000",
        "api_key": "your-api-key-if-required",  # Optional
        "timeout": 600  # 10 minutes for large files (default: 300 seconds)
    }
)

# Using environment variables
# Set OPENAI_COMPATIBLE_BASE_URL=http://localhost:8000
# Set OPENAI_COMPATIBLE_API_KEY=your-api-key (optional)
model = AIFactory.create_speech_to_text("openai-compatible", "your-model-name")
```

### Supported Endpoints

The provider works with any OpenAI-compatible STT endpoint, including:

- **Ready-to-use implementations**:
  - [Speaches](https://github.com/speaches-ai/speaches/) - OpenAI-compatible server with faster-whisper support
- **Local deployments**: Custom faster-whisper, OpenAI Whisper, or other STT models
- **Self-hosted solutions**: Custom OpenAI-format STT servers
- **Development endpoints**: Local testing and development servers
- **Edge deployments**: On-premise or edge computing STT services

### API Compatibility

Your endpoint should implement the OpenAI Speech-to-Text API format:

**Required Endpoint**: `POST /audio/transcriptions`
```
Content-Type: multipart/form-data

file: <audio_file>
model: <model_name>
language: <language_code> (optional)
prompt: <context_prompt> (optional)
```

**Optional Endpoints**:
- `GET /models` - List available models

### Usage Examples

**Basic Usage:**
```python
from esperanto.factory import AIFactory

# Create OpenAI-compatible STT model
stt = AIFactory.create_speech_to_text(
    "openai-compatible",
    model_name="faster-whisper-large-v3",
    config={
        "base_url": "http://localhost:8000",
        "timeout": 300  # 5 minutes (default timeout)
    }
)

# Transcribe audio
response = stt.transcribe("meeting.mp3")
print(response.text)
```

**With Language and Prompt:**
```python
# Improve accuracy with language and context
response = stt.transcribe(
    "podcast.wav",
    language="en",
    prompt="This is a technical discussion about AI and machine learning"
)
print(f"Transcription: {response.text}")
```

**Async Transcription:**
```python
async def transcribe_batch():
    stt = AIFactory.create_speech_to_text(
        "openai-compatible",
        model_name="faster-whisper-large-v3",
        config={"base_url": "http://localhost:8000"}
    )

    files = ["audio1.mp3", "audio2.wav", "audio3.m4a"]
    for audio_file in files:
        response = await stt.atranscribe(audio_file)
        print(f"{audio_file}: {response.text[:100]}...")
```

### Error Handling

The provider includes graceful error handling and fallbacks:

```python
try:
    response = stt.transcribe("audio.mp3")
    print(response.text)
except RuntimeError as e:
    print(f"Transcription failed: {e}")
    # Handle error appropriately
```

### Environment Variables

**Generic (works for all OpenAI-compatible providers):**
- `OPENAI_COMPATIBLE_BASE_URL`: Base URL for your OpenAI-compatible STT endpoint
- `OPENAI_COMPATIBLE_API_KEY`: API key if your endpoint requires authentication

**Provider-specific (takes precedence, new in v2.7.0):**
- `OPENAI_COMPATIBLE_BASE_URL_STT`: Base URL specifically for STT endpoints
- `OPENAI_COMPATIBLE_API_KEY_STT`: API key specifically for STT endpoints

### Timeout Configuration

Speech-to-text operations can take significant time, especially for large audio files. The provider supports configurable timeouts:

**Default Timeout**: 300 seconds (5 minutes)

**Custom Timeout Examples:**
```python
# For large files (10 minutes)
stt = AIFactory.create_speech_to_text(
    "openai-compatible",
    model_name="faster-whisper-large-v3",
    config={
        "base_url": "http://localhost:8000",
        "timeout": 600  # 10 minutes
    }
)

# For very large files (30 minutes)
stt = AIFactory.create_speech_to_text(
    "openai-compatible",
    model_name="faster-whisper-large-v3",
    config={
        "base_url": "http://localhost:8000",
        "timeout": 1800  # 30 minutes
    }
)

# For quick tests (30 seconds)
stt = AIFactory.create_speech_to_text(
    "openai-compatible",
    model_name="faster-whisper-small",
    config={
        "base_url": "http://localhost:8000",
        "timeout": 30  # 30 seconds
    }
)
```

### Troubleshooting

**Common Issues:**

1. **Connection Error**: Ensure your STT endpoint is running and accessible
2. **Authentication Error**: Verify your API key or remove it if not required
3. **Model Not Found**: Check that your model name matches what's available on your endpoint
4. **Audio Format Error**: Ensure your audio file is in a supported format (MP3, WAV, M4A, etc.)
5. **Timeout Error**: For large audio files, increase the timeout value in your configuration

**Debugging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed request/response information
stt = AIFactory.create_speech_to_text("openai-compatible", ...)
```
