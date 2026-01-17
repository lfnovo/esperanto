# Speech-to-Text Providers

Speech-to-text (STT) provider implementations for audio transcription.

## Files

- **`base.py`**: Abstract base class `SpeechToTextModel` defining the interface
- **`openai.py`**: OpenAI Whisper API
- **`groq.py`**: Groq Whisper inference
- **`google.py`**: Google Cloud Speech-to-Text
- **`azure.py`**: Azure Speech Service
- **`elevenlabs.py`**: ElevenLabs speech recognition
- **`openai_compatible.py`**: Generic OpenAI-compatible STT API

## Patterns

### Base Class Contract

All providers inherit from `SpeechToTextModel` (base.py:14) and must:

1. **Implement abstract methods**:
   - `transcribe()`: Synchronous transcription
   - `atranscribe()`: Async transcription
   - `_get_models()`: Return list of available models
   - `_get_default_model()`: Return default model name
   - `provider` property: Return provider name string

2. **Override `__post_init__()`**:
   - Call `super().__post_init__()` first
   - Set `api_key` from parameter or environment variable
   - Set `base_url` (if applicable)
   - Call `self._create_http_clients()` last

3. **Return standardized response**:
   - Use `TranscriptionResponse` from `esperanto.common_types`
   - Contains `text`, optional `language`, `duration`, and provider-specific metadata

### Audio File Handling

`transcribe()` accepts two input types (base.py:53):

1. **File path** (str): Read file from disk
2. **File-like object** (BinaryIO): Read from memory/stream

Handle both cases:

```python
def transcribe(self, audio_file: Union[str, BinaryIO], ...):
    if isinstance(audio_file, str):
        with open(audio_file, "rb") as f:
            audio_data = f.read()
    else:
        audio_data = audio_file.read()
    # Process audio_data...
```

For APIs that expect files, create multipart form data:

```python
files = {
    "file": ("audio.mp3", audio_data, "audio/mpeg")
}
```

### Language Support

Language parameter is optional (base.py:56):

- If provided: Use it to help model (improves accuracy)
- If None: Model auto-detects language
- Format: ISO 639-1 codes (e.g., "en", "es", "fr")

Some providers (Google) use different language codes - map them:

```python
# Google uses BCP-47 (e.g., "en-US")
if language:
    language_code = f"{language}-US"  # or appropriate mapping
```

### Prompt/Context

Optional `prompt` parameter (base.py:57) provides context:

- Helps model understand domain-specific terms
- Improves accuracy for technical/specialized content
- Not all providers support this (check before using)

### HTTP Client Pattern

Same as other providers:

```python
def __post_init__(self):
    super().__post_init__()
    self.api_key = self.api_key or os.getenv("PROVIDER_API_KEY")
    self.base_url = self.base_url or "https://api.provider.com/v1"
    self._create_http_clients()
```

### Response Construction

Build `TranscriptionResponse` from API results:

```python
from esperanto.common_types import TranscriptionResponse

return TranscriptionResponse(
    text=api_response["text"],
    language=api_response.get("language"),  # if available
    duration=audio_duration,  # if available
    segments=segments,  # if available (timestamped segments)
    words=words,  # if available (word-level timestamps)
)
```

## Integration

- Imported by `factory.py` via `AIFactory._provider_modules["speech_to_text"]`
- Uses `TranscriptionResponse` from `esperanto.common_types.stt`
- Inherits mixins from `esperanto.utils.timeout` and `esperanto.utils.ssl`

## Gotchas

- **File format support**: Check provider docs for supported audio formats (WAV, MP3, FLAC, etc.)
- **File size limits**: APIs have max file size (often 25MB) - check and chunk if needed
- **Audio duration limits**: Some providers limit duration (e.g., 30 minutes)
- **Multipart encoding**: When uploading files, use correct MIME type
- **File pointer position**: If using BinaryIO, reset pointer with `seek(0)` if needed
- **Language code formats**: Different providers use different formats (ISO 639-1 vs BCP-47)
- **Model availability**: Not all models support all languages
- **Timeout configuration**: Transcription can be slow - use longer timeouts via mixin
- **Prompt ignored**: Some providers don't support prompt parameter - handle gracefully
- **Deprecation warnings**: Use `_get_models()` internally (not `.models` property)
- **Response formats**: Some APIs support SRT, VTT, JSON with timestamps - handle appropriately

## When Adding a New Provider

1. Create new file `provider_name.py`
2. Import `SpeechToTextModel` from `esperanto.providers.stt.base`
3. Import `TranscriptionResponse` from `esperanto.common_types`
4. Define class inheriting from `SpeechToTextModel`
5. Implement all abstract methods
6. Add `__post_init__()` following the pattern
7. Handle both file path and BinaryIO inputs
8. Map language codes if provider uses different format
9. Add provider to `factory.py` in `_provider_modules["speech_to_text"]` dict
10. Write tests in `tests/providers/stt/test_provider_name.py`
11. Add documentation in `docs/providers/provider_name.md`

## Special Cases

### OpenAI Whisper

- Supports multiple formats: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM
- Max file size: 25MB
- Returns language, duration, and segments with timestamps
- Supports prompt for context and spelling hints

### Google Speech-to-Text

- Requires Google Cloud credentials (JSON key file or ADC)
- Uses `google-cloud-speech` library (not just HTTP)
- Different language code format (BCP-47: "en-US" not "en")
- Supports long-running transcription for files >1 minute
- Supports speaker diarization, word-level timestamps

### Groq

- Uses Whisper models but with fast inference
- OpenAI-compatible API format
- Same file size/format limits as OpenAI
- Much faster than standard Whisper

### Azure Speech Service

- Uses Speech SDK or REST API
- Different authentication (subscription key + region)
- Supports real-time streaming (beyond file transcription)
- Different response format than others

## Common Implementation Patterns

### Handling File Uploads

```python
def transcribe(self, audio_file: Union[str, BinaryIO], language: Optional[str] = None, prompt: Optional[str] = None):
    # Read file
    if isinstance(audio_file, str):
        with open(audio_file, "rb") as f:
            file_content = f.read()
        filename = Path(audio_file).name
    else:
        file_content = audio_file.read()
        filename = "audio.mp3"

    # Create multipart request
    files = {"file": (filename, file_content, "audio/mpeg")}
    data = {"model": self.get_model_name()}

    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    # Make request
    response = self.client.post(f"{self.base_url}/transcriptions", files=files, data=data)
```

### Error Handling

```python
try:
    response = self.client.post(url, files=files, data=data)
    response.raise_for_status()
    result = response.json()
except httpx.HTTPStatusError as e:
    raise RuntimeError(f"Transcription failed: {e.response.text}")
except Exception as e:
    raise RuntimeError(f"Transcription error: {str(e)}")
```

### Async Implementation

```python
async def atranscribe(self, audio_file: Union[str, BinaryIO], language: Optional[str] = None, prompt: Optional[str] = None):
    # Same file handling as sync version
    if isinstance(audio_file, str):
        with open(audio_file, "rb") as f:
            file_content = f.read()
        filename = Path(audio_file).name
    else:
        file_content = audio_file.read()
        filename = "audio.mp3"

    files = {"file": (filename, file_content, "audio/mpeg")}
    data = {"model": self.get_model_name()}

    # Use async_client
    response = await self.async_client.post(f"{self.base_url}/transcriptions", files=files, data=data)
    response.raise_for_status()
    result = response.json()

    return TranscriptionResponse(text=result["text"], ...)
```
