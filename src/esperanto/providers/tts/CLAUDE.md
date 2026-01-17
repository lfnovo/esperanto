# Text-to-Speech Providers

Text-to-speech (TTS) provider implementations for generating audio from text.

## Files

- **`base.py`**: Abstract base class `TextToSpeechModel` defining the interface
- **`openai.py`**: OpenAI TTS API (voices: alloy, echo, fable, nova, onyx, shimmer)
- **`elevenlabs.py`**: ElevenLabs TTS with multilingual support
- **`google.py`**: Google Cloud Text-to-Speech (massive voice catalog)
- **`vertex.py`**: Google Vertex AI TTS
- **`azure.py`**: Azure Speech Service TTS
- **`openai_compatible.py`**: Generic OpenAI-compatible TTS API

## Patterns

### Base Class Contract

All providers inherit from `TextToSpeechModel` (base.py:16) and must:

1. **Implement abstract methods**:
   - `generate_speech()`: Synchronous speech generation
   - `agenerate_speech()`: Async speech generation
   - `_get_models()`: Return list of available models
   - `available_voices` property: Return dict of available voices
   - `provider` property: Return provider name string

2. **Override `__post_init__()`**:
   - Call `super().__post_init__()` first
   - Set `api_key` from parameter or environment variable
   - Set `base_url` (if applicable)
   - Call `self._create_http_clients()` last

3. **Return standardized response**:
   - Use `AudioResponse` from `esperanto.common_types.tts`
   - Contains `audio_data` (bytes), `format`, `duration`, `voice_used`

### Voice Management

Each provider must implement `available_voices` property (base.py:107):

- Returns `Dict[str, Voice]` mapping voice ID → Voice object
- `Voice` contains: `id`, `name`, `language`, `gender`, optional `description`
- Used for voice discovery and validation

Example:

```python
@property
def available_voices(self) -> Dict[str, Voice]:
    return {
        "alloy": Voice(id="alloy", name="Alloy", language="en", gender="neutral"),
        "echo": Voice(id="echo", name="Echo", language="en", gender="male"),
        # ...
    }
```

### SSML Support

Base class defines `COMMON_SSML_TAGS` (base.py:35):

- Standard SSML tags: speak, break, emphasis, prosody, say-as, voice, audio, p, s, phoneme, sub
- Providers can override `get_supported_tags()` to return their specific tags
- Not all providers support SSML - check docs

Some providers support SSML natively, others need custom handling:

- **Google**: Full SSML support
- **ElevenLabs**: Limited support (some tags)
- **OpenAI**: No SSML support (plain text only)

### Parameter Validation

Base class provides `validate_parameters()` (base.py:176):

- Checks text is non-empty string
- Checks voice is non-empty string
- Checks model is string (if provided)
- Call before making API request

```python
def generate_speech(self, text: str, voice: str, output_file: Optional[Union[str, Path]] = None, **kwargs):
    self.validate_parameters(text, voice)
    # Proceed with generation...
```

### File Saving

Base class provides `save_audio()` helper (base.py:198):

- Saves audio bytes to file
- Creates parent directories if needed
- Returns absolute path to saved file
- Handles IOError gracefully

Use in implementations:

```python
audio_data = api_response.content

if output_file:
    file_path = self.save_audio(audio_data, output_file)
else:
    file_path = None

return AudioResponse(audio_data=audio_data, format="mp3", output_file=file_path, ...)
```

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

Build `AudioResponse` from API results:

```python
from esperanto.common_types.tts import AudioResponse

return AudioResponse(
    audio_data=audio_bytes,
    format="mp3",  # or "wav", "ogg", etc.
    voice_used=voice,
    model_used=self.get_model_name(),
    output_file=file_path if output_file else None,
    duration=duration,  # if available
)
```

## Integration

- Imported by `factory.py` via `AIFactory._provider_modules["text_to_speech"]`
- Uses `AudioResponse`, `Voice` from `esperanto.common_types.tts`
- Inherits mixins from `esperanto.utils.timeout` and `esperanto.utils.ssl`

## Gotchas

- **Voice validation**: Check voice exists in `available_voices` before using
- **Audio format**: Different providers support different formats (MP3, WAV, OGG, etc.)
- **File extension**: Ensure output_file has correct extension for audio format
- **Text length limits**: APIs have character limits (OpenAI: 4096 chars, ElevenLabs: varies by plan)
- **Streaming support**: Some providers support streaming audio (OpenAI, ElevenLabs) - handle separately
- **Voice IDs vs names**: Some providers use IDs (google voice IDs), others use names
- **Language codes**: Different providers use different formats (Google uses BCP-47)
- **Speed/pitch/volume**: Providers expose these differently - map to common parameters or kwargs
- **SSML validation**: If provider doesn't support SSML, strip tags or raise error
- **Audio quality**: Providers offer different quality settings (bitrate, sample rate)
- **Billing**: TTS is billed per character - be mindful of costs
- **Deprecation warnings**: Base class doesn't have model deprecation (no `.models` on TTS yet)

## When Adding a New Provider

1. Create new file `provider_name.py`
2. Import `TextToSpeechModel` from `esperanto.providers.tts.base`
3. Import `AudioResponse`, `Voice` from `esperanto.common_types.tts`
4. Define class inheriting from `TextToSpeechModel`
5. Implement all abstract methods and properties
6. Define `available_voices` property with provider's voice catalog
7. Add `__post_init__()` following the pattern
8. Use `validate_parameters()` for input validation
9. Use `save_audio()` for file saving
10. Add provider to `factory.py` in `_provider_modules["text_to_speech"]` dict
11. Write tests in `tests/providers/tts/test_provider_name.py`
12. Add documentation in `docs/providers/provider_name.md`

## Special Cases

### Google TTS

- Massive voice catalog (hundreds of voices across languages)
- Full SSML support with custom Google tags
- Supports audio effects (e.g., telephony, headphones)
- Uses `google-cloud-texttospeech` library (not just HTTP)
- Requires GCP credentials (JSON key or ADC)
- Different language code format (BCP-47: "en-US" not "en")

### ElevenLabs

- High-quality voices with emotional range
- Supports voice cloning (custom voices)
- SSML support for some tags
- Different pricing tiers with different features
- Streaming support for long-form content
- Multilingual voices (single voice, multiple languages)

### OpenAI

- Six preset voices: alloy, echo, fable, nova, onyx, shimmer
- Two models: tts-1 (fast), tts-1-hd (high quality)
- No SSML support (plain text only)
- Supports speed adjustment (0.25-4.0x)
- Returns MP3 or other formats based on request

### Azure

- Uses Speech SDK or REST API
- Different authentication (subscription key + region)
- Supports neural voices and custom voices
- Full SSML support with Azure-specific tags
- Real-time streaming support

### Vertex AI

- Google's Vertex AI TTS (different from Cloud TTS)
- Similar to Google TTS but integrated with Vertex platform
- Requires Vertex AI setup and permissions

## Common Implementation Patterns

### Generating Speech

```python
def generate_speech(
    self,
    text: str,
    voice: str,
    output_file: Optional[Union[str, Path]] = None,
    **kwargs
) -> AudioResponse:
    self.validate_parameters(text, voice)

    # Build request payload
    payload = {
        "model": self.get_model_name(),
        "voice": voice,
        "input": text,
    }
    payload.update(kwargs)  # Add provider-specific params

    # Make API request
    response = self.client.post(
        f"{self.base_url}/audio/speech",
        json=payload,
        headers=self._get_headers()
    )
    response.raise_for_status()

    audio_data = response.content

    # Save to file if requested
    file_path = None
    if output_file:
        file_path = self.save_audio(audio_data, output_file)

    return AudioResponse(
        audio_data=audio_data,
        format="mp3",
        voice_used=voice,
        model_used=self.get_model_name(),
        output_file=file_path
    )
```

### Handling SSML

```python
def generate_speech(self, text: str, voice: str, output_file: Optional[Union[str, Path]] = None, **kwargs):
    # Check if text contains SSML
    is_ssml = text.strip().startswith("<speak>")

    if is_ssml:
        if not self._supports_ssml():
            # Option 1: Strip SSML tags
            import re
            text = re.sub(r'<[^>]+>', '', text)
            # Option 2: Raise error
            # raise ValueError("This provider does not support SSML")

    # Proceed with generation...
```

### Voice Discovery

```python
@property
def available_voices(self) -> Dict[str, Voice]:
    # For API providers with dynamic voice lists, cache this
    if not hasattr(self, "_voices_cache"):
        response = self.client.get(f"{self.base_url}/voices")
        response.raise_for_status()
        voices_data = response.json()

        self._voices_cache = {
            v["voice_id"]: Voice(
                id=v["voice_id"],
                name=v["name"],
                language=v.get("language", "en"),
                gender=v.get("gender", "neutral")
            )
            for v in voices_data
        }

    return self._voices_cache
```

### Async Implementation

```python
async def agenerate_speech(
    self,
    text: str,
    voice: str,
    output_file: Optional[Union[str, Path]] = None,
    **kwargs
) -> AudioResponse:
    self.validate_parameters(text, voice)

    payload = {
        "model": self.get_model_name(),
        "voice": voice,
        "input": text,
    }
    payload.update(kwargs)

    response = await self.async_client.post(
        f"{self.base_url}/audio/speech",
        json=payload,
        headers=self._get_headers()
    )
    response.raise_for_status()

    audio_data = response.content

    file_path = None
    if output_file:
        file_path = self.save_audio(audio_data, output_file)

    return AudioResponse(
        audio_data=audio_data,
        format="mp3",
        voice_used=voice,
        model_used=self.get_model_name(),
        output_file=file_path
    )
```
