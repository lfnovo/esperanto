# Speech-to-Text Providers

Esperanto supports multiple speech-to-text providers for transcribing audio into text.

## Supported Providers

- OpenAI (Whisper)
- Groq (Whisper)

## Usage Examples

### Using AI Factory

```python
from esperanto.factory import AIFactory

# Create a speech-to-text instance
model = AIFactory.create_speech_to_text("openai", "whisper-1")

# Synchronous usage
with open("audio.mp3", "rb") as f:
    transcription = model.transcribe(f)

# Asynchronous usage
async def get_transcription():
    with open("audio.mp3", "rb") as f:
        transcription = await model.atranscribe(f)
```

### Basic Usage with OpenAI
```python
from esperanto.providers.stt.openai import OpenAISpeechToTextModel

model = OpenAISpeechToTextModel(
    api_key="your-api-key",  # or use ENV: OPENAI_API_KEY
    model_name="whisper-1"  # optional
)

# Transcribe from a file path
transcription = model.transcribe("audio.mp3")

# Transcribe from a file object
with open("audio.mp3", "rb") as f:
    transcription = model.transcribe(f)

# Transcribe with language and prompt
transcription = model.transcribe(
    "audio.mp3",
    language="en",  # optional, specify the audio language
    prompt="This is a podcast about AI"  # optional, guide the transcription
)
```

### Using Groq
```python
from esperanto.providers.stt.groq import GroqSpeechToTextModel

model = GroqSpeechToTextModel(
    api_key="your-api-key",  # or use ENV: GROQ_API_KEY
    model_name="whisper-1"  # optional
)

# Basic transcription
transcription = model.transcribe("audio.mp3")

# Async transcription
async def transcribe_async():
    transcription = await model.atranscribe("audio.mp3")
```

## Provider-Specific Configuration

Each provider may have specific configuration options. Here are some examples:

### OpenAI
```python
model = OpenAISpeechToTextModel(
    api_key="your-api-key",  # or use ENV: OPENAI_API_KEY
    model_name="whisper-1",
    base_url=None  # Optional, for custom API endpoint
)
```

### Groq
```python
model = GroqSpeechToTextModel(
    api_key="your-api-key",  # or use ENV: GROQ_API_KEY
    model_name="whisper-1",
    base_url=None  # Optional, for custom API endpoint
)
```

## Response Format

All providers return a standardized `TranscriptionResponse` object with the following attributes:

- `text`: The transcribed text
- `language`: The detected or specified language (if supported by the provider)
- `model`: The model used for transcription
- `provider`: The provider name

Example:
```python
response = model.transcribe("audio.mp3")
print(response.text)  # The transcribed text
print(response.language)  # The language (if available)
print(response.model)  # The model used
print(response.provider)  # The provider name
