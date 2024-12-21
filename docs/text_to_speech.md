# Text-to-Speech Providers

Esperanto supports multiple text-to-speech providers for converting text into natural-sounding speech.

## Supported Providers

- OpenAI (TTS-1, TTS-1-HD)
- ElevenLabs (Multilingual V2)
- Google Cloud TTS (Standard, Neural2)

## Usage Examples

### Using AI Factory

```python
from esperanto.factory import AIFactory

# Create a text-to-speech instance
model = AIFactory.create_text_to_speech("openai", "tts-1")

# Synchronous usage
response = model.generate_speech(
    text="Hello, world!",
    voice="alloy",
    output_file="output.mp3"  # optional
)

# Asynchronous usage
async def generate_audio():
    response = await model.agenerate_speech(
        text="Hello, world!",
        voice="alloy",
        output_file="output.mp3"  # optional
    )
```

### Basic Usage with OpenAI

```python
from esperanto.providers.tts.openai import OpenAITextToSpeechModel

model = OpenAITextToSpeechModel(
    api_key="your-api-key",  # or use ENV: OPENAI_API_KEY
    model_name="tts-1"  # optional, defaults to tts-1
)

# Generate speech
response = model.generate_speech(
    text="Hello, world!",
    voice="alloy",  # one of: alloy, echo, fable, onyx, nova, shimmer
    speed=1.0,  # optional, control speech speed (0.25 to 4.0)
    output_file="output.mp3"  # optional
)

# Play the audio (if you have a suitable audio player)
import IPython.display as ipd
ipd.Audio(response.audio_data)
```

### Basic Usage with ElevenLabs

```python
from esperanto.providers.tts.elevenlabs import ElevenLabsTextToSpeechModel

model = ElevenLabsTextToSpeechModel(
    api_key="your-api-key",  # or use ENV: ELEVENLABS_API_KEY
    model_name="eleven_multilingual_v2"  # optional
)

# Generate speech with voice settings
response = model.generate_speech(
    text="Hello, world!",
    voice="your-voice-id",  # get from model.available_voices
    stability=0.5,  # optional, voice stability (0.0 to 1.0)
    similarity_boost=0.8,  # optional, voice similarity boost (0.0 to 1.0)
    output_file="output.mp3"  # optional
)
```

### Basic Usage with Google Cloud TTS

```python
from esperanto.providers.tts.google import GoogleTextToSpeechModel

model = GoogleTextToSpeechModel(
    model_name="neural2",  # optional, one of: standard, neural2
    credentials={"your": "credentials"}  # or use ENV: GOOGLE_APPLICATION_CREDENTIALS
)

# Generate speech with SSML and voice settings
response = model.generate_speech(
    text='<speak>Hello <break time="1s"/> World!</speak>',
    voice="en-US-Neural2-A",  # get from model.available_voices
    speaking_rate=1.2,  # optional, control speech speed
    pitch=0,  # optional, control voice pitch
    output_file="output.mp3"  # optional
)
```

## Response Format

All providers return a standardized `AudioResponse` object:

```python
response = model.generate_speech(text="Hello!", voice="alloy")

print(response.audio_data)      # The audio bytes
print(response.content_type)    # e.g., "audio/mp3"
print(response.model)           # The model used
print(response.voice)           # The voice used
print(response.provider)        # The provider name
```

## Error Handling

```python
try:
    response = model.generate_speech(text="Hello!", voice="invalid-voice")
except ValueError as e:
    print("Invalid parameters:", e)
except Exception as e:
    print("API error:", e)
