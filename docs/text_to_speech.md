# Text-to-Speech Models

Text-to-speech (TTS) models convert written text into natural-sounding audio speech. These models can generate speech in various voices, languages, and styles, making them useful for accessibility applications, content creation, voice assistants, and multimedia production.

## Supported Providers

- **OpenAI** (TTS-1, TTS-1-HD models)
- **ElevenLabs** (Multilingual and specialized voice models)
- **Google Cloud** (Standard and Neural2 models)
- **Vertex AI** (Google Cloud text-to-speech models)

## Available Methods

All text-to-speech model providers implement the following methods:

- **`generate_speech(text, voice, output_file=None, **kwargs)`**: Generate audio from text
- **`agenerate_speech(text, voice, output_file=None, **kwargs)`**: Async version of generate_speech

### Parameters:
- `text`: Text to convert to speech
- `voice`: Voice identifier (varies by provider)
- `output_file`: Optional path to save audio file
- `**kwargs`: Provider-specific parameters (speed, pitch, etc.)

## Common Interface

All text-to-speech models return standardized response objects:

### AudioResponse
```python
response = model.generate_speech(text="Hello!", voice="alloy")
# Access attributes:
response.audio_data      # Raw audio bytes
response.content_type    # MIME type (e.g., "audio/mp3")
response.model           # Model used
response.voice           # Voice used
response.provider        # Provider name
```

## Examples

### Basic Speech Generation
```python
from esperanto.factory import AIFactory

# Create a text-to-speech model
model = AIFactory.create_text_to_speech("openai", "tts-1")

# Generate speech
response = model.generate_speech(
    text="Hello, world!",
    voice="alloy",
    output_file="greeting.mp3"
)

print(f"Generated {len(response.audio_data)} bytes of audio")
print(f"Content type: {response.content_type}")
```

### Async Speech Generation
```python
async def generate_audio_async():
    model = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")
    
    response = await model.agenerate_speech(
        text="This is an async speech generation example",
        voice="your-voice-id",
        output_file="async_output.mp3"
    )
    print(f"Audio saved to async_output.mp3")
```

### Batch Text-to-Speech
```python
from esperanto.factory import AIFactory

model = AIFactory.create_text_to_speech("google", "neural2")

# Generate multiple audio files
texts = [
    "Welcome to our service",
    "Please hold while we connect you",
    "Thank you for your patience",
    "Have a great day!"
]

for i, text in enumerate(texts):
    response = model.generate_speech(
        text=text,
        voice="en-US-Neural2-A",
        output_file=f"message_{i+1}.mp3",
        speaking_rate=1.1,
        pitch=0.0
    )
    print(f"Generated message_{i+1}.mp3: {text[:30]}...")
```

### Voice Customization
```python
# OpenAI with speed control
model = AIFactory.create_text_to_speech("openai", "tts-1-hd")
response = model.generate_speech(
    text="This speech has custom speed settings",
    voice="nova",
    speed=1.2,  # Faster speech
    output_file="fast_speech.mp3"
)

# ElevenLabs with voice settings
model = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")
response = model.generate_speech(
    text="This uses custom voice stability and similarity",
    voice="your-voice-id",
    stability=0.7,
    similarity_boost=0.8,
    output_file="custom_voice.mp3"
)

# Google with SSML markup
model = AIFactory.create_text_to_speech("google", "neural2")
response = model.generate_speech(
    text='<speak>Hello <break time="1s"/> <emphasis level="strong">world</emphasis>!</speak>',
    voice="en-US-Neural2-C",
    speaking_rate=0.9,
    pitch=-2.0,
    output_file="ssml_speech.mp3"
)
```

### Multiple Languages
```python
async def multilingual_generation():
    model = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")
    
    languages = {
        "english": ("Hello, how are you today?", "en-voice-id"),
        "spanish": ("Hola, ¿cómo estás hoy?", "es-voice-id"),
        "french": ("Bonjour, comment allez-vous aujourd'hui?", "fr-voice-id"),
        "german": ("Hallo, wie geht es dir heute?", "de-voice-id")
    }
    
    for lang, (text, voice_id) in languages.items():
        response = await model.agenerate_speech(
            text=text,
            voice=voice_id,
            output_file=f"greeting_{lang}.mp3"
        )
        print(f"Generated {lang} greeting")
```

### Audio Processing Integration
```python
import base64
from esperanto.factory import AIFactory

model = AIFactory.create_text_to_speech("openai", "tts-1")

# Generate speech and process audio data
response = model.generate_speech(
    text="This audio will be processed further",
    voice="shimmer"
)

# Convert to base64 for web applications
audio_base64 = base64.b64encode(response.audio_data).decode()
print(f"Base64 audio data: {audio_base64[:50]}...")

# Save with custom filename
with open("processed_audio.mp3", "wb") as f:
    f.write(response.audio_data)

# Display in Jupyter notebook
try:
    import IPython.display as ipd
    ipd.Audio(response.audio_data)
except ImportError:
    print("IPython not available for audio playback")
```

## Provider-Specific Information

### Voice Models and Selection

Each provider offers different voice models and selection methods. It's important to understand the available options for each provider:

**OpenAI Voices:**
- `alloy`: Balanced, neutral voice
- `echo`: Male voice with clarity
- `fable`: British accent, storytelling tone
- `onyx`: Deep, authoritative male voice  
- `nova`: Young, energetic female voice
- `shimmer`: Warm, expressive female voice

**ElevenLabs Voices:**
- Each account has access to different voice libraries
- Voice IDs are specific to your account and subscription
- Supports custom voice cloning and fine-tuning
- Offers multilingual capabilities with voice consistency
- **Multi-speaker Feature**: Supports text-to-dialogue for conversations with multiple speakers

**Google Cloud Voices:**
- 30 unique predefined voices with distinct personalities (e.g., achernar, charon, kore, puck)
- Each voice has specific characteristics: gender, tone, and personality traits
- Examples: `achernar` (UpbeatAchernar, Female), `charon` (UpbeatCharon, Male), `kore` (InformativeKore, Female)
- **Multi-speaker Feature**: Supports conversations with different voices per speaker

**ElevenLabs Voices:**
- Voice IDs are specific to your account and subscription
- Supports custom voice cloning and fine-tuning
- Offers multilingual capabilities with voice consistency
- **Multi-speaker Feature**: Now supports text-to-dialogue for conversations with multiple speakers

**Voice Discovery:**
```python
# List available voices (provider-dependent)
model = AIFactory.create_text_to_speech("google", "neural2")
try:
    voices = model.available_voices
    for voice in voices[:5]:  # Show first 5
        print(f"Voice: {voice['name']}, Language: {voice['language']}")
except AttributeError:
    print("Voice listing not available for this provider")
```

**Model Quality and Use Cases:**
- **Standard models**: Good for basic applications, lower cost
- **Neural/HD models**: Higher quality, more natural sounding, higher cost
- **Multilingual models**: Consistent voice across languages
- **Specialized models**: Optimized for specific use cases (news, conversation, etc.)

### Multi-Speaker Conversations

Both Google and ElevenLabs TTS providers offer multi-speaker features that allow you to create conversations with different voices for each speaker. This is perfect for creating dialogues, interviews, or multi-character audio content.

**Additional Methods for Google and ElevenLabs Providers:**
- **`generate_multi_speaker_speech(text, speaker_configs, output_file=None, **kwargs)`**: Generate conversation with multiple speakers
- **`agenerate_multi_speaker_speech(text, speaker_configs, output_file=None, **kwargs)`**: Async version of multi-speaker generation

**Multi-Speaker Example:**
```python
from esperanto.factory import AIFactory

# Create Google TTS model
model = AIFactory.create_text_to_speech("google", "gemini-2.5-flash-preview-tts")

# Define conversation with speaker names
conversation_text = """
Joe: Hi there! How are you doing today?
Jane: I'm doing great, thanks for asking! How about you?
Joe: I'm wonderful. Did you see the latest AI developments?
Jane: Yes! The multi-speaker TTS technology is really impressive.
"""

# Configure speakers with different voices
speaker_configs = [
    {"speaker": "Joe", "voice": "charon"},      # Male, upbeat voice
    {"speaker": "Jane", "voice": "kore"}       # Female, informative voice
]

# Generate multi-speaker audio
response = model.generate_multi_speaker_speech(
    text=conversation_text,
    speaker_configs=speaker_configs,
    output_file="conversation.wav"
)

print(f"Generated {len(response.audio_data)} bytes of multi-speaker audio")
print(f"Speakers: {[config['speaker'] for config in speaker_configs]}")
```

**Async Multi-Speaker Example:**
```python
async def create_dialogue():
    model = AIFactory.create_text_to_speech("google", "gemini-2.5-flash-preview-tts")
    
    # Define a more complex conversation
    interview_text = """
    Interviewer: Welcome to our tech podcast. Today we're discussing AI.
    Expert: Thank you for having me. AI is transforming every industry.
    Interviewer: What's the most exciting development you've seen recently?
    Expert: Multi-modal AI that can understand and generate text, images, and audio.
    Interviewer: That sounds fascinating. How will this impact developers?
    Expert: It will enable more natural human-computer interactions.
    """
    
    # Use different voice personalities
    speaker_configs = [
        {"speaker": "Interviewer", "voice": "puck"},      # Bright, engaging male voice
        {"speaker": "Expert", "voice": "sulafat"}         # Knowledgeable female voice
    ]
    
    response = await model.agenerate_multi_speaker_speech(
        text=interview_text,
        speaker_configs=speaker_configs,
        output_file="ai_interview.wav"
    )
    
    print(f"Interview audio generated: {response.content_type}")
    return response

# Run the async function
# response = await create_dialogue()
```

**Multi-Speaker Tips:**
- **Speaker Names**: Use consistent speaker names throughout your text
- **Voice Selection**: Choose voices with different characteristics for better distinction
- **Text Format**: Prefix each line with "SpeakerName:" for clear speaker identification
- **Voice Combinations**: Mix male/female voices or different personality types for variety
- **Content Length**: Works well for both short dialogues and longer conversations

### ElevenLabs Multi-Speaker Example

```python
from esperanto.factory import AIFactory

# Create ElevenLabs TTS model
model = AIFactory.create_text_to_speech("elevenlabs", "eleven_v3")

# Define conversation with speaker names
conversation_text = """
Joe: Hi there! How are you doing today?
Jane: I'm doing great, thanks for asking! How about you?
Joe: I'm wonderful. Did you see the latest AI developments?
Jane: Yes! The text-to-dialogue technology is really impressive.
"""

# Configure speakers with different voice IDs from your ElevenLabs account
speaker_configs = [
    {"speaker": "Joe", "voice": "JBFqnCBsd6RMkjVDRZzb"},    # Replace with your voice ID
    {"speaker": "Jane", "voice": "Aw4FAjKCGjjNkVhN1Xmq"}   # Replace with your voice ID
]

# Generate multi-speaker audio
response = model.generate_multi_speaker_speech(
    text=conversation_text,
    speaker_configs=speaker_configs,
    output_file="elevenlabs_conversation.mp3",
    model_id="eleven_v3",
    output_format="mp3_44100_128"
)

print(f"Generated {len(response.audio_data)} bytes of multi-speaker audio")
print(f"Speakers: {[config['speaker'] for config in speaker_configs]}")
```

**ElevenLabs Multi-Speaker Tips:**
- **Voice IDs**: Use actual voice IDs from your ElevenLabs account
- **Model Requirements**: Multi-speaker requires Eleven v3 API access
- **Text Format**: Same as Google - prefix each line with "SpeakerName:"
- **Custom Parameters**: Supports model_id, seed, output_format, and settings
- **Content Length**: Works well for both short dialogues and longer conversations

**Available Voice Personalities for Google Multi-Speaker:**
- **Engaging/Upbeat**: `achernar` (F), `charon` (M), `leda` (F)
- **Clear/Professional**: `algenib` (M), `umbriel` (M), `gacrux` (F)
- **Informative/Knowledgeable**: `kore` (F), `sulafat` (F), `laomedeia` (F)
- **Bright/Energetic**: `enceladus` (M), `puck` (M)
- **Smooth/Gentle**: `despina` (F), `erinome` (F), `sadachbia` (M)
