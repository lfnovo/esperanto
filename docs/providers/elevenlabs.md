# ElevenLabs

## Overview

ElevenLabs provides premium text-to-speech and speech-to-text capabilities with high-quality voice generation, voice cloning, and emotional control.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ❌ | Not available |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ✅ | Multilingual speech recognition |
| Text-to-Speech | ✅ | Premium voices, voice cloning, multi-speaker |

**Official Documentation:** https://elevenlabs.io/docs

## Prerequisites

### Account Requirements
- ElevenLabs account (sign up at https://elevenlabs.io)
- API key
- Active subscription for advanced features (voice cloning, higher limits)

### Getting API Keys
1. Visit https://elevenlabs.io/app/settings
2. Navigate to "API Key" section
3. Click "Generate new API key" or copy existing key
4. Store the key securely

## Environment Variables

```bash
# ElevenLabs API key (required)
ELEVENLABS_API_KEY="your-api-key"
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`ELEVENLABS_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Text-to-speech
speaker = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")

# Speech-to-text
transcriber = AIFactory.create_speech_to_text("elevenlabs", "speech-to-text-1")
```

### Direct Instantiation

```python
from esperanto.providers.text_to_speech.elevenlabs import ElevenLabsTextToSpeech
from esperanto.providers.speech_to_text.elevenlabs import ElevenLabsSpeechToText

# Text-to-speech
tts = ElevenLabsTextToSpeech(
    api_key="your-api-key",
    model_name="eleven_multilingual_v2"
)

# Speech-to-text
stt = ElevenLabsSpeechToText(
    api_key="your-api-key",
    model_name="speech-to-text-1"
)
```

## Capabilities

### Text-to-Speech

**Available Models:**
- **eleven_multilingual_v2** - Latest multilingual model, best quality
- **eleven_multilingual_v1** - Previous multilingual version
- **eleven_monolingual_v1** - English-only, optimized
- **eleven_turbo_v2** - Fast generation with good quality
- **eleven_turbo_v2_5** - Latest turbo model
- **eleven_v3** - Advanced model with multi-speaker support

**Configuration:**

```python
from esperanto.factory import AIFactory

speaker = AIFactory.create_text_to_speech(
    "elevenlabs",
    "eleven_multilingual_v2",
    config={
        "stability": 0.5,         # Voice stability (0.0-1.0)
        "similarity_boost": 0.75, # Voice similarity (0.0-1.0)
        "style": 0.0,             # Style exaggeration (0.0-1.0)
        "use_speaker_boost": True, # Enhance speaker clarity
        "timeout": 300.0          # Request timeout
    }
)
```

**Example - Basic Speech Generation:**

```python
from esperanto.factory import AIFactory

# Create text-to-speech model
model = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")

# Generate speech with voice ID from your account
response = model.generate_speech(
    text="Welcome to ElevenLabs! Experience premium voice quality.",
    voice="your-voice-id",  # Use voice ID from your ElevenLabs account
    output_file="welcome.mp3"
)

print(f"Generated {len(response.audio_data)} bytes of high-quality audio")
```

**Example - Voice Customization:**

```python
# Professional voice with high stability
response = model.generate_speech(
    text="Welcome to our professional presentation.",
    voice="your-professional-voice-id",
    stability=0.8,              # Higher stability (less variation)
    similarity_boost=0.9,       # Higher similarity to original voice
    output_file="professional.mp3"
)

# Expressive voice with more variation
response = model.generate_speech(
    text="This is an exciting announcement!",
    voice="your-expressive-voice-id",
    stability=0.3,              # Lower stability (more variation)
    similarity_boost=0.6,       # Less strict similarity
    style=0.5,                  # Some style exaggeration
    output_file="expressive.mp3"
)
```

**Example - Multilingual Content:**

```python
# Generate speech in multiple languages
languages = {
    "english": ("Hello, welcome to our global platform.", "en-voice-id"),
    "spanish": ("Hola, bienvenido a nuestra plataforma global.", "es-voice-id"),
    "french": ("Bonjour, bienvenue sur notre plateforme mondiale.", "fr-voice-id"),
    "german": ("Hallo, willkommen auf unserer globalen Plattform.", "de-voice-id")
}

for lang, (text, voice_id) in languages.items():
    response = model.generate_speech(
        text=text,
        voice=voice_id,
        output_file=f"greeting_{lang}.mp3"
    )
    print(f"Generated {lang} greeting")
```

**Example - High-Quality Settings:**

```python
# Maximum quality settings for production content
response = model.generate_speech(
    text="This content requires the highest audio quality.",
    voice="your-best-voice-id",
    stability=0.75,
    similarity_boost=0.85,
    use_speaker_boost=True,
    output_file="premium_quality.mp3"
)
```

**Example - Fast Generation with Turbo:**

```python
# Use turbo model for faster generation
turbo_model = AIFactory.create_text_to_speech("elevenlabs", "eleven_turbo_v2_5")

response = turbo_model.generate_speech(
    text="Quick response with good quality.",
    voice="your-voice-id",
    output_file="turbo_output.mp3"
)
# Significantly faster than standard models
```

**Example - Multi-Speaker Conversations:**

ElevenLabs supports text-to-dialogue for creating conversations with multiple speakers:

```python
# Create text-to-speech model with v3 API
model = AIFactory.create_text_to_speech("elevenlabs", "eleven_v3")

# Define conversation with speaker names
conversation_text = """
Sarah: Hi there! How are you enjoying the new voice technology?
Mike: It's incredible! The voices sound so natural and expressive.
Sarah: I agree. The emotional range is really impressive.
Mike: And the multi-speaker feature makes dialogues seamless.
"""

# Configure speakers with different voice IDs from your account
speaker_configs = [
    {"speaker": "Sarah", "voice": "your-sarah-voice-id"},
    {"speaker": "Mike", "voice": "your-mike-voice-id"}
]

# Generate multi-speaker audio
response = model.generate_multi_speaker_speech(
    text=conversation_text,
    speaker_configs=speaker_configs,
    output_file="conversation.mp3",
    model_id="eleven_v3",
    output_format="mp3_44100_128"
)

print(f"Generated multi-speaker dialogue with {len(speaker_configs)} voices")
```

**Example - Async Multi-Speaker:**

```python
async def create_podcast_dialogue():
    model = AIFactory.create_text_to_speech("elevenlabs", "eleven_v3")

    podcast_script = """
    Host: Welcome back to Tech Talk! Today we're discussing AI voice technology.
    Guest: Thanks for having me. AI voices have come incredibly far.
    Host: What makes modern AI voices so convincing?
    Guest: It's the combination of neural models and extensive training data.
    Host: How do you see this technology evolving?
    Guest: We'll see even more natural emotional expressions and real-time generation.
    """

    speaker_configs = [
        {"speaker": "Host", "voice": "host-voice-id"},
        {"speaker": "Guest", "voice": "guest-voice-id"}
    ]

    response = await model.agenerate_multi_speaker_speech(
        text=podcast_script,
        speaker_configs=speaker_configs,
        output_file="podcast_episode.mp3",
        model_id="eleven_v3",
        settings={
            "stability": 0.7,
            "similarity_boost": 0.8
        }
    )

    return response
```

**Voice Management:**

```python
# Get available voices from your account
voices = model.available_voices

for voice_id, voice_info in voices.items():
    print(f"Voice: {voice_info.name}")
    print(f"  ID: {voice_id}")
    print(f"  Language: {voice_info.language_code}")
    print(f"  Category: {voice_info.category}")
    print()
```

### Speech-to-Text

**Available Models:**
- **speech-to-text-1** - Multilingual speech recognition

**Configuration:**

```python
from esperanto.factory import AIFactory

transcriber = AIFactory.create_speech_to_text(
    "elevenlabs",
    "speech-to-text-1",
    config={
        "timeout": 300.0  # 5 minutes for large files
    }
)
```

**Example - Basic Transcription:**

```python
from esperanto.factory import AIFactory

# Create speech-to-text model
model = AIFactory.create_speech_to_text("elevenlabs", "speech-to-text-1")

# Transcribe audio
response = model.transcribe("audio.mp3")
print(response.text)

# Transcribe from file object
with open("audio.mp3", "rb") as f:
    response = model.transcribe(f)
    print(response.text)
```

**Example - Multilingual Transcription:**

```python
# ElevenLabs automatically detects language
audio_files = {
    "english.mp3": "English content",
    "spanish.mp3": "Spanish content",
    "french.mp3": "French content"
}

for audio_file, description in audio_files.items():
    response = model.transcribe(audio_file)
    print(f"{description}: {response.text}")
    print(f"Detected language: {response.language}\n")
```

**Example - With Context:**

```python
# Provide context for better accuracy
response = model.transcribe(
    "technical_talk.mp3",
    prompt="This is a technical discussion about artificial intelligence and machine learning"
)
print(f"Transcription: {response.text}")
```

**Example - Async Transcription:**

```python
async def transcribe_async():
    model = AIFactory.create_speech_to_text("elevenlabs", "speech-to-text-1")

    response = await model.atranscribe("meeting.wav")
    print(f"Transcription: {response.text}")
    print(f"Language: {response.language}")
```

**Example - Batch Processing:**

```python
import os
from esperanto.factory import AIFactory

model = AIFactory.create_speech_to_text("elevenlabs", "speech-to-text-1")

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
    output_file = transcript["file"].replace(".mp3", ".txt").replace(".wav", ".txt")
    with open(output_file, "w") as f:
        f.write(transcript["text"])
```

## Advanced Features

### Voice Cloning

ElevenLabs allows you to clone voices (requires appropriate subscription):

```python
# Use your cloned voice
response = model.generate_speech(
    text="This uses my cloned voice.",
    voice="your-cloned-voice-id",
    stability=0.75,
    similarity_boost=0.85,
    output_file="cloned_voice.mp3"
)
```

### Emotional Control

Adjust voice characteristics for different emotions:

```python
# Calm and stable narration
calm_response = model.generate_speech(
    text="This is a calm, professional narration.",
    voice="your-voice-id",
    stability=0.9,          # High stability for calm voice
    similarity_boost=0.8,
    output_file="calm.mp3"
)

# Excited and expressive delivery
excited_response = model.generate_speech(
    text="This is incredibly exciting news!",
    voice="your-voice-id",
    stability=0.4,          # Lower stability for variation
    style=0.6,              # More style exaggeration
    output_file="excited.mp3"
)
```

### Audio Quality Control

```python
# Maximum quality for professional production
response = model.generate_speech(
    text="Professional production quality audio.",
    voice="your-voice-id",
    stability=0.75,
    similarity_boost=0.85,
    use_speaker_boost=True,  # Enhanced clarity
    output_file="production.mp3"
)
```

### Long-Form Content

```python
# Generate long-form content efficiently
long_text = """
[Your long article, chapter, or script here]
This can be several paragraphs or even pages of text.
ElevenLabs handles long-form content smoothly.
"""

response = model.generate_speech(
    text=long_text,
    voice="your-voice-id",
    stability=0.7,
    similarity_boost=0.8,
    output_file="long_form.mp3"
)
```

### Real-Time Processing

```python
async def process_audio_stream():
    model = AIFactory.create_speech_to_text("elevenlabs", "speech-to-text-1")

    # Process audio files as they become available
    audio_queue = ["chunk1.wav", "chunk2.wav", "chunk3.wav"]

    for audio_chunk in audio_queue:
        response = await model.atranscribe(audio_chunk)
        print(f"Chunk transcription: {response.text}")

        # Process the transcription immediately
        if "action required" in response.text.lower():
            print("Action item detected!")
```

## Voice Selection Guide

### Finding Voice IDs

1. Visit https://elevenlabs.io/app/voice-library
2. Browse or search for voices
3. Click on a voice to see its ID
4. Use the ID in your code

### Voice Categories

**Professional Voices:**
- Clear articulation
- Neutral tone
- Good for business content

**Narrative Voices:**
- Storytelling quality
- Engaging delivery
- Good for audiobooks, podcasts

**Character Voices:**
- Distinctive personalities
- Expressive delivery
- Good for entertainment content

**Cloned Voices:**
- Custom voice replicas
- Personalized delivery
- Good for brand consistency

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key is correct and active in your ElevenLabs account.

**Quota Exceeded:**
```
Error: Character quota exceeded
```
**Solution:** Check your subscription limits and upgrade if needed.

**Voice Not Found:**
```
Error: Voice ID not found
```
**Solution:** Verify the voice ID exists in your account and is spelled correctly.

**Audio File Too Large:**
```
Error: File size exceeds limit
```
**Solution:** Split large audio files into smaller chunks for transcription.

**Invalid Voice Settings:**
```
Error: Invalid stability value
```
**Solution:** Ensure all voice parameters are in valid ranges (0.0-1.0).

### Best Practices

1. **Voice Selection:** Choose voices that match your content's tone and audience.

2. **Voice Settings:** Start with default settings and adjust based on results.

3. **Stability:** Higher stability (0.7-0.9) for professional content, lower (0.3-0.5) for expressive content.

4. **Similarity Boost:** Higher values (0.8-0.9) maintain voice consistency, lower allows more variation.

5. **Testing:** Test different voices and settings to find the best fit for your use case.

6. **Quota Management:** Monitor your usage to avoid hitting character limits.

7. **Error Handling:** Implement proper error handling for production applications.

## Use Cases

### Audiobook Production

```python
# Generate audiobook chapters with consistent voice
model = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")

chapters = [
    "Chapter 1 content...",
    "Chapter 2 content...",
    "Chapter 3 content..."
]

for i, chapter_text in enumerate(chapters):
    response = model.generate_speech(
        text=chapter_text,
        voice="narrative-voice-id",
        stability=0.7,
        similarity_boost=0.8,
        output_file=f"chapter_{i+1}.mp3"
    )
    print(f"Generated chapter {i+1}")
```

### Podcast Creation

```python
# Create podcast with multiple speakers
model = AIFactory.create_text_to_speech("elevenlabs", "eleven_v3")

podcast_script = """
Host: Welcome to our weekly tech podcast!
Guest: Thanks for having me on the show.
Host: Let's dive into today's topic...
"""

speaker_configs = [
    {"speaker": "Host", "voice": "host-voice-id"},
    {"speaker": "Guest", "voice": "guest-voice-id"}
]

response = model.generate_multi_speaker_speech(
    text=podcast_script,
    speaker_configs=speaker_configs,
    output_file="podcast_episode.mp3"
)
```

### Video Voiceovers

```python
# Generate voiceovers for video content
model = AIFactory.create_text_to_speech("elevenlabs", "eleven_turbo_v2_5")

voiceover_scripts = {
    "intro": "Welcome to our product demonstration...",
    "feature1": "Our first feature enables...",
    "feature2": "The second feature provides...",
    "outro": "Thank you for watching!"
}

for section, script in voiceover_scripts.items():
    response = model.generate_speech(
        text=script,
        voice="professional-voice-id",
        stability=0.8,
        output_file=f"voiceover_{section}.mp3"
    )
```

### Meeting Transcription

```python
# Transcribe meeting recordings
model = AIFactory.create_speech_to_text("elevenlabs", "speech-to-text-1")

meeting_files = ["meeting_part1.mp3", "meeting_part2.mp3"]
full_transcript = []

for meeting_file in meeting_files:
    response = model.transcribe(meeting_file)
    full_transcript.append(response.text)

# Combine and save full transcript
with open("meeting_transcript.txt", "w") as f:
    f.write("\n\n".join(full_transcript))
```

## See Also

- [Text-to-Speech Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/text_to_speech.md)
- [Speech-to-Text Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/speech_to_text.md)
- [OpenAI Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/openai.md)
- [Google Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/google.md)
- [Groq Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/groq.md)
