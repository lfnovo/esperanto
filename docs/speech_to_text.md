# Speech-to-Text Models

Speech-to-text models convert audio recordings into written text through automatic speech recognition (ASR). These models can transcribe speech from various audio formats and languages, making them useful for creating transcripts, voice assistants, accessibility tools, and content analysis.

## Supported Providers

- **OpenAI** (Whisper models)
- **Groq** (Whisper models with faster inference)
- **ElevenLabs** (Multilingual speech recognition)

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
