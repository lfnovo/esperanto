"""ElevenLabs Text-to-Speech provider implementation."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs, ElevenLabs

from .base import AudioResponse, Model, TextToSpeechModel, Voice

# Load environment variables
load_dotenv()

class ElevenLabsTextToSpeechModel(TextToSpeechModel):
    """ElevenLabs Text-to-Speech provider implementation.
    
    Supports multiple models including:
    - eleven_multilingual_v2: Multilingual model
    - eleven_monolingual_v1: English-only model
    - eleven_turbo_v2: Faster, lower-quality model
    """
    
    DEFAULT_MODEL = "eleven_multilingual_v2"
    DEFAULT_VOICE = "Aria"  # One of their default voices
    PROVIDER = "elevenlabs"
    
    # Default voice settings
    DEFAULT_VOICE_SETTINGS = {
        "stability": 0.5,  # Range 0-1
        "similarity_boost": 0.75,  # Range 0-1
        "style": 0.0,  # Range 0-1
        "use_speaker_boost": True
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize ElevenLabs TTS provider.
        
        Args:
            model_name: Name of the model to use
            api_key: ElevenLabs API key. If not provided, will try to get from ELEVENLABS_API_KEY env var
            base_url: Optional base URL for the API
            **kwargs: Additional configuration options including voice_settings
        """
        api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key not provided. Set ELEVENLABS_API_KEY environment variable or pass api_key parameter.")

        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url,
            config=kwargs
        )
        
        self.voice_settings = {
            **self.DEFAULT_VOICE_SETTINGS,
            **(kwargs.get("voice_settings", {}) or {})
        }
        
        # Initialize client with API key
        self.client = ElevenLabs(
            api_key=api_key,
        )
        self.async_client = AsyncElevenLabs(
            api_key=api_key,
        )
        
        # Cache available voices
        self._available_voices = None

    def _get_sync_client(self):
        return self.client

    async def _get_async_client(self):
        return self.async_client

    def generate_speech(self, text: str, voice: str, output_file: Optional[Union[str, Path]] = None) -> AudioResponse:
        """Generate speech synchronously."""
        client = self._get_sync_client()

        # Convert text to speech using voice_id as a path parameter
        response = client.text_to_speech.convert(
            voice_id=voice,  # voice_id is a path parameter
            text=text,
            model_id=self.model_name
        )

        # Collect all bytes from the iterator
        audio_bytes = b''.join(response)

        response_audio = AudioResponse(
            audio_data=audio_bytes,
            content_type="audio/mp3",
            model=self.model_name,
            voice=voice,
            provider="elevenlabs"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return response_audio

    async def agenerate_speech(self, text: str, voice: str, output_file: Optional[Union[str, Path]] = None) -> AudioResponse:
        """Generate speech asynchronously."""
        client = await self._get_async_client()

        # Convert text to speech using voice_id as a path parameter
        response = client.text_to_speech.convert(
            voice_id=voice,  # voice_id is a path parameter
            text=text,
            model_id=self.model_name
        )

        # Collect all bytes from the async iterator
        audio_bytes = b''
        async for chunk in response:
            audio_bytes += chunk

        response_audio = AudioResponse(
            audio_data=audio_bytes,
            content_type="audio/mp3",
            model=self.model_name,
            voice=voice,
            provider="elevenlabs"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return response_audio

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices."""
        client = self._get_sync_client()
        response = client.voices.get_all()
        
        voices = {}
        for voice_data in response.voices:
            voices[voice_data.voice_id] = Voice(
                name=voice_data.name,
                id=voice_data.voice_id,
                gender=voice_data.labels.get("gender", "unknown").upper(),
                language_code=voice_data.labels.get("language", "en"),
                description=voice_data.description,
                preview_url=voice_data.preview_url
            )
        return voices

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return []  # For now, return empty list as requested

    def get_supported_tags(self) -> List[str]:
        """Get list of supported SSML tags.
        
        ElevenLabs has limited SSML support compared to other providers.
        """
        return ["speak", "break", "emphasis", "prosody"]
