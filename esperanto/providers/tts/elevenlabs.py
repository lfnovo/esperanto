"""ElevenLabs Text-to-Speech provider implementation."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs, ElevenLabs

from .base import TextToSpeechModel, AudioResponse, Voice

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

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from ElevenLabs.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        voices = {}
        all_voices = self.client.voices.get_all()
        
        # GetVoicesResponse has a voices attribute containing the list of voices
        for voice in all_voices.voices:
            voices[voice.voice_id] = Voice(
                name=voice.name,
                id=voice.voice_id,
                gender=voice.labels.get("gender", "UNKNOWN"),
                language_code="en-US",  # ElevenLabs primarily supports English
                description=voice.labels.get("description", ""),
                accent=voice.labels.get("accent"),
                age=voice.labels.get("age"),
                use_case=voice.labels.get("use_case"),
                preview_url=voice.preview_url
            )
        return voices

    def _get_voice_info(self, voice: str):
        """Get voice info from available voices.

        Args:
            voice: Voice name to look up

        Returns:
            Voice info if found, None otherwise
        """
        voice = voice or self.DEFAULT_VOICE
        return self.available_voices.get(voice)

    def generate_speech(
        self,
        text: str,
        voice: str = "Adam",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using ElevenLabs TTS.

        Args:
            text: Text to convert to speech
            voice: Voice ID or name to use
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the ElevenLabs API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Extract settings from kwargs
            voice_settings = {
                **self.voice_settings,
                **(kwargs.get("voice_settings", {}) or {})
            }

            # Generate speech - returns a generator
            audio_generator = self.client.generate(
                text=text,
                voice=voice,
                model=self.model_name,
                voice_settings=voice_settings,
                **kwargs
            )
            
            # Read audio data
            audio_data = b"".join(chunk for chunk in audio_generator)
            
            # Save to file if specified
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/mp3",
                model=self.model_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text, "voice_settings": voice_settings}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str = "Adam",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text asynchronously using ElevenLabs TTS.

        Args:
            text: Text to convert to speech
            voice: Voice ID or name to use
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the ElevenLabs API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Extract settings from kwargs
            voice_settings = {
                **self.voice_settings,
                **(kwargs.get("voice_settings", {}) or {})
            }

            # Generate speech using text_to_speech
            audio_generator = self.async_client.text_to_speech.convert(
                text=text,
                voice_id=voice,  # ElevenLabs async client uses voice_id
                model_id=self.model_name,
                output_format="mp3_44100_128",
                voice_settings=voice_settings,
                **kwargs
            )
            
            # Read audio data
            audio_data = b""
            async for chunk in audio_generator:
                audio_data += chunk
            
            # Save to file if specified
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/mp3",
                model=self.model_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text, "voice_settings": voice_settings}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    def get_supported_tags(self) -> List[str]:
        """Get list of supported SSML tags.
        
        ElevenLabs has limited SSML support compared to other providers.
        """
        return ["speak", "break", "emphasis", "prosody"]
