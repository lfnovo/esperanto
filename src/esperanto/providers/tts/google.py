"""Google Cloud Text-to-Speech provider implementation."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from google.cloud import texttospeech_v1 as texttospeech
from google.cloud.texttospeech_v1.types import (
    AudioConfig,
    SynthesisInput,
    VoiceSelectionParams,
)

from .base import TextToSpeechModel, AudioResponse, Voice, Model

class GoogleTextToSpeechModel(TextToSpeechModel):
    """Google Cloud Text-to-Speech provider implementation.
    
    Supports multiple voice models and languages.
    Default model uses standard voices, but can be configured to use:
    - Standard (default)
    - WaveNet (higher quality)
    - Neural2
    - Studio (premium voices)
    """
    
    DEFAULT_MODEL = "neural2"
    DEFAULT_LANGUAGE = "en-US"
    DEFAULT_VOICE = "en-US-Neural2-A"
    PROVIDER = "google"
    
    # Audio encoding options
    AUDIO_ENCODINGS = {
        "mp3": texttospeech.AudioEncoding.MP3,
        "wav": texttospeech.AudioEncoding.LINEAR16,
        "ogg": texttospeech.AudioEncoding.OGG_OPUS,
        "mulaw": texttospeech.AudioEncoding.MULAW,
        "alaw": texttospeech.AudioEncoding.ALAW
    }
    
    def __init__(
        self,
        model_name: str = "neural2",
        base_url: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize Google Cloud TTS model.

        Args:
            model_name: Model name to use
            base_url: Base URL for the API
            **kwargs: Additional arguments passed to the provider
        """
        super().__init__(model_name=model_name)
        self.credentials = kwargs.get('credentials')
        self.client = None
        self.async_client = None

    def _get_sync_client(self):
        """Get the synchronous client."""
        if not self.client:
            self.client = texttospeech.TextToSpeechClient(credentials=self.credentials)
        return self.client

    async def _get_async_client(self):
        """Get the asynchronous client."""
        if not self.async_client:
            self.async_client = texttospeech.TextToSpeechAsyncClient(credentials=self.credentials)
        return self.async_client

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices."""
        client = self._get_sync_client()
        voices = {}

        response = client.list_voices()
        for voice in response.voices:
            voices[voice.name] = Voice(
                name=voice.name,
                id=voice.name,
                gender=voice.ssml_gender,
                language_code=voice.language_codes[0],
                description=f"{voice.name} - {voice.language_codes[0]}"
            )

        return voices

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        models_list = texttospeech.TextToSpeechClient.list_voices()
        return [
            Model(
                id=model.name.split('/')[-1],
                owned_by="Google",
                context_window=None,  # Audio models don't have context windows
                type="text_to_speech"
            )
            for model in models_list.voices
            if "generateSpeech" in ["generateSpeech"]  # Assuming this is the correct method
        ]

    def generate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text.

        Args:
            text: Text to convert to speech
            voice: Voice ID to use
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
        """
        client = self._get_sync_client()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=voice[:5],  # e.g., "en-US" from "en-US-Neural2-A"
            name=voice,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            **kwargs
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )

        response_audio = AudioResponse(
            audio_data=response.audio_content,
            content_type="audio/mp3",
            model=self.model_name,
            voice=voice,
            provider="google"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.audio_content)

        return response_audio

    async def agenerate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text asynchronously.

        Args:
            text: Text to convert to speech
            voice: Voice ID to use
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
        """
        client = await self._get_async_client()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=voice[:5],  # e.g., "en-US" from "en-US-Neural2-A"
            name=voice,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            **kwargs
        )

        response = await client.synthesize_speech(
            request={
                "input": synthesis_input,
                "voice": voice_params,
                "audio_config": audio_config
            }
        )

        response_audio = AudioResponse(
            audio_data=response.audio_content,
            content_type="audio/mp3",
            model=self.model_name,
            voice=voice,
            provider="google"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.audio_content)

        return response_audio

    def get_supported_tags(self) -> List[str]:
        """Get list of supported SSML tags.
        
        Google Cloud TTS supports a rich set of SSML tags.
        """
        return [
            "speak", "break", "emphasis", "prosody", "say-as",
            "voice", "audio", "p", "s", "phoneme", "sub",
            "mark", "desc", "parallel", "seq", "media"
        ]

    def list_voices(self, language_code: Optional[str] = None) -> Dict[str, Any]:
        """List available voices.
        
        Args:
            language_code: Optional language code to filter voices
            
        Returns:
            Dictionary of available voices and their properties
        """
        try:
            response = self.client.list_voices(language_code=language_code)
            return {
                voice.name: {
                    "language_codes": voice.language_codes,
                    "name": voice.name,
                    "ssml_gender": voice.ssml_gender.name,
                    "natural_sample_rate_hertz": voice.natural_sample_rate_hertz
                }
                for voice in response.voices
            }
        except Exception as e:
            raise RuntimeError(f"Failed to list voices: {str(e)}") from e
