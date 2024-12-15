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

from .base import TextToSpeechModel, AudioResponse, Voice

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
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize Google Cloud TTS provider.
        
        Args:
            model_name: Type of voice model to use (Standard, WaveNet, Neural2, Studio)
            api_key: Google Cloud API key
            credentials_path: Path to service account credentials JSON file
            **kwargs: Additional configuration options including:
                - language_code: Default language code (e.g., 'en-US')
                - audio_encoding: Output audio encoding (mp3, wav, ogg, etc.)
                - speaking_rate: Speaking rate/speed between 0.25 and 4.0
                - pitch: Speaking pitch between -20.0 and 20.0
                - volume_gain_db: Volume gain between -96.0 and 16.0
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            config=kwargs
        )
        
        # Set up credentials
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(credentials_path).absolute())
        
        # Initialize client
        try:
            client_options = {"api_key": self.api_key} if self.api_key else None
            self.client = texttospeech.TextToSpeechClient(client_options=client_options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google TTS client: {str(e)}") from e
        
        # Set default configuration
        self.language_code = kwargs.get("language_code", self.DEFAULT_LANGUAGE)
        self.audio_encoding = self.AUDIO_ENCODINGS.get(
            kwargs.get("audio_encoding", "mp3"),
            texttospeech.AudioEncoding.MP3
        )
        
        # Voice configuration
        self.speaking_rate = kwargs.get("speaking_rate", 1.0)
        self.pitch = kwargs.get("pitch", 0.0)
        self.volume_gain_db = kwargs.get("volume_gain_db", 0.0)
        
        self._available_voices = None

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from Google Cloud TTS.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        if self._available_voices is None:
            try:
                voices_response = self.client.list_voices()
                self._available_voices = {}
                for voice in voices_response.voices:
                    # Use the first language code as primary
                    language_code = voice.language_codes[0] if voice.language_codes else "en-US"
                    self._available_voices[voice.name] = Voice(
                        name=voice.name,
                        id=voice.name,
                        gender=voice.ssml_gender.name,  # Already normalized as MALE/FEMALE/NEUTRAL
                        language_code=language_code,
                        description=f"{voice.name} - {language_code}"
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to fetch available voices: {str(e)}") from e
        return self._available_voices

    def _create_voice_params(self, voice: str) -> VoiceSelectionParams:
        """Create voice selection parameters.
        
        Args:
            voice: Voice name (e.g., 'en-US-Standard-A')
            
        Returns:
            VoiceSelectionParams for the synthesis request
        """
        # Parse voice name to get language and voice type
        parts = voice.split("-")
        if len(parts) < 4:
            raise ValueError(
                f"Invalid voice format: {voice}. Expected format: [language]-[region]-[model]-[voice]"
            )
        
        language_code = f"{parts[0]}-{parts[1]}"
        
        return VoiceSelectionParams(
            language_code=language_code,
            name=voice
        )

    def _create_audio_config(self, **kwargs) -> AudioConfig:
        """Create audio configuration for synthesis request.
        
        Args:
            **kwargs: Override default audio settings
            
        Returns:
            AudioConfig for the synthesis request
        """
        return AudioConfig(
            audio_encoding=self.audio_encoding,
            speaking_rate=kwargs.get("speaking_rate", self.speaking_rate),
            pitch=kwargs.get("pitch", self.pitch),
            volume_gain_db=kwargs.get("volume_gain_db", self.volume_gain_db)
        )

    def generate_speech(
        self,
        text: str,
        voice: str = "en-US-Neural2-A",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using Google Cloud TTS.

        Args:
            text: Text to convert to speech
            voice: Voice name (e.g., 'en-US-Standard-A')
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the Google Cloud TTS API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Create the synthesis input
            synthesis_input = SynthesisInput(text=text)

            # Create the voice parameters
            voice_params = self._create_voice_params(voice)

            # Create the audio config
            audio_config = self._create_audio_config(**kwargs)

            # Generate the speech
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )

            # Get the audio content
            audio_data = response.audio_content

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
                metadata={
                    "text": text,
                    "language_code": voice_params.language_code,
                    "audio_config": {
                        "speaking_rate": audio_config.speaking_rate,
                        "pitch": audio_config.pitch,
                        "volume_gain_db": audio_config.volume_gain_db
                    }
                }
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str = "en-US-Neural2-A",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Async version of generate_speech.
        
        Note: Google Cloud TTS client doesn't have native async support,
        so this is a wrapper around the sync version.
        """
        return self.generate_speech(text, voice, output_file, **kwargs)

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
