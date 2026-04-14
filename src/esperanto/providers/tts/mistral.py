"""Mistral Text-to-Speech provider implementation (Voxtral)."""
import base64
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import httpx

from .base import TextToSpeechModel, AudioResponse, Voice, Model

RESPONSE_FORMAT_TO_CONTENT_TYPE = {
    "mp3": "audio/mp3",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class MistralTextToSpeechModel(TextToSpeechModel):
    """Mistral Text-to-Speech provider implementation (Voxtral).

    Supports the Voxtral TTS model with 20 preset voice options.
    """

    DEFAULT_MODEL = "voxtral-mini-tts-2603"
    DEFAULT_VOICE = "neutral_female"
    AVAILABLE_VOICES = [
        "ar_male", "casual_female", "casual_male", "cheerful_female",
        "de_female", "de_male", "es_female", "es_male",
        "fr_female", "fr_male", "hi_female", "hi_male",
        "it_female", "it_male", "neutral_female", "neutral_male",
        "nl_female", "nl_male", "pt_female", "pt_male",
    ]
    PROVIDER = "mistral"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize Mistral TTS provider.

        Args:
            model_name: Name of the model to use (default: voxtral-mini-tts-2603)
            api_key: Mistral API key. If not provided, will try to get from MISTRAL_API_KEY env var
            base_url: Optional base URL for the API
            **kwargs: Additional configuration options
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("MISTRAL_API_KEY"),
            base_url=base_url,
            config=kwargs
        )

        self.base_url = self.base_url or "https://api.mistral.ai/v1"

        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Mistral API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Mistral API error: {error_message}")

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from Mistral TTS.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        voice_metadata = {
            "ar_male": ("MALE", "ar", "Arabic male voice"),
            "casual_female": ("FEMALE", "en", "Casual female voice"),
            "casual_male": ("MALE", "en", "Casual male voice"),
            "cheerful_female": ("FEMALE", "en", "Cheerful female voice"),
            "de_female": ("FEMALE", "de", "German female voice"),
            "de_male": ("MALE", "de", "German male voice"),
            "es_female": ("FEMALE", "es", "Spanish female voice"),
            "es_male": ("MALE", "es", "Spanish male voice"),
            "fr_female": ("FEMALE", "fr", "French female voice"),
            "fr_male": ("MALE", "fr", "French male voice"),
            "hi_female": ("FEMALE", "hi", "Hindi female voice"),
            "hi_male": ("MALE", "hi", "Hindi male voice"),
            "it_female": ("FEMALE", "it", "Italian female voice"),
            "it_male": ("MALE", "it", "Italian male voice"),
            "neutral_female": ("FEMALE", "en", "Neutral female voice"),
            "neutral_male": ("MALE", "en", "Neutral male voice"),
            "nl_female": ("FEMALE", "nl", "Dutch female voice"),
            "nl_male": ("MALE", "nl", "Dutch male voice"),
            "pt_female": ("FEMALE", "pt", "Portuguese female voice"),
            "pt_male": ("MALE", "pt", "Portuguese male voice"),
        }
        return {
            name: Voice(
                name=name,
                id=name,
                gender=gender,
                language_code=lang,
                description=desc,
            )
            for name, (gender, lang, desc) in voice_metadata.items()
        }

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self.PROVIDER

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return self.DEFAULT_MODEL

    def _get_models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="voxtral-mini-tts-2603",
                owned_by="mistralai",
                context_window=None,
            )
        ]

    def generate_speech(
        self,
        text: str,
        voice: str = "neutral_female",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using Mistral's Voxtral TTS API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "neutral_female")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the Mistral API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            response_format = kwargs.pop("response_format", "mp3")

            payload = {
                "model": self.model_name,
                "voice_id": voice,
                "input": text,
                "response_format": response_format,
                **kwargs
            }

            response = self.client.post(
                f"{self.base_url}/audio/speech",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)

            audio_data = base64.b64decode(response.json()["audio_data"])

            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            content_type = RESPONSE_FORMAT_TO_CONTENT_TYPE.get(
                response_format, f"audio/{response_format}"
            )
            return AudioResponse(
                audio_data=audio_data,
                content_type=content_type,
                model=self.model_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str = "neutral_female",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using Mistral's Voxtral TTS API asynchronously.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "neutral_female")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the Mistral API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            response_format = kwargs.pop("response_format", "mp3")

            payload = {
                "model": self.model_name,
                "voice_id": voice,
                "input": text,
                "response_format": response_format,
                **kwargs
            }

            response = await self.async_client.post(
                f"{self.base_url}/audio/speech",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)

            audio_data = base64.b64decode(response.json()["audio_data"])

            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            content_type = RESPONSE_FORMAT_TO_CONTENT_TYPE.get(
                response_format, f"audio/{response_format}"
            )
            return AudioResponse(
                audio_data=audio_data,
                content_type=content_type,
                model=self.model_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e
