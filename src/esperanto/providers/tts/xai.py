"""xAI Text-to-Speech provider implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice

class XAITextToSpeechModel(TextToSpeechModel):
    """xAI Text-to-Speech implementation using direct HTTP.

    Supports xAI TTS deployments with multiple voice options.
    Available voices: eve, ara, rex, sal, leo
    """

    DEFAULT_VOICE = "eve"
    AVAILABLE_VOICES = ["eve", "ara", "rex", "sal", "leo"]
    PROVIDER = "xai"
    DEFAULT_BASE_URL = "https://api.x.ai"
    RESPONSE_FORMAT_TO_CONTENT_TYPE = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mulaw": "audio/basic",
        "alaw": "audio/alaw",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize xAI TTS provider.

        Args:
            api_key: xAI API key. If not provided, will try env vars
            **kwargs: Additional configuration options
        """
        super().__init__(
            api_key=api_key,
            config=kwargs
        )

        # Resolve configuration with priority: config dict → modality env var → generic env var
        self.api_key = (
            self.api_key or
            self._config.get("api_key") or
            os.getenv("XAI_API_KEY_TTS") or
            os.getenv("XAI_API_KEY")
        )

        # Validate required parameters
        if not self.api_key:
            raise ValueError(
                "xAI API key not found. Set XAI_API_KEY_TTS "
                "or XAI_API_KEY environment variable, or provide in config."
            )

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for xAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return headers

    def build_url(self, path: str) -> str:
        """Build full URL for API endpoint."""
        return f"{self.DEFAULT_BASE_URL}/{path}"

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"xAI API error: {error_message}")

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from xAI TTS.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        voices = {
            "eve": Voice(
                name="eve",
                id="eve",
                gender="FEMALE",
                description="Engaging and enthusiastic"
            ),
            "ara": Voice(
                name="ara",
                id="ara",
                gender="FEMALE",
                description="Balanced and conversational"
            ),
            "rex": Voice(
                name="rex",
                id="rex",
                gender="MALE",
                description="Professional and articulate - ideal for business"
            ),
            "sal": Voice(
                name="sal",
                id="sal",
                gender="MALE",
                description="Versatile voice for a wide range of contexts"
            ),
            "leo": Voice(
                name="leo",
                id="leo",
                gender="MALE",
                description="Commanding and decisive - great for instructional content"
            ),
        }
        return voices

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self.PROVIDER

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return ""
    
    def _get_models(self) -> List[Model]:
        """List all available models for this provider.

        Note: xAI doesn't have a models API endpoint.
        Returns an empty list since model discovery isn't available.
        """
        return []

    def generate_speech(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using xAI TTS.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "eve")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the xAI API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            response_format = kwargs.pop("response_format", "mp3")
            language = kwargs.pop("language", "auto")
            url = self.build_url("v1/tts")

            # Prepare request payload
            payload = {
                "voice_id": voice,
                "text": text,
                "language": language,
                "output_format": {
                    "codec": response_format,
                    **kwargs
                }
            }

            # Generate speech
            response = self.client.post(
                url,
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)

            # Get audio data (binary content)
            audio_data = response.content

            # Save to file if specified
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            content_type = self.RESPONSE_FORMAT_TO_CONTENT_TYPE.get(
                response_format, f"audio/{response_format}"
            )
            return AudioResponse(
                audio_data=audio_data,
                content_type=content_type,
                model=self._get_default_model(),
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str = "eve",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Async version of generate_speech.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "eve")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the xAI API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            response_format = kwargs.pop("response_format", "mp3")
            language = kwargs.pop("language", "auto")
            url = self.build_url("v1/tts")

            # Prepare request payload
            payload = {
                "voice_id": voice,
                "text": text,
                "language": language,
                "output_format": {
                    "codec": response_format,
                    **kwargs
                }
            }

            # Generate speech
            response = await self.async_client.post(
                url,
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)

            # Get audio data (binary content)
            audio_data = response.content

            # Save to file if specified
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            content_type = self.RESPONSE_FORMAT_TO_CONTENT_TYPE.get(
                response_format, f"audio/{response_format}"
            )
            return AudioResponse(
                audio_data=audio_data,
                content_type=content_type,
                model=self._get_default_model(),
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e