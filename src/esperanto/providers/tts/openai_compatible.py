"""OpenAI-compatible Text-to-Speech provider implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from esperanto.common_types import Model
from esperanto.utils.logging import logger

from .base import AudioResponse, Voice
from .openai import OpenAITextToSpeechModel


class OpenAICompatibleTextToSpeechModel(OpenAITextToSpeechModel):
    """OpenAI-compatible Text-to-Speech provider implementation for custom endpoints.

    This provider extends OpenAI's TTS implementation to work with any OpenAI-compatible
    TTS endpoint, providing graceful fallback for features that may not be supported
    by all endpoints.

    Example:
        >>> from esperanto import AIFactory
        >>> tts = AIFactory.create_text_to_speech(
        ...     "openai-compatible",
        ...     model_name="piper-tts",
        ...     config={"base_url": "http://localhost:8000"}
        ... )
        >>> response = tts.generate_speech("Hello world", voice="default")
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize OpenAI-compatible TTS provider.

        Args:
            model_name: Name of the model to use
            api_key: API key for the provider. If not provided, will try to get from environment
            base_url: Base URL for the OpenAI-compatible endpoint
            config: Additional configuration options
            **kwargs: Additional configuration options
        """
        # Merge config and kwargs
        config = config or {}
        config.update(kwargs)

        # Configuration precedence: Direct params > config > Environment variables
        self.base_url = (
            base_url or
            config.get("base_url") or
            os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        )

        self.api_key = (
            api_key or
            config.get("api_key") or
            os.getenv("OPENAI_COMPATIBLE_API_KEY")
        )

        # Validation
        if not self.base_url:
            raise ValueError(
                "OpenAI-compatible base URL is required. "
                "Set OPENAI_COMPATIBLE_BASE_URL environment variable or provide base_url in config."
            )

        # Use a default API key if none is provided (some endpoints don't require authentication)
        if not self.api_key:
            logger.warning("No API key provided for OpenAI-compatible endpoint. Using default 'not-required' value.")
            self.api_key = "not-required"

        # Ensure base_url doesn't end with trailing slash for consistency
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

        # Remove base_url and api_key from config to avoid duplication
        clean_config = {k: v for k, v in config.items() if k not in ['base_url', 'api_key']}

        # Call parent's __init__ to set up HTTP clients and other initialization
        super().__init__(
            model_name=model_name or self._get_default_model(),
            api_key=self.api_key,
            base_url=self.base_url,
            **clean_config
        )

    def _handle_error(self, response) -> None:
        """Handle HTTP error responses with graceful degradation."""
        if response.status_code >= 400:
            # Log original response for debugging
            logger.debug(f"OpenAI-compatible endpoint error: {response.text}")

            # Try to parse error message from multiple common formats
            try:
                error_data = response.json()
                # Try multiple error message formats
                error_message = (
                    error_data.get("error", {}).get("message") or
                    error_data.get("detail", {}).get("message") or  # Some APIs use this
                    error_data.get("message") or  # Direct message field
                    f"HTTP {response.status_code}"
                )
            except Exception:
                # Fall back to HTTP status code
                error_message = f"HTTP {response.status_code}: {response.text}"

            raise RuntimeError(f"OpenAI-compatible endpoint error: {error_message}")

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.

        Note: This attempts to fetch models from the /models endpoint.
        If the endpoint doesn't support this, it will return an empty list.
        """
        try:
            response = self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            )
            self._handle_error(response)

            models_data = response.json()
            return [
                Model(
                    id=model["id"],
                    owned_by=model.get("owned_by", "custom"),
                    context_window=None,  # Audio models don't have context windows
                    type="text_to_speech",
                )
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            # Log the error but don't fail completely
            logger.debug(f"Could not fetch models from OpenAI-compatible endpoint: {e}")
            return []

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from OpenAI-compatible TTS endpoint.

        This method attempts to fetch voices from a custom /audio/voices endpoint.
        If the endpoint doesn't support this, it falls back to a default voice.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        try:
            response = self.client.get(
                f"{self.base_url}/audio/voices",
                headers=self._get_headers()
            )
            self._handle_error(response)

            voices_data = response.json()
            voices = {}

            for voice_dict in voices_data.get("voices", []):
                if voice_dict.get('id'):
                    voice_id = voice_dict['id']
                    voices[voice_id] = Voice(
                        name=voice_dict.get('name', voice_id),
                        id=voice_id,
                        gender=voice_dict.get('gender', 'NEUTRAL'),
                        language_code=voice_dict.get('language_code', 'en-US'),
                        description=voice_dict.get('description', '')
                    )

            return voices

        except Exception as e:
            logger.debug(f"Could not fetch voices from OpenAI-compatible endpoint: {e}")
            # Return a default voice if the endpoint doesn't support voice listing
            return {
                "default": Voice(
                    name="default",
                    id="default",
                    gender="NEUTRAL",
                    language_code="en-US",
                    description="Default voice for OpenAI-compatible endpoint"
                )
            }

    def _get_default_model(self) -> str:
        """Get the default model name.

        For OpenAI-compatible endpoints, we use a generic default
        that users should override with their specific model.
        """
        return "tts-1"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai-compatible"

    def generate_speech(
        self,
        text: str,
        voice: str = "default",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using OpenAI-compatible Text-to-Speech API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "default")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Prepare request payload using OpenAI standard format
            payload = {
                "model": self.model_name,
                "voice": voice,
                "input": text,  # OpenAI standard uses "input", not "text"
                **kwargs
            }

            # Generate speech
            response = self.client.post(
                f"{self.base_url}/audio/speech",
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

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/mp3",
                model=self.model_name,
                voice=voice,
                provider=self.provider,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str = "default",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using OpenAI-compatible Text-to-Speech API asynchronously.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "default")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Prepare request payload using OpenAI standard format
            payload = {
                "model": self.model_name,
                "voice": voice,
                "input": text,  # OpenAI standard uses "input", not "text"
                **kwargs
            }

            # Generate speech
            response = await self.async_client.post(
                f"{self.base_url}/audio/speech",
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

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/mp3",
                model=self.model_name,
                voice=voice,
                provider=self.provider,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e