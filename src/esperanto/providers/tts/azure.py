"""Azure OpenAI Text-to-Speech provider implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice


class AzureTextToSpeechModel(TextToSpeechModel):
    """Azure OpenAI Text-to-Speech implementation using direct HTTP.

    Supports Azure OpenAI TTS deployments with multiple voice options.
    Available voices: alloy, echo, fable, onyx, nova, shimmer
    """

    DEFAULT_MODEL = "tts-1"
    DEFAULT_VOICE = "alloy"
    AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    PROVIDER = "azure"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize Azure TTS provider.

        Args:
            model_name: Name of the Azure deployment to use
            api_key: Azure API key. If not provided, will try env vars
            base_url: Azure endpoint URL
            **kwargs: Additional configuration options
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            config=kwargs
        )

        # Resolve configuration with priority: config dict → modality env var → generic env var
        self.api_key = (
            self.api_key or
            self._config.get("api_key") or
            os.getenv("AZURE_OPENAI_API_KEY_TTS") or
            os.getenv("AZURE_OPENAI_API_KEY")
        )

        self.azure_endpoint = (
            self.base_url or
            self._config.get("azure_endpoint") or
            os.getenv("AZURE_OPENAI_ENDPOINT_TTS") or
            os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        self.api_version = (
            self._config.get("api_version") or
            os.getenv("AZURE_OPENAI_API_VERSION_TTS") or
            os.getenv("AZURE_OPENAI_API_VERSION")
        )

        # deployment_name is model_name for Azure
        self.deployment_name = self.model_name or self._get_default_model()

        # Validate required parameters
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY_TTS "
                "or AZURE_OPENAI_API_KEY environment variable, or provide in config."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT_TTS "
                "or AZURE_OPENAI_ENDPOINT environment variable, or provide in config."
            )
        if not self.api_version:
            raise ValueError(
                "Azure OpenAI API version not found. Set AZURE_OPENAI_API_VERSION_TTS "
                "or AZURE_OPENAI_API_VERSION environment variable, or provide in config."
            )

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Azure API requests."""
        return {
            "api-key": self.api_key,  # Azure uses api-key, not Bearer
            "Content-Type": "application/json",
        }

    def _build_url(self, path: str) -> str:
        """Build Azure OpenAI URL with deployment name."""
        # Remove trailing slash from endpoint
        endpoint = self.azure_endpoint.rstrip('/')
        # Azure URL pattern: {endpoint}/openai/deployments/{deployment}/{path}?api-version={version}
        return f"{endpoint}/openai/deployments/{self.deployment_name}/{path}?api-version={self.api_version}"

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Azure OpenAI API error: {error_message}")

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from Azure OpenAI TTS.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        voices = {
            "alloy": Voice(
                name="alloy",
                id="alloy",
                gender="NEUTRAL",
                language_code="en-US",
                description="Neutral and balanced voice"
            ),
            "echo": Voice(
                name="echo",
                id="echo",
                gender="MALE",
                language_code="en-US",
                description="Mature and deep voice"
            ),
            "fable": Voice(
                name="fable",
                id="fable",
                gender="FEMALE",
                language_code="en-US",
                description="Warm and expressive voice"
            ),
            "onyx": Voice(
                name="onyx",
                id="onyx",
                gender="MALE",
                language_code="en-US",
                description="Smooth and authoritative voice"
            ),
            "nova": Voice(
                name="nova",
                id="nova",
                gender="FEMALE",
                language_code="en-US",
                description="Energetic and bright voice"
            ),
            "shimmer": Voice(
                name="shimmer",
                id="shimmer",
                gender="FEMALE",
                language_code="en-US",
                description="Clear and professional voice"
            ),
        }
        return voices

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self.PROVIDER

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return self.DEFAULT_MODEL

    def _get_models(self) -> List[Model]:
        """List all available models for this provider.

        Note: Azure doesn't have a models API endpoint - it uses deployments.
        Returns an empty list since model discovery isn't available.
        """
        return []

    def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using Azure OpenAI TTS.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "alloy")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the Azure API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            url = self._build_url("audio/speech")

            # Prepare request payload
            payload = {
                "model": self.deployment_name,
                "voice": voice,
                "input": text,
                **kwargs
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

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/mp3",
                model=self.deployment_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str = "alloy",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using Azure OpenAI TTS asynchronously.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "alloy")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the Azure API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            url = self._build_url("audio/speech")

            # Prepare request payload
            payload = {
                "model": self.deployment_name,
                "voice": voice,
                "input": text,
                **kwargs
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

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/mp3",
                model=self.deployment_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e
