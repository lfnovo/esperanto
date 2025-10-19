"""OpenAI-compatible Speech-to-Text provider implementation."""

import os
import mimetypes
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import Model, TranscriptionResponse
from esperanto.utils.logging import logger

from .base import SpeechToTextModel


class OpenAICompatibleSpeechToTextModel(SpeechToTextModel):
    """OpenAI-compatible Speech-to-Text provider implementation for custom endpoints.

    This provider extends OpenAI's STT implementation to work with any OpenAI-compatible
    STT endpoint, providing graceful fallback for features that may not be supported
    by all endpoints.

    Note: This provider inherits from the base SpeechToTextModel class and manually
    initializes HTTP clients, unlike the TTS provider which inherits from OpenAI's
    implementation to reuse existing functionality.

    Example:
        >>> from esperanto import AIFactory
        >>> stt = AIFactory.create_speech_to_text(
        ...     "openai-compatible",
        ...     model_name="faster-whisper",
        ...     config={
        ...         "base_url": "http://localhost:8000",
        ...         "timeout": 600  # 10 minutes for large files
        ...     }
        ... )
        >>> response = stt.transcribe("audio.mp3")
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize OpenAI-compatible STT provider.

        Args:
            model_name: Name of the model to use
            api_key: API key for the provider. If not provided, will try to get from environment
            base_url: Base URL for the OpenAI-compatible endpoint
            config: Additional configuration options including:
                - timeout: Request timeout in seconds (default: 300)
            **kwargs: Additional configuration options
        """
        # Merge config and kwargs
        config = config or {}
        config.update(kwargs)

        # Configuration precedence: Direct params > config > Environment variables
        self.base_url = (
            base_url or
            config.get("base_url") or
            os.getenv("OPENAI_COMPATIBLE_BASE_URL_STT") or
            os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        )

        self.api_key = (
            api_key or
            config.get("api_key") or
            os.getenv("OPENAI_COMPATIBLE_API_KEY_STT") or
            os.getenv("OPENAI_COMPATIBLE_API_KEY")
        )

        # Validation
        if not self.base_url:
            raise ValueError(
                "OpenAI-compatible base URL is required. "
                "Set OPENAI_COMPATIBLE_BASE_URL_STT or OPENAI_COMPATIBLE_BASE_URL "
                "environment variable or provide base_url in config."
            )

        # Use a default API key if none is provided (some endpoints don't require authentication)
        if not self.api_key:
            self.api_key = "not-required"

        # Ensure base_url doesn't end with trailing slash for consistency
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

        # Get timeout configuration (default to 300 seconds for STT operations)
        self.timeout = config.get("timeout", 300.0)

        # Remove base_url, api_key, and timeout from config to avoid duplication
        clean_config = {k: v for k, v in config.items() if k not in ['base_url', 'api_key', 'timeout']}

        # Initialize attributes for dataclass
        self.model_name = model_name or self._get_default_model()
        self.config = clean_config

        # Initialize configuration
        self._config = {
            "model_name": self.model_name,
        }
        if self.config:
            self._config.update(self.config)

        # Initialize HTTP clients with configurable timeout
        # STT operations can take much longer than typical API calls
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI-compatible API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def _handle_error(self, response: httpx.Response) -> None:
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

            raise RuntimeError(f"OpenAI-compatible STT endpoint error: {error_message}")

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
                    type="speech_to_text",
                )
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            # Log the error but don't fail completely
            logger.info(f"Models endpoint not supported by OpenAI-compatible STT endpoint: {e}")
            return []

    def _get_default_model(self) -> str:
        """Get the default model name.

        For OpenAI-compatible endpoints, we use a generic default
        that users should override with their specific model.
        """
        return "whisper-1"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai-compatible"

    def _get_api_kwargs(
        self, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get kwargs for API calls."""
        kwargs = {
            "model": self.get_model_name(),
        }

        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        return kwargs

    def _get_audio_mime_type(self, filename: str) -> str:
        """Detect MIME type for audio file.

        Args:
            filename: Path or name of the audio file

        Returns:
            MIME type string, defaults to 'audio/mpeg' if detection fails
        """
        # Get MIME type from file extension
        mime_type, _ = mimetypes.guess_type(filename)

        # If detection fails or it's not an audio type, default to audio/mpeg
        if not mime_type or not mime_type.startswith('audio/'):
            # Check for common audio extensions
            ext = Path(filename).suffix.lower()
            audio_mime_types = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.m4a': 'audio/mp4',
                '.aac': 'audio/aac',
                '.ogg': 'audio/ogg',
                '.flac': 'audio/flac',
                '.webm': 'audio/webm'
            }
            mime_type = audio_mime_types.get(ext, 'audio/mpeg')

        return mime_type

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text using OpenAI-compatible Speech-to-Text API.

        Args:
            audio_file: Path to audio file or file-like object
            language: Optional language code (e.g., 'en', 'es')
            prompt: Optional text to guide the transcription

        Returns:
            TranscriptionResponse containing the transcribed text and metadata

        Raises:
            RuntimeError: If transcription fails
        """
        try:
            kwargs = self._get_api_kwargs(language, prompt)

            # Handle file input
            if isinstance(audio_file, str):
                # For file path, open and send as multipart form data
                mime_type = self._get_audio_mime_type(audio_file)
                with open(audio_file, "rb") as f:
                    files = {"file": (audio_file, f, mime_type)}
                    response = self.client.post(
                        f"{self.base_url}/audio/transcriptions",
                        headers=self._get_headers(),
                        files=files,
                        data=kwargs
                    )
            else:
                # For BinaryIO, send the file object directly
                filename = getattr(audio_file, 'name', 'audio.mp3')
                mime_type = self._get_audio_mime_type(filename)
                files = {"file": (filename, audio_file, mime_type)}
                response = self.client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=self._get_headers(),
                    files=files,
                    data=kwargs
                )

            self._handle_error(response)
            response_data = response.json()

            return TranscriptionResponse(
                text=response_data["text"],
                language=language,  # OpenAI-compatible endpoints may not return detected language
                model=self.get_model_name(),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}") from e

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text using OpenAI-compatible Speech-to-Text API asynchronously.

        Args:
            audio_file: Path to audio file or file-like object
            language: Optional language code (e.g., 'en', 'es')
            prompt: Optional text to guide the transcription

        Returns:
            TranscriptionResponse containing the transcribed text and metadata

        Raises:
            RuntimeError: If transcription fails
        """
        try:
            kwargs = self._get_api_kwargs(language, prompt)

            # Handle file input
            if isinstance(audio_file, str):
                # For file path, open and send as multipart form data
                mime_type = self._get_audio_mime_type(audio_file)
                with open(audio_file, "rb") as f:
                    files = {"file": (audio_file, f, mime_type)}
                    response = await self.async_client.post(
                        f"{self.base_url}/audio/transcriptions",
                        headers=self._get_headers(),
                        files=files,
                        data=kwargs
                    )
            else:
                # For BinaryIO, send the file object directly
                filename = getattr(audio_file, 'name', 'audio.mp3')
                mime_type = self._get_audio_mime_type(filename)
                files = {"file": (filename, audio_file, mime_type)}
                response = await self.async_client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=self._get_headers(),
                    files=files,
                    data=kwargs
                )

            self._handle_error(response)
            response_data = response.json()

            return TranscriptionResponse(
                text=response_data["text"],
                language=language,  # OpenAI-compatible endpoints may not return detected language
                model=self.get_model_name(),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}") from e

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            str: The model name.
        """
        # First try to get from config
        model_name = self._config.get("model_name")
        if model_name:
            return model_name

        # If not in config, use default
        return self._get_default_model()