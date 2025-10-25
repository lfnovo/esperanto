"""Base speech-to-text model interface."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Union

from esperanto.common_types import Model, TranscriptionResponse
from esperanto.utils.timeout import TimeoutMixin


@dataclass
class SpeechToTextModel(TimeoutMixin, ABC):
    """Base class for speech-to-text models.

    Attributes:
        model_name: Name of the model to use. If not provided, a default will be used.
        api_key: API key for the provider. If not provided, will try to get from environment.
        base_url: Optional base URL for the API endpoint.
        config: Additional configuration options.
        timeout: HTTP timeout in seconds. If not provided, will use default.
    """

    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    _config: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize configuration after dataclass initialization."""
        # Initialize config with default values
        self._config = {
            "model_name": self.model_name,
        }

        # Add timeout to config if provided as direct parameter
        if self.timeout is not None:
            self._config["timeout"] = self.timeout

        # Update with any provided config
        if hasattr(self, "config") and self.config:
            self._config.update(self.config)

            # Update instance attributes from config
            for key, value in self._config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    @abstractmethod
    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text.

        Args:
            audio_file: Path to audio file or file-like object.
            language: Optional language code (e.g., 'en', 'es'). If not provided,
                     the model will try to detect the language.
            prompt: Optional text to guide the transcription.

        Returns:
            TranscriptionResponse containing the transcribed text and metadata.
        """
        pass

    @abstractmethod
    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Async transcribe audio to text.

        Args:
            audio_file: Path to audio file or file-like object.
            language: Optional language code (e.g., 'en', 'es'). If not provided,
                     the model will try to detect the language.
            prompt: Optional text to guide the transcription.

        Returns:
            TranscriptionResponse containing the transcribed text and metadata.
        """
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider name."""
        pass

    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model name.

        Returns:
            str: The default model name.
        """
        pass

    def _get_provider_type(self) -> str:
        """Return provider type for timeout configuration.

        Returns:
            str: "speech_to_text" for STT providers
        """
        return "speech_to_text"

    def _create_http_clients(self) -> None:
        """Create HTTP clients with configured timeout.

        Call this method in provider's __post_init__ after setting up
        API keys and base URLs.
        """
        import httpx
        timeout = self._get_timeout()
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.

        .. deprecated:: 2.8.0
            The `.models` property is deprecated and will be removed in version 3.0.
            Use `AIFactory.get_provider_models(provider_name)` instead for static
            model discovery without creating provider instances.

        Returns:
            List[Model]: List of available models
        """
        warnings.warn(
            f"The `.models` property is deprecated and will be removed in version 3.0. "
            f"Use AIFactory.get_provider_models('{self.provider}') instead for static "
            f"model discovery without creating provider instances.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._get_models()

    @abstractmethod
    def _get_models(self) -> List[Model]:
        """Internal method to get available models.

        This method should be implemented by providers. The public `.models` property
        will emit a deprecation warning and call this method.

        Returns:
            List[Model]: List of available models
        """
        pass

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

    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from config dictionary."""
        return {k: v for k, v in config.items() if v is not None}
