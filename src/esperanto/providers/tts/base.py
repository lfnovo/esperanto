"""Base text-to-speech model interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from esperanto.common_types import Model
from esperanto.common_types.tts import AudioResponse, Voice


@dataclass
class TextToSpeechModel(ABC):
    """Base class for text-to-speech models.

    Attributes:
        model_name: Name of the model to use. If not provided, a default will be used.
        api_key: API key for the provider. If not provided, will try to get from environment.
        base_url: Optional base URL for the API endpoint.
        config: Additional configuration options.
    """

    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    _config: Dict[str, Any] = field(init=False, repr=False)

    # Common SSML tags supported across providers
    COMMON_SSML_TAGS = [
        "speak",
        "break",
        "emphasis",
        "prosody",
        "say-as",
        "voice",
        "audio",
        "p",
        "s",
        "phoneme",
        "sub",
    ]

    def __post_init__(self):
        """Initialize configuration after dataclass initialization."""
        # Initialize config with default values
        self._config = {
            "model_name": self.model_name,
        }

        # Update with any provided config
        if hasattr(self, "config") and self.config:
            self._config.update(self.config)

            # Update instance attributes from config
            for key, value in self._config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    @abstractmethod
    def generate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> AudioResponse:
        """Generate speech from text.

        Args:
            text: The text to convert to speech.
            voice: The voice ID or name to use.
            output_file: Optional path to save the audio file.
            **kwargs: Additional provider-specific parameters.

        Returns:
            AudioResponse containing the audio data and metadata.

        Raises:
            ValueError: If the input parameters are invalid.
            RuntimeError: If speech generation fails.
        """
        pass

    @abstractmethod
    async def agenerate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> AudioResponse:
        """Async version of generate_speech."""
        pass

    @property
    @abstractmethod
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices for this TTS provider.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information.
        """
        pass

    @property
    @abstractmethod
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        pass

    def get_supported_tags(self) -> List[str]:
        """Get list of supported SSML tags for this provider.

        Returns:
            List of supported SSML tag names.
        """
        return self.COMMON_SSML_TAGS

    def validate_parameters(
        self, text: str, voice: str, model: Optional[str] = None
    ) -> None:
        """Validate input parameters before generating speech.

        Args:
            text: Input text to validate
            voice: Voice ID/name to validate
            model: Optional model name to validate

        Raises:
            ValueError: If any parameters are invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        if not voice or not isinstance(voice, str):
            raise ValueError("Voice must be a non-empty string")

        if model and not isinstance(model, str):
            raise ValueError("Model must be a string")

    def save_audio(self, audio_data: bytes, output_file: Union[str, Path]) -> str:
        """Save audio data to a file.

        Args:
            audio_data: Raw audio data in bytes
            output_file: Path to save the audio file

        Returns:
            Absolute path to the saved file

        Raises:
            IOError: If saving the file fails
        """
        output_path = Path(output_file).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return str(output_path)
        except IOError as e:
            raise IOError(f"Failed to save audio file: {str(e)}") from e
