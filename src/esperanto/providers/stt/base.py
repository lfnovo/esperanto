"""Base speech-to-text model interface."""

import mimetypes
import pathlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Union

from esperanto.common_types import (
    Model,
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionUsage,
)
from esperanto.utils.connect import HttpConnectionMixin

_AUDIO_CONTAINER_EXTENSIONS = {
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".m4a": "audio/mp4",
}

# Whisper model-family per-segment fields surfaced via TranscriptionSegment.metadata
# rather than promoted to first-class fields. Reused by OpenAI, Groq (inherited),
# and Azure providers. See ARCHITECTURE.md ("Per-item Metadata Escape Hatch").
_WHISPER_SEGMENT_METADATA_KEYS = (
    "id",
    "seek",
    "tokens",
    "temperature",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
)


def _guess_audio_content_type(filename: Optional[str]) -> str:
    """Guess audio MIME type from filename, falling back to audio/mpeg.

    Returns audio/mpeg as a safe default when filename is None or empty
    (e.g., a BinaryIO whose ``.name`` attribute is explicitly None).
    """
    if not filename:
        return "audio/mpeg"
    ext = pathlib.Path(filename).suffix.lower()
    if ext in _AUDIO_CONTAINER_EXTENSIONS:
        return _AUDIO_CONTAINER_EXTENSIONS[ext]
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type and mime_type.startswith("audio/"):
        return mime_type
    return "audio/mpeg"


def _build_transcription_response(
    response_data: Dict[str, Any],
    *,
    model: str,
    provider: str,
    metadata_keys: Sequence[str],
    usage: Optional[TranscriptionUsage] = None,
    language_fallback: Optional[str] = None,
) -> TranscriptionResponse:
    """Map a verbose transcription payload into a :class:`TranscriptionResponse`.

    Handles the common shape returned by Whisper-family providers (OpenAI,
    Groq, Azure) and Mistral Voxtral. Each provider only needs to supply its
    own ``metadata_keys`` tuple and (for Mistral) a pre-built ``usage`` object.

    - ``segments`` is mapped into a list of :class:`TranscriptionSegment` with
      provider-specific per-segment extras routed through ``segment.metadata``
      per ARCHITECTURE.md ("Per-item Metadata Escape Hatch").
    - ``duration`` is cast to float when present, ``None`` otherwise.
    - ``language`` falls back to ``language_fallback`` (typically the
      caller-supplied input language) when the provider doesn't echo a
      detected language.
    - ``usage`` is passed through verbatim; providers without an STT usage
      block simply leave it ``None``.
    """
    raw_segments = response_data.get("segments") or []
    segments: Optional[List[TranscriptionSegment]] = None
    if raw_segments:
        segments = [
            TranscriptionSegment(
                text=segment.get("text", ""),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                metadata={
                    key: segment[key]
                    for key in metadata_keys
                    if key in segment
                }
                or None,
            )
            for segment in raw_segments
        ]

    duration_raw = response_data.get("duration")
    duration = float(duration_raw) if duration_raw is not None else None

    return TranscriptionResponse(
        text=response_data["text"],
        language=response_data.get("language") or language_fallback,
        duration=duration,
        usage=usage,
        model=model,
        provider=provider,
        segments=segments,
    )


@dataclass
class SpeechToTextModel(HttpConnectionMixin, ABC):
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
