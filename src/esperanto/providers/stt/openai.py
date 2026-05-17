"""OpenAI speech-to-text provider."""

import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import (
    TranscriptionResponse,
    TranscriptionSegment,
)
from esperanto.providers.stt.base import (
    Model,
    SpeechToTextModel,
    _guess_audio_content_type,
)

# Whisper-specific per-segment fields that are surfaced via TranscriptionSegment.metadata
# rather than promoted to first-class fields. See ARCHITECTURE.md
# ("Per-item Metadata Escape Hatch").
_WHISPER_SEGMENT_METADATA_KEYS = (
    "id",
    "seek",
    "tokens",
    "temperature",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
)


@dataclass
class OpenAISpeechToTextModel(SpeechToTextModel):
    """OpenAI speech-to-text model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Get API key
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        # Set base URL
        self.base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        # Organization is optional for STT models
        if hasattr(self, 'organization') and self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI API error: {error_message}")

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "whisper-1"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai"

    def _get_models(self) -> List[Model]:
        """List all available models for this provider."""
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
                    owned_by=model.get("owned_by", "openai"),
                    context_window=None,  # Audio models don't have context windows
                )
                for model in models_data["data"]
                if model["id"].startswith("whisper")
            ]
        except Exception:
            # Handle the case when the API key is not valid for model listing
            return []

    def _get_api_kwargs(
        self, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get kwargs for API calls.

        Always requests ``verbose_json`` so that segments and duration are returned,
        per Esperanto's Hot-Swap-First Defaults principle. Users don't have to know
        the per-provider response_format quirk to get consistent output.
        """
        kwargs: Dict[str, Any] = {
            "model": self.get_model_name(),
            "response_format": "verbose_json",
        }

        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        return kwargs

    def _build_response(
        self,
        response_data: Dict[str, Any],
        language: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Build a TranscriptionResponse from a Whisper ``verbose_json`` payload."""
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
                        for key in _WHISPER_SEGMENT_METADATA_KEYS
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
            language=response_data.get("language") or language,
            duration=duration,
            model=self.get_model_name(),
            provider=self.provider,
            segments=segments,
        )

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text using OpenAI's Whisper model."""
        kwargs = self._get_api_kwargs(language, prompt)

        # Handle file input
        if isinstance(audio_file, str):
            # For file path, open and send as multipart form data
            with open(audio_file, "rb") as f:
                files: Dict[str, Any] = {"file": (audio_file, f, _guess_audio_content_type(audio_file))}
                response = self.client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=self._get_headers(),
                    files=files,
                    data=kwargs
                )
        else:
            # For BinaryIO, send the file object directly
            filename = getattr(audio_file, 'name', 'audio.mp3')
            files = {"file": (filename, audio_file, _guess_audio_content_type(filename))}
            response = self.client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=self._get_headers(),
                files=files,
                data=kwargs
            )

        self._handle_error(response)
        response_data = response.json()

        return self._build_response(response_data, language=language)

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Async transcribe audio to text using OpenAI's Whisper model."""
        kwargs = self._get_api_kwargs(language, prompt)

        # Handle file input
        if isinstance(audio_file, str):
            # For file path, open and send as multipart form data
            with open(audio_file, "rb") as f:
                files: Dict[str, Any] = {"file": (audio_file, f, _guess_audio_content_type(audio_file))}
                response = await self.async_client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=self._get_headers(),
                    files=files,
                    data=kwargs
                )
        else:
            # For BinaryIO, send the file object directly
            filename = getattr(audio_file, 'name', 'audio.mp3')
            files = {"file": (filename, audio_file, _guess_audio_content_type(filename))}
            response = await self.async_client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=self._get_headers(),
                files=files,
                data=kwargs
            )

        self._handle_error(response)
        response_data = response.json()

        return self._build_response(response_data, language=language)
