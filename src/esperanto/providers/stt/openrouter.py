"""OpenRouter Speech-to-Text provider implementation."""

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import (
    Model,
    TranscriptionResponse,
    TranscriptionUsage,
)
from esperanto.providers.stt.base import SpeechToTextModel

# OpenRouter's transcription endpoint takes a base64 payload with an explicit
# codec token (NOT OpenAI's multipart file upload). Map common audio file
# extensions onto the codecs OpenRouter documents as supported.
_EXTENSION_TO_FORMAT = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".mpeg": "mp3",
    ".mpga": "mp3",
    ".flac": "flac",
    ".m4a": "m4a",
    ".mp4": "m4a",
    ".ogg": "ogg",
    ".oga": "ogg",
    ".webm": "webm",
    ".aac": "aac",
}
_DEFAULT_FORMAT = "mp3"


@dataclass
class OpenRouterSpeechToTextModel(SpeechToTextModel):
    """OpenRouter speech-to-text provider.

    Unlike OpenAI's multipart transcription API, OpenRouter's
    ``POST /api/v1/audio/transcriptions`` endpoint accepts a JSON body with a
    base64-encoded ``input_audio`` object and returns ``{"text", "usage"}``.
    This provider builds that request shape while returning Esperanto's standard
    :class:`TranscriptionResponse`.

    Model names follow OpenRouter's ``vendor/model`` convention, e.g.
    ``openai/whisper-1`` or ``openai/whisper-large-v3``.
    """

    DEFAULT_MODEL = "openai/whisper-1"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __post_init__(self):
        """Resolve credentials and initialize HTTP clients."""
        super().__post_init__()

        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable."
            )

        self.base_url = (
            self.base_url or os.getenv("OPENROUTER_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")

        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/lfnovo/esperanto",
            "X-Title": "Esperanto",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get(
                    "message", f"HTTP {response.status_code}"
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenRouter API error: {error_message}")

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return self.DEFAULT_MODEL

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openrouter"

    def _get_models(self) -> List[Model]:
        """List available models.

        OpenRouter's ``/models`` endpoint does not expose a reliable
        modality filter for transcription models, so model discovery is not
        supported for this provider. Returns an empty list.
        """
        return []

    def _detect_format(self, filename: Optional[str]) -> str:
        """Map an audio filename to an OpenRouter codec token."""
        if not filename:
            return _DEFAULT_FORMAT
        return _EXTENSION_TO_FORMAT.get(Path(filename).suffix.lower(), _DEFAULT_FORMAT)

    def _read_audio(
        self, audio_file: Union[str, BinaryIO]
    ) -> tuple[bytes, str]:
        """Read audio bytes and resolve a filename for format detection."""
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                return f.read(), audio_file
        filename = getattr(audio_file, "name", None) or "audio.mp3"
        return audio_file.read(), filename

    def _build_payload(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str],
    ) -> Dict[str, Any]:
        """Build the OpenRouter transcription request payload."""
        audio_bytes, filename = self._read_audio(audio_file)
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "format": self._detect_format(filename),
            },
        }
        if language:
            payload["language"] = language
        return payload

    def _build_response(
        self,
        response_data: Dict[str, Any],
        language: Optional[str],
    ) -> TranscriptionResponse:
        """Map OpenRouter's transcription JSON into a TranscriptionResponse."""
        usage_data = response_data.get("usage") or {}
        seconds = usage_data.get("seconds")
        usage = (
            TranscriptionUsage(
                input_seconds=seconds,
                input_tokens=usage_data.get("input_tokens"),
                output_tokens=usage_data.get("output_tokens"),
                total_tokens=usage_data.get("total_tokens"),
            )
            if usage_data
            else None
        )
        return TranscriptionResponse(
            text=response_data["text"],
            language=response_data.get("language") or language,
            duration=float(seconds) if seconds is not None else None,
            usage=usage,
            model=self.get_model_name(),
            provider=self.provider,
        )

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text using OpenRouter's transcription API.

        Note: OpenRouter's transcription endpoint does not document a ``prompt``
        parameter, so ``prompt`` is accepted for interface parity but not sent.
        """
        payload = self._build_payload(audio_file, language)
        response = self.client.post(
            f"{self.base_url}/audio/transcriptions",
            headers=self._get_headers(),
            json=payload,
        )
        self._handle_error(response)
        return self._build_response(response.json(), language)

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Async transcribe audio to text using OpenRouter's transcription API.

        Note: OpenRouter's transcription endpoint does not document a ``prompt``
        parameter, so ``prompt`` is accepted for interface parity but not sent.
        """
        payload = self._build_payload(audio_file, language)
        response = await self.async_client.post(
            f"{self.base_url}/audio/transcriptions",
            headers=self._get_headers(),
            json=payload,
        )
        self._handle_error(response)
        return self._build_response(response.json(), language)
