"""Mistral speech-to-text provider."""

import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import (
    TranscriptionResponse,
    TranscriptionUsage,
)
from esperanto.providers.stt.base import (
    Model,
    SpeechToTextModel,
    _build_transcription_response,
    _guess_audio_content_type,
)

# Mistral-specific per-segment fields routed through TranscriptionSegment.metadata
# rather than promoted to first-class fields. See ARCHITECTURE.md
# ("Per-item Metadata Escape Hatch").
_MISTRAL_SEGMENT_METADATA_KEYS = (
    "id",
    "confidence",
    "speaker",
    "language",
)


@dataclass
class MistralSpeechToTextModel(SpeechToTextModel):
    """Mistral speech-to-text model implementation."""

    def __post_init__(self):
        super().__post_init__()

        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable."
            )

        self.base_url = (self.base_url or "https://api.mistral.ai/v1").rstrip("/")

        self._create_http_clients()

    @property
    def provider(self) -> str:
        return "mistral"

    def _get_default_model(self) -> str:
        return "voxtral-mini-latest"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get(
                    "message", f"HTTP {response.status_code}"
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Mistral API error: {error_message}")

    def _get_models(self) -> List[Model]:
        return [
            Model(id="voxtral-mini-latest", owned_by="mistralai", context_window=None),
            Model(id="voxtral-small-latest", owned_by="mistralai", context_window=None),
        ]

    def _build_request_data(
        self, language: Optional[str], prompt: Optional[str]
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {"model": self.get_model_name()}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        return data

    @staticmethod
    def _build_usage(usage_data: Optional[Dict[str, Any]]) -> Optional[TranscriptionUsage]:
        """Map Mistral's usage payload into TranscriptionUsage, or return None."""
        if not usage_data:
            return None
        return TranscriptionUsage(
            input_seconds=usage_data.get("prompt_audio_seconds"),
            input_tokens=usage_data.get("prompt_tokens"),
            output_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )

    def _build_response(self, response_data: Dict[str, Any]) -> TranscriptionResponse:
        """Build a TranscriptionResponse from a Mistral Voxtral payload."""
        return _build_transcription_response(
            response_data,
            model=self.get_model_name(),
            provider=self.provider,
            metadata_keys=_MISTRAL_SEGMENT_METADATA_KEYS,
            usage=self._build_usage(response_data.get("usage")),
        )

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text using Mistral's Voxtral model."""
        data = self._build_request_data(language, prompt)

        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                files: Dict[str, Any] = {"file": (audio_file, f, _guess_audio_content_type(audio_file))}
                response = self.client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=self._get_headers(),
                    files=files,
                    data=data,
                )
        else:
            filename = getattr(audio_file, "name", "audio.mp3")
            files = {"file": (filename, audio_file, _guess_audio_content_type(filename))}
            response = self.client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=self._get_headers(),
                files=files,
                data=data,
            )

        self._handle_error(response)
        response_data = response.json()

        return self._build_response(response_data)

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Async transcribe audio to text using Mistral's Voxtral model."""
        data = self._build_request_data(language, prompt)

        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                files: Dict[str, Any] = {"file": (audio_file, f, _guess_audio_content_type(audio_file))}
                response = await self.async_client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=self._get_headers(),
                    files=files,
                    data=data,
                )
        else:
            filename = getattr(audio_file, "name", "audio.mp3")
            files = {"file": (filename, audio_file, _guess_audio_content_type(filename))}
            response = await self.async_client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=self._get_headers(),
                files=files,
                data=data,
            )

        self._handle_error(response)
        response_data = response.json()

        return self._build_response(response_data)
