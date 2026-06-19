"""Deepgram speech-to-text provider."""

import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import TranscriptionResponse, TranscriptionSegment
from esperanto.providers.stt.base import (
    Model,
    SpeechToTextModel,
    _guess_audio_content_type,
)

# Deepgram-specific per-utterance fields routed through TranscriptionSegment.metadata.
# See ARCHITECTURE.md ("Per-item Metadata Escape Hatch").
_DEEPGRAM_UTTERANCE_METADATA_KEYS = (
    "channel",
    "confidence",
    "id",
    "speaker",
)


@dataclass
class DeepgramSpeechToTextModel(SpeechToTextModel):
    """Deepgram speech-to-text provider."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.api_key = self.api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key not found. Set DEEPGRAM_API_KEY environment variable."
            )

        self.base_url = (self.base_url or "https://api.deepgram.com/v1").rstrip("/")

        self._create_http_clients()

    @property
    def provider(self) -> str:
        return "deepgram"

    def _get_default_model(self) -> str:
        return "nova-3"

    def _get_models(self) -> List[Model]:
        return [
            Model(id="nova-3", owned_by="deepgram", context_window=None),
            Model(id="nova-2", owned_by="deepgram", context_window=None),
            Model(id="whisper-large", owned_by="deepgram", context_window=None),
            Model(id="whisper-medium", owned_by="deepgram", context_window=None),
            Model(id="whisper-small", owned_by="deepgram", context_window=None),
            Model(id="whisper-base", owned_by="deepgram", context_window=None),
            Model(id="whisper-tiny", owned_by="deepgram", context_window=None),
        ]

    def _get_headers(self, content_type: str) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": content_type,
        }

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "err_msg",
                    error_data.get("message", f"HTTP {response.status_code}"),
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Deepgram API error: {error_message}")

    def _build_params(self, language: Optional[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.get_model_name(),
            "utterances": "true",
        }
        if language:
            params["language"] = language
        return params

    def _build_response(self, response_data: Dict[str, Any]) -> TranscriptionResponse:
        results = response_data.get("results", {})
        channels = results.get("channels") or []
        alternatives = channels[0].get("alternatives") or [] if channels else []
        text = alternatives[0].get("transcript", "") if alternatives else ""

        metadata = response_data.get("metadata") or {}
        duration_raw = metadata.get("duration")
        duration = float(duration_raw) if duration_raw is not None else None

        raw_utterances = results.get("utterances")
        segments: Optional[List[TranscriptionSegment]] = None
        if raw_utterances:
            segments = [
                TranscriptionSegment(
                    text=u.get("text", ""),
                    start=float(u.get("start", 0.0)),
                    end=float(u.get("end", 0.0)),
                    metadata={
                        key: u[key]
                        for key in _DEEPGRAM_UTTERANCE_METADATA_KEYS
                        if key in u
                    }
                    or None,
                )
                for u in raw_utterances
            ]

        return TranscriptionResponse(
            text=text,
            language=None,
            duration=duration,
            usage=None,
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
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            content_type = _guess_audio_content_type(audio_file)
        else:
            audio_data = audio_file.read()
            filename = getattr(audio_file, "name", None)
            content_type = _guess_audio_content_type(filename)

        params = self._build_params(language)

        response = self.client.post(
            f"{self.base_url}/listen",
            headers=self._get_headers(content_type),
            content=audio_data,
            params=params,
        )
        self._handle_error(response)
        return self._build_response(response.json())

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            content_type = _guess_audio_content_type(audio_file)
        else:
            audio_data = audio_file.read()
            filename = getattr(audio_file, "name", None)
            content_type = _guess_audio_content_type(filename)

        params = self._build_params(language)

        response = await self.async_client.post(
            f"{self.base_url}/listen",
            headers=self._get_headers(content_type),
            content=audio_data,
            params=params,
        )
        self._handle_error(response)
        return self._build_response(response.json())
