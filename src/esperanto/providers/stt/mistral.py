"""Mistral speech-to-text provider."""

import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import TranscriptionResponse
from esperanto.providers.stt.base import (
    Model,
    SpeechToTextModel,
    _guess_audio_content_type,
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

        self.base_url = self.base_url or "https://api.mistral.ai/v1"

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

        return TranscriptionResponse(
            text=response_data["text"],
            language=response_data.get("language"),
            model=self.get_model_name(),
            provider=self.provider,
        )

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

        return TranscriptionResponse(
            text=response_data["text"],
            language=response_data.get("language"),
            model=self.get_model_name(),
            provider=self.provider,
        )
