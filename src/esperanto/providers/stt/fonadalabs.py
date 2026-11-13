"""FonadaLabs Kapture V3 transcription over the native REST API."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
from uuid import uuid4

import httpx

from esperanto.common_types import TranscriptionResponse
from esperanto.providers.stt.base import (
    Model,
    SpeechToTextModel,
    _guess_audio_content_type,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25_000_000


class FonadaLabsAPIError(RuntimeError):
    """Sanitized provider failure carrying a machine-readable HTTP status."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


@dataclass
class FonadaLabsSpeechToTextModel(SpeechToTextModel):
    """Plain-text ASR with one delayed retry for explicit HTTP 429 responses."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("FONADALABS_API_KEY")
        if not self.api_key:
            raise ValueError("Set FONADALABS_API_KEY or provide api_key in config.")
        if self.get_model_name() != "kapture-v3":
            raise ValueError("FonadaLabs supports the kapture-v3 model.")
        self.base_url = (self.base_url or "https://api.fonada.ai/v3/asr").rstrip("/")
        # Configuration forms may contain the documented full endpoint.
        # Keep a single /transcribe suffix when constructing requests.
        if self.base_url.endswith("/transcribe"):
            self.base_url = self.base_url.removesuffix("/transcribe")
        self._create_http_clients()

    @property
    def provider(self) -> str:
        return "fonadalabs"

    def _get_default_model(self) -> str:
        return "kapture-v3"

    def _get_models(self) -> List[Model]:
        return [Model(id="kapture-v3", owned_by="FonadaLabs", type="speech_to_text")]

    def _prepare(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str],
        prompt: Optional[str],
    ):
        if prompt:
            raise ValueError("FonadaLabs does not support transcription prompts.")
        language_id = (
            language
            or self._config.get("language")
            or os.getenv("FONADALABS_LANGUAGE", "en")
        )
        if not isinstance(language_id, str) or not language_id.strip():
            raise ValueError("A non-empty FonadaLabs language code is required.")
        if isinstance(audio_file, str):
            filename = Path(audio_file).name
            with open(audio_file, "rb") as stream:
                data = stream.read(MAX_UPLOAD_BYTES + 1)
        else:
            filename = Path(str(getattr(audio_file, "name", "audio.wav"))).name
            data = audio_file.read(MAX_UPLOAD_BYTES + 1)
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(
                "FonadaLabs requires non-empty audio up to 25 MB; split larger recordings."
            )
        return {"file": (filename, data, _guess_audio_content_type(filename))}, {
            "language_id": language_id.strip()
        }

    @staticmethod
    def _rate_limit_delay(response: httpx.Response) -> float:
        """Wait at least one minute and honor a longer server cooldown."""
        value = response.headers.get("retry-after", "").strip()
        try:
            seconds = float(int(value))
        except ValueError:
            try:
                deadline = parsedate_to_datetime(value)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = 60
        # Do not occupy a worker indefinitely for a long quota suspension.
        if seconds > 3600:
            raise FonadaLabsAPIError(
                429,
                "FonadaLabs requests a cooldown longer than one hour. Retry later after checking account limits.",
            )
        return max(60, seconds)

    def _response(self, response: httpx.Response) -> TranscriptionResponse:
        if not response.is_success:
            messages = {
                401: "Invalid API key",
                403: "Access denied",
                402: "Credits exhausted",
                413: "Audio exceeds 25 MB",
                422: "Unsupported audio or language",
                429: "Rate or credit limit reached",
            }
            message = messages.get(response.status_code, "Transcription request failed")
            logger.warning(
                "FonadaLabs transcription failed: HTTP %s (%s)",
                response.status_code,
                message,
            )
            raise FonadaLabsAPIError(
                response.status_code,
                f"FonadaLabs HTTP {response.status_code}: {message}. Check request status before retrying.",
            )
        try:
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
                payload = payload["result"]
            if not isinstance(payload, dict):
                raise ValueError("Expected object")
            text = next(
                (
                    payload[key]
                    for key in ("text", "transcript", "transcription")
                    if isinstance(payload.get(key), str)
                ),
                None,
            )
            if text is None:
                raise ValueError("Missing text")
        except (ValueError, TypeError):
            raise RuntimeError(
                "FonadaLabs returned an unrecognized transcription response."
            ) from None
        return TranscriptionResponse(
            text=text, model=self.get_model_name(), provider=self.provider
        )

    def _post(self, files, data):
        request_id = uuid4().hex
        started = time.monotonic()
        response = self.client.post(
            f"{self.base_url}/transcribe",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Request-ID": request_id,
            },
            files=files,
            data=data,
        )
        logger.info(
            "FonadaLabs ASR request_id=%s status=%s elapsed_seconds=%.2f",
            request_id,
            response.status_code,
            time.monotonic() - started,
        )
        return response

    async def _apost(self, files, data):
        request_id = uuid4().hex
        started = time.monotonic()
        response = await self.async_client.post(
            f"{self.base_url}/transcribe",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Request-ID": request_id,
            },
            files=files,
            data=data,
        )
        logger.info(
            "FonadaLabs ASR request_id=%s status=%s elapsed_seconds=%.2f",
            request_id,
            response.status_code,
            time.monotonic() - started,
        )
        return response

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        files, data = self._prepare(audio_file, language, prompt)
        try:
            response = self._post(files, data)
            if response.status_code == 429:
                delay = self._rate_limit_delay(response)
                logger.warning(
                    "FonadaLabs rate limit: waiting %s seconds before one retry", delay
                )
                time.sleep(delay)
                response = self._post(files, data)
        except httpx.HTTPError as error:
            logger.warning(
                "FonadaLabs transcription transport failure: %s", type(error).__name__
            )
            raise RuntimeError(
                "FonadaLabs connection or timeout error; request may already be billed. Check before retrying."
            ) from None
        return self._response(response)

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        files, data = await asyncio.to_thread(
            self._prepare, audio_file, language, prompt
        )
        try:
            response = await self._apost(files, data)
            if response.status_code == 429:
                delay = self._rate_limit_delay(response)
                logger.warning(
                    "FonadaLabs rate limit: waiting %s seconds before one retry", delay
                )
                await asyncio.sleep(delay)
                response = await self._apost(files, data)
        except httpx.HTTPError as error:
            logger.warning(
                "FonadaLabs transcription transport failure: %s", type(error).__name__
            )
            raise RuntimeError(
                "FonadaLabs connection or timeout error; request may already be billed. Check before retrying."
            ) from None
        return self._response(response)
