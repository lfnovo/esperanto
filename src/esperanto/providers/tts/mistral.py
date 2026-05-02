"""Mistral Text-to-Speech provider implementation."""
import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice

RESPONSE_FORMAT_TO_CONTENT_TYPE = {
    "mp3": "audio/mp3",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class MistralTextToSpeechModel(TextToSpeechModel):
    """Mistral Text-to-Speech provider (Voxtral)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable."
            )
        self.base_url = self.base_url or "https://api.mistral.ai/v1"
        self.model_name = self.model_name or self._get_default_model()
        self._voices_cache: Optional[Dict[str, Voice]] = None
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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

    @property
    def provider(self) -> str:
        return "mistral"

    def _get_default_model(self) -> str:
        return "voxtral-mini-tts-2603"

    def _get_models(self) -> List[Model]:
        return [
            Model(id="voxtral-mini-tts-2603", owned_by="mistralai", context_window=None)
        ]

    @property
    def available_voices(self) -> Dict[str, Voice]:
        if self._voices_cache is not None:
            return self._voices_cache
        items = []
        page = 1
        while True:
            response = self.client.get(
                f"{self.base_url}/audio/voices",
                headers=self._get_headers(),
                params={"page": page},
            )
            self._handle_error(response)
            body = response.json()
            items.extend(body.get("items", []))
            if page >= body.get("total_pages", 1):
                break
            page += 1
        self._voices_cache = {
            item["id"]: Voice(
                id=item["id"],
                name=item.get("name", item["id"]),
                gender=item.get("gender", "NEUTRAL"),
                language_code=item["languages"][0] if item.get("languages") else None,
            )
            for item in items
        }
        return self._voices_cache

    def generate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        try:
            self.validate_parameters(text, voice)
            response_format = kwargs.pop("response_format", "mp3")

            payload: Dict[str, Any] = {
                "model": self.model_name or self._get_default_model(),
                "input": text,
                "voice_id": voice,
                "response_format": response_format,
            }

            response = self.client.post(
                f"{self.base_url}/audio/speech",
                headers=self._get_headers(),
                json=payload,
            )
            self._handle_error(response)

            audio_data = base64.b64decode(response.json()["audio_data"])

            if output_file:
                self.save_audio(audio_data, output_file)

            content_type = RESPONSE_FORMAT_TO_CONTENT_TYPE.get(
                str(response_format), f"audio/{response_format}"
            )
            return AudioResponse(
                audio_data=audio_data,
                content_type=content_type,
                model=self.model_name or self._get_default_model(),
                voice=voice,
                provider="mistral",
                metadata={"text": text},
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        try:
            self.validate_parameters(text, voice)
            response_format = kwargs.pop("response_format", "mp3")

            payload: Dict[str, Any] = {
                "model": self.model_name or self._get_default_model(),
                "input": text,
                "voice_id": voice,
                "response_format": response_format,
            }

            response = await self.async_client.post(
                f"{self.base_url}/audio/speech",
                headers=self._get_headers(),
                json=payload,
            )
            self._handle_error(response)

            audio_data = base64.b64decode(response.json()["audio_data"])

            if output_file:
                self.save_audio(audio_data, output_file)

            content_type = RESPONSE_FORMAT_TO_CONTENT_TYPE.get(
                str(response_format), f"audio/{response_format}"
            )
            return AudioResponse(
                audio_data=audio_data,
                content_type=content_type,
                model=self.model_name or self._get_default_model(),
                voice=voice,
                provider="mistral",
                metadata={"text": text},
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e
