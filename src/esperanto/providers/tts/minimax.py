"""MiniMax text-to-speech provider implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice

RESPONSE_FORMAT_TO_CONTENT_TYPE = {
    "mp3": "audio/mpeg",
    "pcm": "audio/pcm",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcmu_raw": "audio/basic",
    "pcmu_wav": "audio/wav",
    "opus": "audio/opus",
}

MINIMAX_TTS_MODELS = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
)


class MiniMaxTextToSpeechModel(TextToSpeechModel):
    """MiniMax T2A v2 implementation using direct HTTP requests."""

    DEFAULT_MODEL = "speech-2.8-hd"
    DEFAULT_VOICE = "English_Graceful_Lady"
    DEFAULT_BASE_URL = "https://api.minimax.io"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("MINIMAX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MiniMax API key not found. Set MINIMAX_API_KEY environment "
                "variable, or provide it in config."
            )

        self.base_url = (
            self.base_url
            or os.getenv("MINIMAX_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.model_name = self.model_name or self._get_default_model()
        self._voices_cache: Optional[Dict[str, Voice]] = None
        self._create_http_clients()

    @property
    def provider(self) -> str:
        return "minimax"

    def _get_default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _get_models(self) -> List[Model]:
        return [
            Model(
                id=model_id,
                owned_by="MiniMax",
                context_window=None,
                type="text_to_speech",
            )
            for model_id in MINIMAX_TTS_MODELS
        ]

    def _build_url(self, path: str) -> str:
        assert self.base_url is not None
        base_url = self.base_url.rstrip("/")
        # Accept the shared OpenAI-compatible MINIMAX_BASE_URL without adding /v1 twice.
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url}/v1/{path.lstrip('/')}"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _response_json(self, response: httpx.Response) -> Dict[str, Any]:
        if response.status_code >= 400:
            try:
                payload = response.json()
                error_message = (
                    payload.get("base_resp", {}).get("status_msg")
                    or payload.get("error", {}).get("message")
                    or payload.get("message")
                    or f"HTTP {response.status_code}"
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"MiniMax API error: {error_message}")

        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError("MiniMax API returned a non-JSON response") from exc

        # MiniMax reports some API failures in base_resp while still returning HTTP 200.
        base_resp = payload.get("base_resp") or {}
        status_code = base_resp.get("status_code", 0)
        if status_code != 0:
            status_msg = base_resp.get("status_msg") or f"status code {status_code}"
            raise RuntimeError(f"MiniMax API error: {status_msg}")
        return payload

    @property
    def available_voices(self) -> Dict[str, Voice]:
        if self._voices_cache is not None:
            return self._voices_cache

        response = self.client.post(
            self._build_url("get_voice"),
            headers=self._get_headers(),
            json={"voice_type": "all"},
        )
        payload = self._response_json(response)

        voices: Dict[str, Voice] = {}
        categories = (
            ("system_voice", "System voice"),
            ("voice_cloning", "Cloned voice"),
            ("voice_generation", "Generated voice"),
        )
        for field, category in categories:
            for item in payload.get(field) or []:
                voice_id = item.get("voice_id")
                if not voice_id:
                    continue
                descriptions = item.get("description") or []
                description = " ".join(str(value) for value in descriptions).strip()
                voices[voice_id] = Voice(
                    id=voice_id,
                    name=item.get("voice_name") or voice_id,
                    gender="NEUTRAL",
                    description=description or category,
                )

        self._voices_cache = voices
        return voices

    @staticmethod
    def _build_payload(
        model_name: str,
        text: str,
        voice: str,
        kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], str]:
        if kwargs.pop("stream", False):
            raise ValueError(
                "MiniMax streaming TTS is not exposed by generate_speech(); "
                "use non-streaming output."
            )

        output_format = kwargs.pop("output_format", "hex")
        if output_format != "hex":
            raise ValueError(
                "MiniMax output_format must be 'hex' so Esperanto can return "
                "audio bytes."
            )

        response_format = kwargs.pop("response_format", None)
        response_format = str(response_format or "mp3")

        # Map Esperanto's flat TTS options to MiniMax's nested request objects.
        voice_setting = dict(kwargs.pop("voice_setting", {}) or {})
        voice_setting["voice_id"] = voice
        voice_fields = {
            "speed": "speed",
            "volume": "vol",
            "vol": "vol",
            "pitch": "pitch",
            "emotion": "emotion",
            "text_normalization": "text_normalization",
            "latex_read": "latex_read",
        }
        for source, target in voice_fields.items():
            if source in kwargs:
                voice_setting[target] = kwargs.pop(source)

        audio_setting = dict(kwargs.pop("audio_setting", {}) or {})
        audio_setting["format"] = response_format
        audio_fields = {
            "sample_rate": "sample_rate",
            "bitrate": "bitrate",
            "channel": "channel",
            "channels": "channel",
            "force_cbr": "force_cbr",
        }
        for source, target in audio_fields.items():
            if source in kwargs:
                audio_setting[target] = kwargs.pop(source)

        payload: Dict[str, Any] = {
            "model": model_name,
            "text": text,
            "stream": False,
            "output_format": "hex",
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            **kwargs,
        }
        return payload, response_format

    def _build_audio_response(
        self,
        payload: Dict[str, Any],
        text: str,
        voice: str,
        response_format: str,
        output_file: Optional[Union[str, Path]],
    ) -> AudioResponse:
        data = payload.get("data") or {}
        audio_hex = data.get("audio")
        if not audio_hex:
            raise RuntimeError("MiniMax API response did not include audio data")
        try:
            # Non-streaming T2A responses carry the audio bytes as a hex JSON field.
            audio_data = bytes.fromhex(audio_hex)
        except ValueError as exc:
            raise RuntimeError("MiniMax API returned invalid hex audio data") from exc

        if output_file:
            self.save_audio(audio_data, output_file)

        extra_info = payload.get("extra_info") or {}
        duration_ms = extra_info.get("audio_length")
        duration = float(duration_ms) / 1000 if duration_ms is not None else None
        metadata: Dict[str, Any] = {
            "text": text,
            "trace_id": payload.get("trace_id"),
            "extra_info": extra_info,
        }
        if data.get("subtitle_file"):
            metadata["subtitle_file"] = data["subtitle_file"]

        return AudioResponse(
            audio_data=audio_data,
            duration=duration,
            content_type=RESPONSE_FORMAT_TO_CONTENT_TYPE.get(
                response_format, f"audio/{response_format}"
            ),
            model=self.model_name,
            voice=voice,
            provider=self.provider,
            metadata=metadata,
        )

    def generate_speech(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        self.validate_parameters(text, voice, self.model_name)
        model_name = self.model_name or self._get_default_model()
        request_payload, response_format = self._build_payload(
            model_name, text, voice, kwargs
        )
        response = self.client.post(
            self._build_url("t2a_v2"),
            headers=self._get_headers(),
            json=request_payload,
        )
        payload = self._response_json(response)
        return self._build_audio_response(
            payload, text, voice, response_format, output_file
        )

    async def agenerate_speech(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        self.validate_parameters(text, voice, self.model_name)
        model_name = self.model_name or self._get_default_model()
        request_payload, response_format = self._build_payload(
            model_name, text, voice, kwargs
        )
        response = await self.async_client.post(
            self._build_url("t2a_v2"),
            headers=self._get_headers(),
            json=request_payload,
        )
        payload = self._response_json(response)
        return self._build_audio_response(
            payload, text, voice, response_format, output_file
        )
