"""CAMB AI Text-to-Speech provider implementation."""

import asyncio
import concurrent.futures
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice

_CAMB_TTS_RESULT_URL = "https://client.camb.ai/apis/tts-result/{run_id}"


def _parse_voice_id(voice: str) -> int:
    """Convert string voice ID to int for the CAMB API."""
    try:
        return int(voice)
    except (ValueError, TypeError):
        raise ValueError(
            f"CAMB voice ID must be numeric, got: {voice!r}"
        )


def _run_async(coro):
    """Run an async coroutine from any context (sync or async)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class CambTextToSpeechModel(TextToSpeechModel):
    """CAMB AI Text-to-Speech provider implementation.

    Supports multiple models including:
    - mars-pro: High-quality TTS
    - mars-flash: Fast TTS
    - mars-instruct: Instructable TTS with user instructions
    """

    DEFAULT_MODEL = "mars-pro"
    DEFAULT_VOICE = "147320"
    PROVIDER = "camb"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        api_key = api_key or os.getenv("CAMB_API_KEY")
        if not api_key:
            raise ValueError(
                "CAMB API key not provided. Set CAMB_API_KEY environment "
                "variable or pass api_key parameter."
            )

        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url,
            config=kwargs,
        )

        self.language = self._config.get("language", "en-us")
        self._available_voices: Optional[Dict[str, Voice]] = None
        self._client_instance = None
        self._create_http_clients()

    def _create_camb_client(self):
        """Create a new AsyncCambAI client instance."""
        try:
            from camb.client import AsyncCambAI
        except ImportError as exc:
            raise ImportError(
                "The 'camb' package is required to use CAMB TTS provider. "
                "Install it with: pip install camb-sdk"
            ) from exc
        return AsyncCambAI(
            api_key=self.api_key, timeout=self.timeout or 60.0
        )

    def _get_async_client(self):
        """Get a lazily-initialized AsyncCambAI client."""
        if self._client_instance is None:
            self._client_instance = self._create_camb_client()
        return self._client_instance

    async def _stream_tts(self, text: str, voice: str, **kwargs) -> bytes:
        """Stream TTS audio chunks and return concatenated bytes."""
        from camb import StreamTtsOutputConfiguration

        client = self._get_async_client()
        language = kwargs.get("language", self.language)

        tts_kwargs = {
            "text": text,
            "language": language,
            "voice_id": _parse_voice_id(voice),
            "speech_model": self.model_name,
            "output_configuration": StreamTtsOutputConfiguration(format="mp3"),
        }

        user_instructions = kwargs.get("user_instructions")
        if user_instructions and self.model_name == "mars-instruct":
            tts_kwargs["user_instructions"] = user_instructions

        chunks = []
        async for chunk in client.text_to_speech.tts(**tts_kwargs):
            chunks.append(chunk)

        audio_bytes = b"".join(chunks)
        if not audio_bytes:
            raise RuntimeError("No audio data received from CAMB API")
        return audio_bytes

    async def _translated_tts(self, text: str, voice: str, **kwargs) -> bytes:
        """Generate translated TTS using CAMB's translation API."""
        client = self._get_async_client()
        source_language = kwargs["source_language"]
        target_language = kwargs["target_language"]

        tts_kwargs = {
            "text": text,
            "voice_id": _parse_voice_id(voice),
            "source_language": source_language,
            "target_language": target_language,
        }
        formality = kwargs.get("formality")
        if formality:
            tts_kwargs["formality"] = formality

        result = await client.translated_tts.create_translated_tts(**tts_kwargs)

        # Poll for completion using configurable timeout
        timeout = self.timeout or 120.0
        poll_interval = 2.0
        max_attempts = int(timeout / poll_interval)
        for _ in range(max_attempts):
            status = await client.translated_tts.get_translated_tts_task_status(
                result.task_id, run_id=None
            )
            if hasattr(status, "status"):
                if status.status in ("completed", "SUCCESS"):
                    break
                if status.status in ("failed", "FAILED", "error"):
                    raise RuntimeError(
                        f"Translated TTS failed: {getattr(status, 'error', 'Unknown error')}"
                    )
            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(
                f"Translated TTS task did not complete within {timeout}s"
            )

        run_id = getattr(status, "run_id", None)
        if not run_id:
            raise RuntimeError("Translated TTS completed but no run_id returned")

        url = _CAMB_TTS_RESULT_URL.format(run_id=run_id)
        async with httpx.AsyncClient(timeout=self.timeout or 60.0) as http:
            resp = await http.get(
                url, headers={"x-api-key": self.api_key or ""}
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch translated TTS audio: HTTP {resp.status_code}"
                )
            return resp.content

    def generate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> AudioResponse:
        """Generate speech synchronously."""
        self.validate_parameters(text, voice)
        audio_bytes = _run_async(self._do_generate(text, voice, **kwargs))
        return self._build_response(audio_bytes, voice, output_file)

    async def agenerate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> AudioResponse:
        """Generate speech asynchronously."""
        self.validate_parameters(text, voice)
        audio_bytes = await self._do_generate(text, voice, **kwargs)
        return self._build_response(audio_bytes, voice, output_file)

    async def _do_generate(self, text: str, voice: str, **kwargs) -> bytes:
        """Route to streaming TTS or translated TTS based on kwargs."""
        if "target_language" in kwargs and "source_language" in kwargs:
            return await self._translated_tts(text, voice, **kwargs)
        return await self._stream_tts(text, voice, **kwargs)

    def _build_response(
        self,
        audio_bytes: bytes,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
    ) -> AudioResponse:
        """Build AudioResponse and optionally save to file."""
        response = AudioResponse(
            audio_data=audio_bytes,
            content_type="audio/mp3",
            model=self.model_name,
            voice=voice,
            provider="camb",
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return response

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from CAMB AI."""
        if self._available_voices is not None:
            return self._available_voices

        self._available_voices = _run_async(self._fetch_voices())
        return self._available_voices

    async def _fetch_voices(self) -> Dict[str, Voice]:
        """Fetch voices from the CAMB API."""
        try:
            # Use a fresh client to avoid event loop issues when called
            # from a background thread via _run_async
            client = self._create_camb_client()
            voice_list = await client.voice_cloning.list_voices()
        except Exception:
            return {}

        voices = {}
        for v in voice_list:
            # Voice list items can be dicts or Voice objects
            if isinstance(v, dict):
                voice_id = str(v.get("id", ""))
                voice_name = v.get("voice_name", v.get("name", "Unknown"))
                gender_code = v.get("gender", 0)
                language = v.get("language")
                age = v.get("age")
            else:
                voice_id = str(getattr(v, "id", ""))
                voice_name = getattr(v, "voice_name", getattr(v, "name", "Unknown"))
                gender_code = getattr(v, "gender", 0)
                language = getattr(v, "language", None)
                age = getattr(v, "age", None)

            gender = {0: "NEUTRAL", 1: "MALE", 2: "FEMALE", 9: "NEUTRAL"}.get(
                gender_code, "NEUTRAL"
            )

            voices[voice_id] = Voice(
                name=voice_name,
                id=voice_id,
                gender=gender,
                language_code=str(language) if language else None,
                description=None,
                age=str(age) if age is not None else None,
            )

        return voices

    def _get_models(self) -> List[Model]:
        """List available CAMB TTS models."""
        return [
            Model(id="mars-pro", owned_by="camb"),
            Model(id="mars-flash", owned_by="camb"),
            Model(id="mars-instruct", owned_by="camb"),
        ]
