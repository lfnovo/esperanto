"""ElevenLabs speech-to-text provider."""

import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import TranscriptionResponse
from esperanto.providers.stt.base import Model, SpeechToTextModel


@dataclass
class ElevenLabsSpeechToTextModel(SpeechToTextModel):
    """ElevenLabs speech-to-text model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Get API key
        self.api_key = self.api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found")

        # Set base URL
        self.base_url = self.base_url or "https://api.elevenlabs.io/v1"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for ElevenLabs API requests."""
        return {
            "xi-api-key": self.api_key,
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"ElevenLabs API error: {error_message}")

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "scribe_v1"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "elevenlabs"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="scribe_v1",
                owned_by="ElevenLabs",
                context_window=None,  # Audio models don't have context windows
                type="speech_to_text",
            ),
            Model(
                id="scribe_v1_experimental",
                owned_by="ElevenLabs",
                context_window=None,  # Audio models don't have context windows
                type="speech_to_text",
            ),
        ]

    def _get_api_kwargs(
        self, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get kwargs for API calls."""
        kwargs = {
            "model_id": self.get_model_name(),
        }

        # ElevenLabs STT doesn't support language or prompt parameters
        # according to their API docs, so we'll ignore them for now
        
        return kwargs

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text using ElevenLabs' model."""
        kwargs = self._get_api_kwargs(language, prompt)

        # Handle file input
        if isinstance(audio_file, str):
            # For file path, open and send as multipart form data
            with open(audio_file, "rb") as f:
                files = {"file": (audio_file, f, "audio/mpeg")}
                response = self.client.post(
                    f"{self.base_url}/speech-to-text",
                    headers=self._get_headers(),
                    files=files,
                    data=kwargs
                )
        else:
            # For BinaryIO, send the file object directly
            filename = getattr(audio_file, 'name', 'audio.mp3')
            files = {"file": (filename, audio_file, "audio/mpeg")}
            response = self.client.post(
                f"{self.base_url}/speech-to-text",
                headers=self._get_headers(),
                files=files,
                data=kwargs
            )

        self._handle_error(response)
        response_data = response.json()

        return TranscriptionResponse(
            text=response_data["text"],
            language=language,  # ElevenLabs doesn't return detected language
            model=self.get_model_name(),
            provider=self.provider,
        )

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Async transcribe audio to text using ElevenLabs' model."""
        kwargs = self._get_api_kwargs(language, prompt)

        # Handle file input
        if isinstance(audio_file, str):
            # For file path, open and send as multipart form data
            with open(audio_file, "rb") as f:
                files = {"file": (audio_file, f, "audio/mpeg")}
                response = await self.async_client.post(
                    f"{self.base_url}/speech-to-text",
                    headers=self._get_headers(),
                    files=files,
                    data=kwargs
                )
        else:
            # For BinaryIO, send the file object directly
            filename = getattr(audio_file, 'name', 'audio.mp3')
            files = {"file": (filename, audio_file, "audio/mpeg")}
            response = await self.async_client.post(
                f"{self.base_url}/speech-to-text",
                headers=self._get_headers(),
                files=files,
                data=kwargs
            )

        self._handle_error(response)
        response_data = response.json()

        return TranscriptionResponse(
            text=response_data["text"],
            language=language,  # ElevenLabs doesn't return detected language
            model=self.get_model_name(),
            provider=self.provider,
        )