"""OpenAI speech-to-text provider."""

import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Union

from esperanto.utils.openai_http import AsyncOpenAIHTTPClient, OpenAIHTTPClient

from esperanto.common_types import TranscriptionResponse
from esperanto.providers.stt.base import Model, SpeechToTextModel


@dataclass
class OpenAISpeechToTextModel(SpeechToTextModel):
    """OpenAI speech-to-text model implementation."""

    def __post_init__(self):
        """Initialize OpenAI client."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Get API key
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        # Initialize clients
        config = {
            "api_key": self.api_key,
        }
        if self.base_url:
            config["base_url"] = self.base_url

        self.client = OpenAIHTTPClient(**config)
        self.async_client = AsyncOpenAIHTTPClient(**config)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "whisper-1"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        try:
            models = self.client.models.list()
        except Exception:
            return []

        return [
            Model(
                id=m["id"],
                owned_by=m.get("owned_by"),
                context_window=None,
                type="speech_to_text",
            )
            for m in models
            if m.get("id", "").startswith("whisper")
        ]

    def _get_api_kwargs(
        self, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get kwargs for API calls."""
        kwargs = {
            "model": self.get_model_name(),
        }

        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        return kwargs

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
            with open(audio_file, "rb") as f:
                response = self.client.audio.transcriptions.create(file=f, **kwargs)
        else:
            response = self.client.audio.transcriptions.create(
                file=audio_file, **kwargs
            )
        text = response.get("text", "") if isinstance(response, dict) else getattr(response, "text", "")
        return TranscriptionResponse(
            text=text,
            language=language,
            model=self.get_model_name(),
            provider=self.provider,
        )

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
            with open(audio_file, "rb") as f:
                response = await self.async_client.audio.transcriptions.create(
                    file=f, **kwargs
                )
        else:
            response = await self.async_client.audio.transcriptions.create(
                file=audio_file, **kwargs
            )
        text = response.get("text", "") if isinstance(response, dict) else getattr(response, "text", "")
        return TranscriptionResponse(
            text=text,
            language=language,
            model=self.get_model_name(),
            provider=self.provider,
        )
