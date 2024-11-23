from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from esperanto.base.types import SpeechToTextModel


class OpenAISpeechToTextModel(SpeechToTextModel):
    """OpenAI speech-to-text model implementation (Whisper)."""

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, config)
        self._client = None
        self.language = self.config.get("language")
        self.prompt = self.config.get("prompt")
        self.response_format = self.config.get("response_format", "text")
        self.temperature = self.config.get("temperature", 0)
        self.validate_config()

    @property
    def client(self) -> AsyncOpenAI:
        """Get the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
        """Set the OpenAI client."""
        self._client = value

    @client.deleter
    def client(self) -> None:
        """Delete the OpenAI client."""
        self._client = None

    @property
    def provider(self) -> str:
        return "openai"

    def validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.model_name:
            raise ValueError(
                "model_name must be specified for OpenAI speech-to-text model"
            )
        if self.response_format not in ["json", "text", "srt", "verbose_json", "vtt"]:
            raise ValueError(
                "response_format must be one of: json, text, srt, verbose_json, vtt"
            )

    async def transcribe(self, audio_file: str, **kwargs) -> str:
        """Transcribe audio to text using OpenAI Whisper."""
        with open(audio_file, "rb") as f:
            response = await self.client.audio.transcriptions.create(
                model=self.model_name,
                file=f,
                language=self.language,
                prompt=self.prompt,
                response_format=self.response_format,
                temperature=self.temperature,
                **kwargs,
            )

        return response
