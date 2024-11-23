from typing import Any, Dict, Optional

from groq import AsyncGroq

from esperanto.base.types import SpeechToTextModel


class GroqSpeechToTextModel(SpeechToTextModel):
    """Groq speech-to-text model implementation."""

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
    def client(self) -> AsyncGroq:
        """Get the Groq client."""
        if self._client is None:
            self._client = AsyncGroq()
        return self._client

    @client.setter
    def client(self, value: AsyncGroq) -> None:
        """Set the Groq client."""
        self._client = value

    @client.deleter
    def client(self) -> None:
        """Delete the Groq client."""
        self._client = None

    @property
    def provider(self) -> str:
        return "groq"

    def validate_config(self) -> None:
        """Validate Groq configuration."""
        if not self.model_name:
            raise ValueError(
                "model_name must be specified for Groq speech-to-text model"
            )
        if self.response_format not in ["json", "text"]:
            raise ValueError("response_format must be one of: json, text")

    async def transcribe(self, audio_file: str, **kwargs) -> str:
        """
        Transcribe audio file to text using Groq's speech-to-text API.

        Args:
            audio_file: Path to the audio file to transcribe.

        Returns:
            str: The transcribed text.

        Raises:
            ValueError: If the audio file cannot be read or transcription fails.
        """
        try:
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

        except Exception as e:
            raise ValueError(f"Failed to transcribe audio: {str(e)}")
