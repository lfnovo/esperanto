"""Groq speech-to-text provider."""
import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Union

from groq import AsyncGroq, Groq

from esperanto.providers.stt.base import Model, SpeechToTextModel
from esperanto.types import TranscriptionResponse


@dataclass
class GroqSpeechToTextModel(SpeechToTextModel):
    """Groq speech-to-text model implementation."""

    def __post_init__(self):
        """Initialize Groq client."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()
        
        # Get API key
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found")
        
        # Initialize clients
        config = {
            "api_key": self.api_key,
        }
        if self.base_url:
            config["base_url"] = self.base_url
            
        self.client = Groq(**config)
        self.async_client = AsyncGroq(**config)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "whisper-1"  # Update with actual Groq model name when available

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "groq"


    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        try:
            models = self.client.models.list()
            return [
                Model(
                    id=model_id,  # The model ID is the first item in the tuple
                    owned_by="Groq",  # Groq owns all models in their API
                    context_window=None,  # Audio models don't have context windows
                    type="speech_to_text"
                )
                for model_id, *_ in models  # Unpack the tuple, we only need the ID
                if model_id.startswith("whisper")  # Groq uses OpenAI's Whisper models
            ]
        except Exception:
            # Return empty list if we can't fetch models
            return []

    def _get_api_kwargs(self, language: Optional[str] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
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
        """Transcribe audio to text using Groq's model."""
        kwargs = self._get_api_kwargs(language, prompt)
        
        # Handle file input
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                response = self.client.audio.transcriptions.create(file=f, **kwargs)
        else:
            response = self.client.audio.transcriptions.create(file=audio_file, **kwargs)
            
        return TranscriptionResponse(
            text=response.text,
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
        """Async transcribe audio to text using Groq's model."""
        kwargs = self._get_api_kwargs(language, prompt)
        
        # Handle file input
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                response = await self.async_client.audio.transcriptions.create(file=f, **kwargs)
        else:
            response = await self.async_client.audio.transcriptions.create(file=audio_file, **kwargs)
            
        return TranscriptionResponse(
            text=response.text,
            language=language,
            model=self.get_model_name(),
            provider=self.provider,
        )
