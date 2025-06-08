"""Groq speech-to-text provider."""

import os
from dataclasses import dataclass
from typing import List

from esperanto.providers.stt.base import Model
from esperanto.providers.stt.openai import OpenAISpeechToTextModel


@dataclass
class GroqSpeechToTextModel(OpenAISpeechToTextModel):
    """Groq speech-to-text model implementation using OpenAI-compatible API."""

    def __post_init__(self):
        """Initialize HTTP clients with Groq configuration."""
        # Set Groq-specific API key and base URL before calling parent
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found")
        
        # Set Groq's OpenAI-compatible base URL
        self.base_url = self.base_url or "https://api.groq.com/openai/v1"
        
        # Call parent's post_init which will initialize HTTP clients
        super().__post_init__()

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "whisper-large-v3"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "groq"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        try:
            response = self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            )
            self._handle_error(response)
            
            models_data = response.json()
            return [
                Model(
                    id=model["id"],
                    owned_by="Groq",
                    context_window=None,  # Audio models don't have context windows
                    type="speech_to_text",
                )
                for model in models_data["data"]
                if model["id"].startswith("whisper")
            ]
        except Exception:
            # Return empty list if we can't fetch models
            return []