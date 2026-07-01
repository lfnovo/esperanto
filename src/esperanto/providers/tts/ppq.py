"""PayPerQ (PPQ) text-to-speech provider implementation."""

import os
from typing import Any, Dict, Optional

from .openai_compatible import OpenAICompatibleTextToSpeechModel

DEFAULT_BASE_URL = "https://api.ppq.ai/v1"


class PPQTextToSpeechModel(OpenAICompatibleTextToSpeechModel):
    """PayPerQ text-to-speech provider.

    PayPerQ (https://ppq.ai) exposes OpenAI-compatible speech synthesis at
    ``https://api.ppq.ai/v1/audio/speech`` (Deepgram Aura and ElevenLabs voices).
    Authenticates with ``PPQ_API_KEY`` (override the endpoint with ``PPQ_BASE_URL``).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        config = config or {}
        api_key = api_key or config.get("api_key") or os.getenv("PPQ_API_KEY")
        if not api_key:
            raise ValueError(
                "PayPerQ API key not found. "
                "Set the PPQ_API_KEY environment variable or provide api_key in config."
            )
        base_url = (
            base_url
            or config.get("base_url")
            or os.getenv("PPQ_BASE_URL")
            or DEFAULT_BASE_URL
        )
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            config=config,
            **kwargs,
        )

    def _get_default_model(self) -> str:
        """Get the default speech model name."""
        return "deepgram_aura_2"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "ppq"
