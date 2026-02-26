"""Zaps.ai embedding model provider."""

import os
from typing import Dict, List, Optional

from esperanto.common_types import Model
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel


class ZapsEmbeddingModel(OpenAIEmbeddingModel):
    """Zaps.ai embedding model implementation using OpenAI-compatible API.

    Zaps.ai acts as a privacy gateway — requests pass through Zaps for PII
    redaction before being forwarded to the upstream provider.
    """

    def __init__(self, **kwargs):
        # Extract api_key and base_url from config dict first
        config = kwargs.get("config", {})
        if config:
            if "api_key" in config:
                kwargs["api_key"] = config["api_key"]
            if "base_url" in config:
                kwargs["base_url"] = config["base_url"]

        # Set Zaps-specific defaults before parent init
        if not kwargs.get("api_key"):
            kwargs["api_key"] = os.getenv("ZAPS_API_KEY")
        if not kwargs.get("base_url"):
            kwargs["base_url"] = os.getenv(
                "ZAPS_BASE_URL", "https://api.zaps.ai/v1"
            )

        if not kwargs.get("api_key"):
            raise ValueError(
                "Zaps API key not found. Set the ZAPS_API_KEY environment variable."
            )

        super().__init__(**kwargs)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "text-embedding-3-small"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "zaps"

    def _get_models(self) -> List[Model]:
        """List all available models for this provider."""
        response = self.client.get(
            f"{self.base_url}/models",
            headers=self._get_headers(),
        )
        self._handle_error(response)

        models_data = response.json()
        return [
            Model(
                id=model["id"],
                owned_by=model.get("owned_by", "zaps"),
                context_window=model.get("context_window", None),
            )
            for model in models_data["data"]
            if "embedding" in model["id"]
        ]
