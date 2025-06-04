"""Mistral embedding model provider."""
import os
from typing import Any, Dict, List, Optional

from mistralai import Mistral

from esperanto.providers.embedding.base import EmbeddingModel, Model


class MistralEmbeddingModel(EmbeddingModel):
    """Mistral embedding model implementation."""

    def __post_init__(self):
        """Initialize Mistral client."""
        super().__post_init__()

        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")

        self.client = Mistral(api_key=self.api_key)
        # Mistral Python client does not have a separate async client
        self.async_client = self.client

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = self._config.copy()
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None) # Mistral client handles base_url internally if needed via constructor
        kwargs.pop("organization", None)
        return kwargs

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts."""
        # Mistral client expects 'inputs' as a list of strings
        response = self.client.embeddings.create(
            model=self.get_model_name(),
            inputs=texts,
            **{**self._get_api_kwargs(), **kwargs}
        )
        return [data.embedding for data in response.data]

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously."""
        # Use the async client and assume an 'embeddings.create' or similar async method exists
        # The new client unified sync/async, so self.async_client.embeddings.create should work if client is async-capable
        # If a specific `create_async` is needed, this might require adjustment after testing.
        response = await self.async_client.embeddings.create_async(
            model=self.get_model_name(),
            inputs=texts,
            **{**self._get_api_kwargs(), **kwargs}
        )
        return [data.embedding for data in response.data]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "mistral-embed"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "mistral"

    @property
    def models(self) -> List[Model]:
        """List available Mistral embedding models.
        Note: Mistral's API does not provide a dynamic model listing endpoint for embeddings specifically.
        """
        # Based on current knowledge. Mistral might introduce more embedding models later.
        return [
            Model(
                id="mistral-embed",
                owned_by="mistralai",
                context_window=None, # Typically not specified or relevant for embedding models in this way
                type="embedding"
            )
        ]
