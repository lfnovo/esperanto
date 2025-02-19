"""Voyage AI embedding model provider."""

import asyncio
import functools
import os
from typing import Any, Dict, List

import voyageai

from esperanto.providers.embedding.base import EmbeddingModel, Model


class VoyageEmbeddingModel(EmbeddingModel):
    """Voyage AI embedding model implementation."""

    def __init__(self, **kwargs):
        """Initialize the model.

        Args:
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)

        # Get API key
        self.api_key = kwargs.get("api_key") or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key not found")

        # Initialize client
        voyageai.api_key = self.api_key

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        # Start with a copy of the config
        kwargs = self._config.copy()

        # Remove provider-specific kwargs that Voyage doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)

        return kwargs

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts by replacing newlines with spaces
        texts = [text.replace("\n", " ") for text in texts]

        # Get embeddings using the SDK
        response = voyageai.get_embeddings(
            texts,
            model=self.get_model_name(),
            **self._get_api_kwargs(),
            **kwargs,
        )

        return response

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Since the SDK doesn't provide async methods, run in thread pool
        loop = asyncio.get_event_loop()
        partial_embed = functools.partial(self.embed, texts=texts, **kwargs)
        return await loop.run_in_executor(None, partial_embed)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "voyage-large-2"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "voyage"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="voyage-large-2",
                owned_by="Voyage AI",
                context_window=8192,
                type="embedding",
            ),
            Model(
                id="voyage-code-2",
                owned_by="Voyage AI",
                context_window=8192,
                type="embedding",
            ),
        ]
