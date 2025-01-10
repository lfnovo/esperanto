"""Ollama embedding model provider."""

import os
from typing import Any, Dict, List

from ollama import AsyncClient, Client

from esperanto.providers.embedding.base import EmbeddingModel, Model


class OllamaEmbeddingModel(EmbeddingModel):
    """Ollama embedding model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set default base URL if not provided
        self.base_url = (
            kwargs.get("base_url")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )

        # Initialize clients
        self.client = Client(host=self.base_url)
        self.async_client = AsyncClient(host=self.base_url)

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        # Start with a copy of the config
        kwargs = self._config.copy()
        # Remove provider-specific kwargs that Ollama doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("base_url", None)
        return kwargs

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.

        Raises:
            ValueError: If text is None or empty.
        """
        if not texts:
            raise ValueError("Texts cannot be empty")

        api_kwargs = {**self._get_api_kwargs(), **kwargs}
        results = []

        for text in texts:
            if text is None:
                raise ValueError("Text cannot be None")
            if not text.strip():
                raise ValueError("Text cannot be empty")

            text = text.replace("\n", " ")
            try:
                response = self.client.embeddings(
                    model=self.get_model_name(), prompt=text, **api_kwargs
                )
                # Convert embeddings to regular floats
                results.append([float(value) for value in response.embedding])
            except Exception as e:
                raise RuntimeError(f"Failed to get embeddings: {str(e)}") from e

        return results

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.

        Raises:
            ValueError: If text is None or empty.
        """
        if not texts:
            raise ValueError("Texts cannot be empty")

        api_kwargs = {**self._get_api_kwargs(), **kwargs}
        results = []

        for text in texts:
            if text is None:
                raise ValueError("Text cannot be None")
            if not text.strip():
                raise ValueError("Text cannot be empty")

            text = text.replace("\n", " ")
            try:
                response = await self.async_client.embeddings(
                    model=self.get_model_name(), prompt=text, **api_kwargs
                )
                # Convert embeddings to regular floats
                results.append([float(value) for value in response.embedding])
            except Exception as e:
                raise RuntimeError(f"Failed to get embeddings: {str(e)}") from e

        return results

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "mxbai-embed-large"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "ollama"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        try:
            models = self.client.list()
            return [
                Model(
                    id=model['name'],
                    owned_by="Ollama",
                    context_window=32768,  # Default context window for Ollama
                    type="embedding"
                )
                for model in models
            ]
        except Exception:
            return []
