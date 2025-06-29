"""Voyage AI embedding model provider."""

import os
from typing import Any, Dict, List

import httpx

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

        # Set base URL
        self.base_url = "https://api.voyageai.com/v1"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Voyage API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Voyage API error: {error_message}")

    # Voyage doesn't support advanced features, so we can use the base implementation
    # which will automatically filter them out

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts by replacing newlines with spaces
        texts = [self._clean_text(text) for text in texts]

        # Prepare request payload
        payload = {
            "input": texts,
            "model": self.get_model_name(),
            **self._get_api_kwargs(),
            **kwargs
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)
        
        response_data = response.json()
        return [data["embedding"] for data in response_data["data"]]

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts by replacing newlines with spaces
        texts = [self._clean_text(text) for text in texts]

        # Prepare request payload
        payload = {
            "input": texts,
            "model": self.get_model_name(),
            **self._get_api_kwargs(),
            **kwargs
        }

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/embeddings",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)
        
        response_data = response.json()
        return [data["embedding"] for data in response_data["data"]]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "voyage-3-large"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "voyage"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="voyage-3-large",
                owned_by="Voyage AI",
                context_window=32000,
                type="embedding",
            ),
            Model(
                id="voyage-3.5",
                owned_by="Voyage AI",
                context_window=32000,
                type="embedding",
            ),
            Model(
                id="voyage-3.5-lite",
                owned_by="Voyage AI",
                context_window=32000,
                type="embedding",
            ),
            Model(
                id="voyage-code-3",
                owned_by="Voyage AI",
                context_window=32000,
                type="embedding",
            ),
            Model(
                id="voyage-finance-2",
                owned_by="Voyage AI",
                context_window=32000,
                type="embedding",
            ),
            Model(
                id="voyage-law-2",
                owned_by="Voyage AI",
                context_window=16000,
                type="embedding",
            ),
            Model(
                id="voyage-code-2",
                owned_by="Voyage AI",
                context_window=16000,
                type="embedding",
            ),
        ]
