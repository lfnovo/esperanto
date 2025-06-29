"""Mistral embedding model provider."""
import os
from typing import Any, Dict, List

import httpx

from esperanto.providers.embedding.base import EmbeddingModel, Model


class MistralEmbeddingModel(EmbeddingModel):
    """Mistral embedding model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        super().__post_init__()

        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")

        # Set base URL
        self.base_url = "https://api.mistral.ai/v1"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Mistral API requests."""
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
            raise RuntimeError(f"Mistral API error: {error_message}")

    # Mistral doesn't support any advanced features, so we can use the base implementation
    # which will automatically filter them out

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts."""
        # Clean texts using enhanced text cleaning
        texts = [self._clean_text(text) for text in texts]
        
        # Prepare request payload - Mistral uses 'input' instead of 'inputs'
        payload = {
            "model": self.get_model_name(),
            "input": texts,
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
        """Create embeddings for the given texts asynchronously."""
        # Clean texts using enhanced text cleaning
        texts = [self._clean_text(text) for text in texts]
        
        # Prepare request payload - Mistral uses 'input' instead of 'inputs'
        payload = {
            "model": self.get_model_name(),
            "input": texts,
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
