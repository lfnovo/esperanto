"""OpenAI embedding model provider."""
import os
from typing import Any, Dict, List

import httpx

from esperanto.providers.embedding.base import EmbeddingModel, Model


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get API key
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        # Set base URL
        self.base_url = self.base_url or "https://api.openai.com/v1"
        
        # Update config with model_name if provided
        if "model_name" in kwargs:
            self._config["model_name"] = kwargs["model_name"]
        
        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI API error: {error_message}")

    # OpenAI doesn't support advanced features, so we can use the base implementation
    # which will automatically filter them out

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts using enhanced text cleaning
        texts = [self._clean_text(text) for text in texts]

        # Prepare request payload
        payload = {
            "input": texts,
            "model": self.get_model_name(),
            **{**self._get_api_kwargs(), **kwargs}
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        # Parse response
        response_data = response.json()
        return [[float(value) for value in data["embedding"]] for data in response_data["data"]]

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts using enhanced text cleaning
        texts = [self._clean_text(text) for text in texts]

        # Prepare request payload
        payload = {
            "input": texts,
            "model": self.get_model_name(),
            **{**self._get_api_kwargs(), **kwargs}
        }

        # Make HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/embeddings",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        # Parse response
        response_data = response.json()
        return [[float(value) for value in data["embedding"]] for data in response_data["data"]]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "text-embedding-3-small"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        response = self.client.get(
            f"{self.base_url}/models",
            headers=self._get_headers()
        )
        self._handle_error(response)
        
        models_data = response.json()
        return [
            Model(
                id=model["id"],
                owned_by=model.get("owned_by", "openai"),
                context_window=model.get("context_window", None),
                type="embedding"
            )
            for model in models_data["data"]
            if model["id"].startswith("text-embedding")
        ]
