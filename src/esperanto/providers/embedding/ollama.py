"""Ollama embedding model provider."""

import json
import os
from typing import Any, Dict, List

import httpx

from esperanto.providers.embedding.base import EmbeddingModel, Model


class OllamaEmbeddingModel(EmbeddingModel):
    """Ollama embedding model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set default base URL if not provided
        self.base_url = (
            kwargs.get("base_url")
            or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")
            or "http://localhost:11434"
        )

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Ollama API requests."""
        return {
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Ollama API error: {error_message}")

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        # Use base class implementation which handles filtering of unsupported features
        return super()._get_api_kwargs()

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

        results = []

        for text in texts:
            if text is None:
                raise ValueError("Text cannot be None")
            if not text.strip():
                raise ValueError("Text cannot be empty")

            text = self._clean_text(text)
            
            # Prepare request payload
            payload = {
                "model": self.get_model_name(),
                "prompt": text,
                **self._get_api_kwargs(),
                **kwargs
            }

            try:
                # Make HTTP request
                response = self.client.post(
                    f"{self.base_url}/api/embeddings",
                    headers=self._get_headers(),
                    json=payload
                )
                self._handle_error(response)
                
                response_data = response.json()
                # Convert embeddings to regular floats
                results.append([float(value) for value in response_data["embedding"]])
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

        results = []

        for text in texts:
            if text is None:
                raise ValueError("Text cannot be None")
            if not text.strip():
                raise ValueError("Text cannot be empty")

            text = self._clean_text(text)
            
            # Prepare request payload
            payload = {
                "model": self.get_model_name(),
                "prompt": text,
                **self._get_api_kwargs(),
                **kwargs
            }

            try:
                # Make async HTTP request
                response = await self.async_client.post(
                    f"{self.base_url}/api/embeddings",
                    headers=self._get_headers(),
                    json=payload
                )
                self._handle_error(response)
                
                response_data = response.json()
                # Convert embeddings to regular floats
                results.append([float(value) for value in response_data["embedding"]])
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
            response = self.client.get(
                f"{self.base_url}/api/tags",
                headers=self._get_headers()
            )
            self._handle_error(response)
            
            models_data = response.json()
            return [
                Model(
                    id=model["name"],
                    owned_by="Ollama",
                    context_window=32768,  # Default context window for Ollama
                    type="embedding"
                )
                for model in models_data.get("models", [])
            ]
        except Exception:
            return []
