"""OpenRouter embedding model implementation."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from esperanto.common_types import Model
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel


@dataclass
class OpenRouterEmbeddingModel(OpenAIEmbeddingModel):
    """OpenRouter embedding model implementation using OpenAI-compatible API."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        """Initialize OpenRouter-specific configuration."""
        # Call parent's __post_init__ first
        super().__post_init__()

        # Extract from config if provided
        if self.config:
            self.api_key = self.api_key or self.config.get("api_key")
            self.base_url = self.base_url or self.config.get("base_url")

        # Initialize OpenRouter-specific configuration
        self.base_url = self.base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable."
            )

        # Create HTTP clients
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests with required headers."""
        headers = super()._get_headers()
        # Add OpenRouter-specific required headers
        headers.update({
            "HTTP-Referer": "https://github.com/lfnovo/esperanto",
            "X-Title": "Esperanto",
        })
        return headers

    def _handle_error(self, response) -> None:
        """Handle HTTP error responses with detailed OpenRouter logging."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenRouter API error: {error_message}")

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

        # Make HTTP request using OpenRouter format (data parameter with JSON string)
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers=self._get_headers(),
            data=json.dumps(payload)  # Use data= instead of json=
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

        # Make async HTTP request using OpenRouter format (data parameter with JSON string)
        response = await self.async_client.post(
            f"{self.base_url}/embeddings",
            headers=self._get_headers(),
            data=json.dumps(payload)  # Use data= instead of json=
        )
        self._handle_error(response)

        # Parse response
        response_data = response.json()
        return [[float(value) for value in data["embedding"]] for data in response_data["data"]]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "openai/text-embedding-3-small"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openrouter"

    def _get_models(self) -> List[Model]:
        """List all available embedding models for this provider."""
        headers = self._get_headers()

        response = self.client.get(
            f"{self.base_url}/models",
            headers=headers
        )
        self._handle_error(response)

        models_data = response.json()

        # Filter for embedding models only
        embedding_models = []
        for model in models_data["data"]:
            model_id = model["id"]
            # Filter for embedding models (models with IDs containing "embed" or starting with providers known for embeddings)
            if (
                "embed" in model_id.lower() or
                model_id.startswith("text-embedding") or
                model_id.startswith("openai/text-embedding") or
                model_id.startswith("cohere/embed") or
                model_id.startswith("voyage/") or
                model_id.startswith("jina/")
            ):
                embedding_models.append(
                    Model(
                        id=model_id,
                        owned_by=model_id.split("/")[0] if "/" in model_id else "OpenRouter",
                        context_window=model.get("context_length", None),
                    )
                )

        return embedding_models
