"""OpenAI-compatible Embedding provider implementation."""

import os
from typing import Any, Dict, List, Optional

import httpx

from esperanto.common_types import Model
from esperanto.utils.logging import logger

from .base import EmbeddingModel


class OpenAICompatibleEmbeddingModel(EmbeddingModel):
    """OpenAI-compatible Embedding provider implementation for custom endpoints.

    This provider extends OpenAI's embedding implementation to work with any OpenAI-compatible
    embedding endpoint, providing graceful fallback for features that may not be supported
    by all endpoints.

    Example:
        >>> from esperanto import AIFactory
        >>> embedder = AIFactory.create_embedding(
        ...     "openai-compatible",
        ...     model_name="nomic-embed-text",
        ...     config={
        ...         "base_url": "http://localhost:1234/v1",
        ...         "timeout": 120
        ...     }
        ... )
        >>> embeddings = embedder.embed(["Hello world", "How are you?"])
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize OpenAI-compatible embedding provider.

        Args:
            model_name: Name of the model to use
            api_key: API key for the provider. If not provided, will try to get from environment
            base_url: Base URL for the OpenAI-compatible endpoint
            config: Additional configuration options including:
                - timeout: Request timeout in seconds (default: 120)
            **kwargs: Additional configuration options
        """
        # Merge config and kwargs
        config = config or {}
        config.update(kwargs)

        # Configuration precedence: Direct params > config > Environment variables
        self.base_url = (
            base_url
            or config.get("base_url")
            or os.getenv("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING")
            or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        )

        self.api_key = (
            api_key
            or config.get("api_key")
            or os.getenv("OPENAI_COMPATIBLE_API_KEY_EMBEDDING")
            or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        )

        # Validation
        if not self.base_url:
            raise ValueError(
                "OpenAI-compatible base URL is required. "
                "Set OPENAI_COMPATIBLE_BASE_URL_EMBEDDING or OPENAI_COMPATIBLE_BASE_URL "
                "environment variable or provide base_url in config."
            )

        # Use a default API key if none is provided (some endpoints don't require authentication)
        if not self.api_key:
            logger.warning(
                "No API key provided for OpenAI-compatible endpoint. Using default 'not-required' value."
            )
            self.api_key = "not-required"

        # Ensure base_url doesn't end with trailing slash for consistency
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

        # Get timeout configuration (default to 120 seconds for embedding operations)
        self.timeout = config.get("timeout", 120.0)

        # Remove base_url, api_key, and timeout from config to avoid duplication
        clean_config = {
            k: v
            for k, v in config.items()
            if k not in ["base_url", "api_key", "timeout"]
        }

        # Initialize attributes for dataclass
        self.model_name = model_name or self._get_default_model()
        self.config = clean_config

        # Call parent's __init__ to set up configuration
        super().__init__(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            config=self.config,
        )

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI-compatible API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses with graceful degradation."""
        if response.status_code >= 400:
            # Log original response for debugging
            logger.debug(f"OpenAI-compatible endpoint error: {response.text}")

            # Try to parse error message from multiple common formats
            try:
                error_data = response.json()
                # Try multiple error message formats
                error_message = (
                    error_data.get("error", {}).get("message")
                    or error_data.get("detail", {}).get("message")  # Some APIs use this
                    or error_data.get("message")  # Direct message field
                    or f"HTTP {response.status_code}"
                )
            except Exception:
                # Fall back to HTTP status code
                error_message = f"HTTP {response.status_code}: {response.text}"

            raise RuntimeError(
                f"OpenAI-compatible embedding endpoint error: {error_message}"
            )

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.

        Note: This attempts to fetch models from the /models endpoint.
        If the endpoint doesn't support this, it will return an empty list.
        """
        try:
            response = self.client.get(
                f"{self.base_url}/models", headers=self._get_headers()
            )
            self._handle_error(response)

            models_data = response.json()
            return [
                Model(
                    id=model["id"],
                    owned_by=model.get("owned_by", "custom"),
                    context_window=model.get("context_window", None),
                    type="embedding",
                )
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            # Log the error but don't fail completely
            logger.info(
                f"Models endpoint not supported by OpenAI-compatible embedding endpoint: {e}"
            )
            return []

    def _get_default_model(self) -> str:
        """Get the default model name.

        For OpenAI-compatible endpoints, we use a generic default
        that users should override with their specific model.
        """
        return "text-embedding-3-small"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai-compatible"

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts using OpenAI-compatible Embedding API.

        Args:
            texts: List of texts to create embeddings for
            **kwargs: Additional parameters to pass to the API

        Returns:
            List of embeddings, one for each input text

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Clean texts using enhanced text cleaning
            texts = [self._clean_text(text) for text in texts]

            # Prepare request payload using OpenAI standard format
            payload = {
                "input": texts,
                "model": self.get_model_name(),
                **{**self._get_api_kwargs(), **kwargs},
            }

            # Generate embeddings
            response = self.client.post(
                f"{self.base_url}/embeddings", headers=self._get_headers(), json=payload
            )
            self._handle_error(response)

            # Parse response
            response_data = response.json()
            return [
                [float(value) for value in data["embedding"]]
                for data in response_data["data"]
            ]

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts using OpenAI-compatible Embedding API asynchronously.

        Args:
            texts: List of texts to create embeddings for
            **kwargs: Additional parameters to pass to the API

        Returns:
            List of embeddings, one for each input text

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Clean texts using enhanced text cleaning
            texts = [self._clean_text(text) for text in texts]

            # Prepare request payload using OpenAI standard format
            payload = {
                "input": texts,
                "model": self.get_model_name(),
                **{**self._get_api_kwargs(), **kwargs},
            }

            # Generate embeddings
            response = await self.async_client.post(
                f"{self.base_url}/embeddings", headers=self._get_headers(), json=payload
            )
            self._handle_error(response)

            # Parse response
            response_data = response.json()
            return [
                [float(value) for value in data["embedding"]]
                for data in response_data["data"]
            ]

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e
