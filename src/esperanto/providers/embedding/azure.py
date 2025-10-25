"""Azure OpenAI embedding model provider."""

import os
from typing import Any, Dict, List

import httpx

from esperanto.providers.embedding.base import EmbeddingModel, Model


class AzureEmbeddingModel(EmbeddingModel):
    """Azure OpenAI embedding model implementation using direct HTTP."""

    def __init__(self, **kwargs):
        """Initialize Azure embedding provider.

        Args:
            **kwargs: Configuration options including model_name, api_key, base_url, etc.
        """
        # Extract Azure-specific parameters before calling super()
        api_version = kwargs.pop("api_version", None)
        azure_endpoint = kwargs.pop("azure_endpoint", None)

        super().__init__(**kwargs)

        # Resolve configuration with priority: kwargs → config dict → modality env var → generic env var
        self.api_key = (
            kwargs.get("api_key") or
            self._config.get("api_key") or
            os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING") or
            os.getenv("AZURE_OPENAI_API_KEY")
        )

        self.azure_endpoint = (
            kwargs.get("base_url") or
            azure_endpoint or
            self._config.get("azure_endpoint") or
            os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING") or
            os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        self.api_version = (
            api_version or
            self._config.get("api_version") or
            os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING") or
            os.getenv("OPENAI_API_VERSION") or  # Backward compatibility
            os.getenv("AZURE_OPENAI_API_VERSION")
        )

        # deployment_name is model_name for Azure
        self.deployment_name = self.model_name or self._get_default_model()

        # Validate required parameters
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY_EMBEDDING "
                "or AZURE_OPENAI_API_KEY environment variable, or provide in config."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT_EMBEDDING "
                "or AZURE_OPENAI_ENDPOINT environment variable, or provide in config."
            )
        if not self.api_version:
            raise ValueError(
                "Azure OpenAI API version not found. Set AZURE_OPENAI_API_VERSION_EMBEDDING "
                "or AZURE_OPENAI_API_VERSION environment variable, or provide in config."
            )

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Azure API requests."""
        return {
            "api-key": self.api_key,  # Azure uses api-key, not Bearer
            "Content-Type": "application/json",
        }

    def _build_url(self) -> str:
        """Build Azure OpenAI URL with deployment name."""
        # Remove trailing slash from endpoint
        endpoint = self.azure_endpoint.rstrip('/')
        # Azure URL pattern: {endpoint}/openai/deployments/{deployment}/embeddings?api-version={version}
        return f"{endpoint}/openai/deployments/{self.deployment_name}/embeddings?api-version={self.api_version}"

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Azure OpenAI API error: {error_message}")

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
            "model": self.deployment_name,
            **self._get_api_kwargs(),
        }

        # Add any runtime kwargs like dimensions
        if "dimensions" in kwargs:
            payload["dimensions"] = kwargs["dimensions"]

        # Make HTTP request
        url = self._build_url()
        response = self.client.post(
            url,
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
            "model": self.deployment_name,
            **self._get_api_kwargs(),
        }

        # Add any runtime kwargs like dimensions
        if "dimensions" in kwargs:
            payload["dimensions"] = kwargs["dimensions"]

        # Make HTTP request
        url = self._build_url()
        response = await self.async_client.post(
            url,
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
        return "azure"

    def _get_models(self) -> List[Model]:
        """List all available models for this provider.

        Note: Azure doesn't have a models API endpoint - it uses deployments.
        Returns an empty list since model discovery isn't available.
        """
        return []
