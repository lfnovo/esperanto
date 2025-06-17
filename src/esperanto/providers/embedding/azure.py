"""Azure OpenAI embedding model provider."""
import os
import re
from typing import Any, Dict, List, Optional

from openai import AsyncAzureOpenAI, AzureOpenAI

from esperanto.providers.embedding.base import EmbeddingModel, Model


class AzureEmbeddingModel(EmbeddingModel):
    """Azure OpenAI embedding model implementation."""

    def __init__(
            self,
            api_version: Optional[str] = None,
            **kwargs):
        super().__init__(**kwargs)

        self.api_key = kwargs.get("api_key") or os.getenv(
            "AZURE_OPENAI_API_KEY")

        self.azure_endpoint = kwargs.get("base_url") or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )

        self.api_version = api_version or os.getenv(
            "OPENAI_API_VERSION"
        ) or os.getenv(
            "AZURE_OPENAI_API_VERSION"
        )
        # self.model_name is the Azure deployment name, set by base class constructor

        # Validate required parameters and provide specific error messages
        if not self.api_key:
            raise ValueError("Azure OpenAI API key not found")
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not found")
        if not self.api_version:
            raise ValueError("Azure OpenAI API version not found")
        if not self.model_name:
            raise ValueError(
                "Azure OpenAI deployment name (model_name) not found")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        self.async_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        # Start with a copy of the config
        kwargs = self._config.copy()

        # Remove provider-specific kwargs that OpenAI doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        kwargs.pop("organization", None)

        return kwargs

    # See https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/embeddings?tabs=python-new%2Ccommand-line&pivots=programming-language-python
    def _normalize_text(s, sep_token=" \n "):
        s = re.sub(r'\s+',  ' ', s).strip()
        s = re.sub(r". ,", "", s)
        # remove all instances of multiple spaces
        s = s.replace("..", ".")
        s = s.replace(". .", ".")
        s = s.replace("\n", "")
        s = s.strip()

        return s

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """

        # TODO, normalize? Example provided my MS
        # Clean texts by replacing newlines with spaces
        texts = [self._normalize_text(text) for text in texts]

        # TODO, api_kwargs = self._get_api_kwargs()

        # TODO, error if any text tokens > 8192
        # TODO, error if any texts length > 2048

        response = self.client.embeddings.create(
            input=texts,
            model=self.get_model_name()
        )

        return [embedding.embedding for embedding in response.data]

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # TODO, normalize? Example provided my MS
        # Clean texts by replacing newlines with spaces
        texts = [text.replace("\n", " ") for text in texts]

        # TODO, api_kwargs = self._get_api_kwargs()

        # TODO, error if any text tokens > 8192
        # TODO, error if any texts length > 2048

        response = await self.async_client.embeddings.create(
            input=texts,
            model=self.get_model_name()
        )

        return [embedding.embedding for embedding in response.data]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "text-embedding-3-small"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "azure"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        # Azure doesn't have a models API endpoint - it uses deployments
        # Return empty list since model discovery isn't available
        return []
