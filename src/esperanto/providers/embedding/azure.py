"""Azure OpenAI embedding model provider."""
import os
import re
from typing import List, Optional

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

    # Based on notes from https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/embeddings?tabs=python-new%2Ccommand-line&pivots=programming-language-python
    def _clean_text(self, s: str) -> str:
        """Normalize and clean text for embedding."""
        # Normalize spacing
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'\s+([.,])', r'\1', s)

        # Remove unwanted characters or repeated punctuation
        s = re.sub(r'\.{2,}', '.', s)
        s = re.sub(r'[\n\r]+', ' ', s)

        # Strip again to clean up after replacements
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

        texts = [self._clean_text(text) for text in texts]

        # TODO, error if any text tokens > 8192
        # TODO, error if any texts length > 2048

        # Handle known arguments
        api_kwargs = {}
        if "dimensions" in kwargs:
            api_kwargs["dimensions"] = kwargs["dimensions"]

        response = self.client.embeddings.create(
            input=texts,
            model=self.get_model_name(),
            **api_kwargs
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
        texts = [self._clean_text(text) for text in texts]

        # TODO, error if any text tokens > 8192
        # TODO, error if any texts length > 2048

        # Handle known arguments
        api_kwargs = {}
        if "dimensions" in kwargs:
            api_kwargs["dimensions"] = kwargs["dimensions"]

        response = await self.async_client.embeddings.create(
            input=texts,
            model=self.get_model_name(),
            **api_kwargs
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
