"""OpenAI embedding model provider."""
import os
from typing import Any, Dict, List

from openai import AsyncOpenAI, OpenAI

from esperanto.providers.embedding.base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get API key
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        # Update config with model_name if provided
        if "model_name" in kwargs:
            self._config["model_name"] = kwargs["model_name"]
        
        # Initialize clients
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
        )

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = {}
        # Remove provider-specific kwargs that OpenAI doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        kwargs.pop("organization", None)
        return kwargs

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts by replacing newlines with spaces
        texts = [text.replace("\n", " ") for text in texts]

        # Get embeddings
        response = self.client.embeddings.create(
            input=texts,
            model=self.get_model_name(),
            **{**self._get_api_kwargs(), **kwargs}
        )

        return [data.embedding for data in response.data]

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts by replacing newlines with spaces
        texts = [text.replace("\n", " ") for text in texts]

        # Get embeddings
        response = await self.async_client.embeddings.create(
            input=texts,
            model=self.get_model_name(),
            **{**self._get_api_kwargs(), **kwargs}
        )

        return [data.embedding for data in response.data]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "text-embedding-3-small"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai"
