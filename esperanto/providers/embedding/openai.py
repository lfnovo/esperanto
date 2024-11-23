from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from esperanto.base.types import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.validate_config()
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
        """Set the OpenAI client."""
        self._client = value

    @client.deleter
    def client(self) -> None:
        """Delete the OpenAI client."""
        self._client = None

    @property
    def provider(self) -> str:
        return "openai"

    def validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.model_name:
            raise ValueError("model_name must be specified for OpenAI embedding model")

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        # Preprocess texts
        texts = [text.replace("\n", " ") for text in texts]

        # Get embeddings
        response = await self.client.embeddings.create(
            input=texts, model=self.model_name, **kwargs
        )

        return [data.embedding for data in response.data]
