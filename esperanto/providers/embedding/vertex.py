"""Google Vertex AI embedding model provider."""
import asyncio
import functools
import os
from typing import Any, Dict, List, Optional

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from esperanto.providers.embedding.base import EmbeddingModel


class VertexEmbeddingModel(EmbeddingModel):
    """Google Vertex AI embedding model implementation."""

    def __init__(self, vertex_project: Optional[str] = None, **kwargs):
        # Extract vertex_project before calling super().__init__
        self.project_id = vertex_project or os.getenv("VERTEX_PROJECT")
        if not self.project_id:
            raise ValueError("Google Cloud project ID not found")
            
        super().__init__(**kwargs)
        
        # Update config with model_name if provided
        if "model_name" in kwargs:
            self._config["model_name"] = kwargs["model_name"]
        
        # Initialize model lazily
        self._model = None

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = {}
        # Remove provider-specific kwargs that Vertex doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("project_id", None)
        return kwargs

    @property
    def model(self) -> TextEmbeddingModel:
        """Get the Vertex AI model instance, initializing it if necessary."""
        if self._model is None:
            self._model = TextEmbeddingModel.from_pretrained(self.get_model_name())
        return self._model

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
        
        # Create embedding inputs
        inputs = [TextEmbeddingInput(text) for text in texts]
        
        # Get embeddings with any additional kwargs
        api_kwargs = {**self._get_api_kwargs(), **kwargs}
        embeddings = self.model.get_embeddings(inputs, **api_kwargs)
        
        return [embedding.values for embedding in embeddings]

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Since Vertex AI's Python SDK doesn't provide async methods,
        # we'll run the sync version in a thread pool
        loop = asyncio.get_event_loop()
        partial_embed = functools.partial(self.embed, texts=texts, **kwargs)
        return await loop.run_in_executor(None, partial_embed)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "textembedding-gecko"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "vertex"
