"""Vertex AI embedding model implementation."""

from typing import Any, Dict, List, Optional

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from esperanto.base.types import EmbeddingModel


class VertexEmbeddingModel(EmbeddingModel):
    """Vertex AI embedding model implementation."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        if not model_name:
            raise ValueError("model_name must be specified for Vertex AI embedding model")
        super().__init__(model_name, config)
        self._model = None

    @property
    def model(self) -> TextEmbeddingModel:
        """Lazy initialization of Vertex AI model."""
        if self._model is None:
            self._model = TextEmbeddingModel.from_pretrained(self.model_name)
        return self._model

    @property
    def provider(self) -> str:
        return "vertex"

    def validate_config(self) -> None:
        """Validate Vertex AI configuration."""
        pass

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        inputs = [TextEmbeddingInput(text) for text in texts]
        embeddings = self.model.get_embeddings(inputs)
        return [embedding.values for embedding in embeddings]
