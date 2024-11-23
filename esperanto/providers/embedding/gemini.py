from typing import Any, Dict, List, Optional

import google.generativeai as genai  # type: ignore

from esperanto.base.types import EmbeddingModel


class GeminiEmbeddingModel(EmbeddingModel):
    """Gemini embedding model implementation."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        if not model_name:
            raise ValueError("model_name must be specified for Gemini embedding model")
        super().__init__(model_name, config)

    @property
    def provider(self) -> str:
        return "gemini"

    def validate_config(self) -> None:
        """Validate Gemini configuration."""
        pass

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        results = []
        for text in texts:
            model_name = (
                self.model_name
                if self.model_name.startswith("models/")
                else f"models/{self.model_name}"
            )
            result = genai.embed_content(model=model_name, content=text, **kwargs)
            results.append(result["embedding"])
        return results
