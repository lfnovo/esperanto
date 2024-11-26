"""Google Gemini embedding model provider."""
import asyncio
import functools
import os
from typing import Any, Dict, List

import google.generativeai as genai  # type: ignore

from esperanto.providers.embedding.base import EmbeddingModel


class GeminiEmbeddingModel(EmbeddingModel):
    """Google Gemini embedding model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get API key
        self.api_key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found")
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        
        # Update config with model_name if provided
        if "model_name" in kwargs:
            self._config["model_name"] = kwargs["model_name"]

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = {}
        # Remove provider-specific kwargs that Gemini doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        return kwargs

    def _get_model_path(self) -> str:
        """Get the full model path."""
        model_name = self.get_model_name()
        return (
            model_name
            if model_name.startswith("models/")
            else f"models/{model_name}"
        )

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        results = []
        api_kwargs = {**self._get_api_kwargs(), **kwargs}
        model_name = self._get_model_path()
        
        for text in texts:
            text = text.replace("\n", " ")
            result = genai.embed_content(
                model=model_name,
                content=text,
                **api_kwargs
            )
            results.append(result["embedding"])
        
        return results

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Since Gemini's Python SDK doesn't provide async methods,
        # we'll run the sync version in a thread pool
        loop = asyncio.get_event_loop()
        partial_embed = functools.partial(self.embed, texts=texts, **kwargs)
        return await loop.run_in_executor(None, partial_embed)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "embedding-001"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "gemini"
