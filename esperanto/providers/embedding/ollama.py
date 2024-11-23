"""Ollama embedding model implementation."""

import os
from typing import Any, Dict, List, Optional

import requests

from esperanto.base.types import EmbeddingModel


class OllamaEmbeddingModel(EmbeddingModel):
    """Ollama embedding model implementation."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        if not model_name:
            raise ValueError("model_name must be specified for Ollama embedding model")
        super().__init__(model_name, config)
        
        # Get base_url from config or environment
        base_url = None
        if self.config and "base_url" in self.config:
            base_url = self.config["base_url"]
        if not base_url:
            base_url = os.environ.get("OLLAMA_API_BASE")
        if not base_url:
            raise ValueError(
                "base_url must be specified in config or OLLAMA_API_BASE environment variable"
            )
        self.base_url = base_url

    @property
    def provider(self) -> str:
        return "ollama"

    def validate_config(self) -> None:
        """Validate Ollama configuration."""
        pass

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        results = []
        for text in texts:
            text = text.replace("\n", " ")
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": [text]},
                **kwargs,
            )
            response_json = response.json()
            if "embeddings" not in response_json or not response_json["embeddings"]:
                raise ValueError("Invalid response from Ollama API")
            results.append(response_json["embeddings"][0])
        return results
