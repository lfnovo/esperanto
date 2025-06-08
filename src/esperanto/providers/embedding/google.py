"""Google GenAI embedding model provider."""

import os
from typing import Any, Dict, List

import httpx

from esperanto.providers.embedding.base import EmbeddingModel, Model


class GoogleEmbeddingModel(EmbeddingModel):
    """Google GenAI embedding model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get API key
        self.api_key = (
            kwargs.get("api_key")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError("Google API key not found")

        # Set base URL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

        # Update config with model_name if provided
        if "model_name" in kwargs:
            self._config["model_name"] = kwargs["model_name"]

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Google API requests."""
        return {
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Google API error: {error_message}")

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        # Start with a copy of the config
        kwargs = self._config.copy()
        # Remove provider-specific kwargs that Google doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        return kwargs

    def _get_model_path(self) -> str:
        """Get the full model path."""
        model_name = self.get_model_name()
        return (
            model_name if model_name.startswith("models/") else f"models/{model_name}"
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
        model_name = self._get_model_path()

        for text in texts:
            text = text.replace("\n", " ")
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "content": {
                    "parts": [{
                        "text": text
                    }]
                }
            }

            # Make HTTP request
            response = self.client.post(
                f"{self.base_url}/{model_name}:embedContent?key={self.api_key}",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)
            
            response_data = response.json()
            # Convert embeddings to regular floats
            results.append([float(value) for value in response_data["embedding"]["values"]])

        return results

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        results = []
        model_name = self._get_model_path()

        for text in texts:
            text = text.replace("\n", " ")
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "content": {
                    "parts": [{
                        "text": text
                    }]
                }
            }

            # Make async HTTP request
            response = await self.async_client.post(
                f"{self.base_url}/{model_name}:embedContent?key={self.api_key}",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)
            
            response_data = response.json()
            # Convert embeddings to regular floats
            results.append([float(value) for value in response_data["embedding"]["values"]])

        return results

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "text-embedding-004"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "google"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        try:
            response = self.client.get(
                f"{self.base_url}/models?key={self.api_key}",
                headers=self._get_headers()
            )
            self._handle_error(response)
            
            models_data = response.json()
            return [
                Model(
                    id=model["name"].split("/")[-1],
                    owned_by="Google",
                    context_window=model.get("inputTokenLimit"),
                    type="embedding",
                )
                for model in models_data.get("models", [])
                if "embedContent" in model.get("supportedGenerationMethods", [])
            ]
        except Exception:
            # Fallback to known models if API call fails
            return [
                Model(id="text-embedding-004", owned_by="Google", context_window=2048, type="embedding"),
                Model(id="embedding-001", owned_by="Google", context_window=2048, type="embedding"),
            ]
