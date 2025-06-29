"""Google Vertex AI embedding model provider."""
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import httpx

from esperanto.providers.embedding.base import EmbeddingModel, Model


class VertexEmbeddingModel(EmbeddingModel):
    """Google Vertex AI embedding model implementation."""

    def __init__(self, vertex_project: Optional[str] = None, vertex_location: Optional[str] = None, **kwargs):
        # Extract vertex_project before calling super().__init__
        self.project_id = vertex_project or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = vertex_location or os.getenv("VERTEX_LOCATION", "us-central1")
        
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID not found. Please set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT environment variable."
            )
            
        super().__init__(**kwargs)
        
        # Set base URL for Vertex AI
        self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1"
        
        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)
        
        # Cache for access token
        self._access_token = None
        self._token_expiry = 0
        
        # Update config with model_name if provided
        if "model_name" in kwargs:
            self._config["model_name"] = kwargs["model_name"]

    def _get_access_token(self) -> str:
        """Get OAuth 2.0 access token for Google Cloud APIs."""
        current_time = time.time()
        
        # Check if token is still valid (with 5-minute buffer)
        if self._access_token and current_time < (self._token_expiry - 300):
            return self._access_token
            
        try:
            # Use gcloud to get access token
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                check=True
            )
            self._access_token = result.stdout.strip()
            # Tokens typically expire in 1 hour
            self._token_expiry = current_time + 3600
            return self._access_token
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to get access token. Make sure you're authenticated with 'gcloud auth application-default login': {e}"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Vertex AI API requests."""
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
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
            raise RuntimeError(f"Vertex AI API error: {error_message}")

    def _get_model_path(self) -> str:
        """Get the full model path for Vertex AI."""
        model_name = self.get_model_name()
        return f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{model_name}"

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        # Use base class implementation which handles filtering of unsupported features
        return super()._get_api_kwargs()

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts by replacing newlines with spaces
        texts = [self._clean_text(text) for text in texts]
        
        results = []
        model_path = self._get_model_path()
        
        for text in texts:
            # Prepare request payload for Vertex AI embedding
            payload = {
                "instances": [{
                    "content": text
                }]
            }
            
            # Make HTTP request
            response = self.client.post(
                f"{self.base_url}/{model_path}:predict",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)
            
            response_data = response.json()
            # Extract embedding from response
            embedding = response_data["predictions"][0]["embeddings"]["values"]
            results.append([float(value) for value in embedding])
        
        return results

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        # Clean texts by replacing newlines with spaces
        texts = [self._clean_text(text) for text in texts]
        
        results = []
        model_path = self._get_model_path()
        
        for text in texts:
            # Prepare request payload for Vertex AI embedding
            payload = {
                "instances": [{
                    "content": text
                }]
            }
            
            # Make async HTTP request
            response = await self.async_client.post(
                f"{self.base_url}/{model_path}:predict",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)
            
            response_data = response.json()
            # Extract embedding from response
            embedding = response_data["predictions"][0]["embeddings"]["values"]
            results.append([float(value) for value in embedding])
        
        return results

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "text-embedding-005"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "vertex"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="text-embedding-005",
                owned_by="Google",
                context_window=3072,
                type="embedding",
            ),
            Model(
                id="textembedding-gecko",
                owned_by="Google",
                context_window=3072,
                type="embedding",
            ),
            Model(
                id="textembedding-gecko-multilingual",
                owned_by="Google",
                context_window=3072,
                type="embedding",
            ),
        ]
