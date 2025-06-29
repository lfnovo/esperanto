"""Jina AI embedding model implementation."""

import os
from typing import Any, Dict, List

import httpx

from esperanto.common_types import Model
from esperanto.common_types.task_type import EmbeddingTaskType

from .base import EmbeddingModel


class JinaEmbeddingModel(EmbeddingModel):
    """Jina AI embeddings with native support for all advanced features."""
    
    # Jina supports all advanced features natively
    SUPPORTED_FEATURES = ["task_type", "late_chunking", "output_dimensions", "truncate_at_max_length"]
    
    # Task type mapping from universal enum to Jina API values
    TASK_MAPPING = {
        EmbeddingTaskType.RETRIEVAL_QUERY: "retrieval.query",
        EmbeddingTaskType.RETRIEVAL_DOCUMENT: "retrieval.passage",
        EmbeddingTaskType.SIMILARITY: "text-matching",
        EmbeddingTaskType.CLASSIFICATION: "classification", 
        EmbeddingTaskType.CLUSTERING: "separation",
        EmbeddingTaskType.CODE_RETRIEVAL: "code.query",
        EmbeddingTaskType.DEFAULT: None
    }
    
    def __init__(self, **kwargs):
        """Initialize Jina embedding model.
        
        Args:
            **kwargs: Configuration parameters including:
                - api_key: Jina API key (or set JINA_API_KEY env var)
                - model_name: Model to use (default: jina-embeddings-v3)
                - base_url: API base URL (default: https://api.jina.ai/v1/embeddings)
                - config: Dict with task_type, late_chunking, output_dimensions, etc.
        """
        super().__init__(**kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina API key not found. Please set the JINA_API_KEY environment "
                "variable or pass it as 'api_key' parameter."
            )
        self.base_url = kwargs.get("base_url", "https://api.jina.ai/v1/embeddings")
        
        # Get timeout from config or use default
        timeout = self._config.get("timeout", 30.0) if hasattr(self, "_config") else 30.0
        
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)
    
    def close(self):
        """Explicitly close HTTP clients."""
        try:
            if hasattr(self, 'client') and not self.client.is_closed:
                self.client.close()
        except Exception:
            pass  # Ignore cleanup errors
            
    async def aclose(self):
        """Asynchronously close HTTP clients."""
        try:
            if hasattr(self, 'async_client') and not self.async_client.is_closed:
                await self.async_client.aclose()
        except Exception:
            pass  # Ignore cleanup errors
            
    def __del__(self):
        """Clean up HTTP clients on object destruction."""
        # Only handle sync client cleanup in destructor
        try:
            if hasattr(self, 'client') and not self.client.is_closed:
                self.client.close()
        except Exception:
            pass  # Ignore cleanup errors
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.aclose()
        
    def _apply_task_optimization(self, texts: List[str]) -> List[str]:
        """Jina handles task optimization natively via API."""
        # Don't apply prefix - Jina API handles this natively
        return texts
    
    def _apply_late_chunking(self, texts: List[str]) -> List[str]:
        """Jina handles late chunking natively via API."""
        # Don't chunk here - Jina API handles this natively
        return texts
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _map_task_type(self) -> str:
        """Map universal task type to Jina-specific value."""
        if not self.task_type:
            return None
        return self.TASK_MAPPING.get(self.task_type)
    
    def _build_request_payload(self, texts: List[str]) -> Dict[str, Any]:
        """Build request payload for Jina API.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            Request payload dict.
        """
        # Clean texts using base class method
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Build Jina-specific payload
        input_data = [{"text": text} for text in cleaned_texts]
        
        payload = {
            "model": self.get_model_name(),
            "input": input_data
        }
        
        # Map and add task type if specified
        if self.task_type:
            jina_task = self._map_task_type()
            if jina_task:
                payload["task"] = jina_task
                
        # Add other advanced features - all natively supported
        if self.late_chunking:
            payload["late_chunking"] = True
            
        if self.truncate_at_max_length:
            payload["truncate"] = True
            
        if self.output_dimensions:
            payload["dimensions"] = self.output_dimensions
            
        return payload
    
    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses from Jina API.
        
        Args:
            response: HTTP response object.
            
        Raises:
            RuntimeError: With details from the error response.
        """
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
            error_type = error_data.get("error", {}).get("type", "Unknown")
            raise RuntimeError(f"Jina API error ({error_type}): {error_message}")
        except (KeyError, ValueError):
            raise RuntimeError(f"Jina API error: {response.status_code} - {response.text}")

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.
        
        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments (not used for Jina).
            
        Returns:
            List of embeddings, one for each input text.
        """
        if not texts:
            return []
            
        payload = self._build_request_payload(texts)
        
        try:
            response = self.client.post(
                self.base_url,
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                self._handle_error(response)
                
            response_data = response.json()
            
            # Extract embeddings from response
            embeddings = []
            for item in response_data.get("data", []):
                embedding = item.get("embedding")
                if embedding:
                    # Ensure all values are floats for consistency
                    embeddings.append([float(value) for value in embedding])
                    
            return embeddings
            
        except httpx.TimeoutException:
            raise RuntimeError("Request to Jina API timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling Jina API: {str(e)}")
        finally:
            pass  # Client is reused, don't close

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.
        
        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments (not used for Jina).
            
        Returns:
            List of embeddings, one for each input text.
        """
        if not texts:
            return []
            
        payload = self._build_request_payload(texts)
        
        try:
            response = await self.async_client.post(
                self.base_url,
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                self._handle_error(response)
                
            response_data = response.json()
            
            # Extract embeddings from response
            embeddings = []
            for item in response_data.get("data", []):
                embedding = item.get("embedding")
                if embedding:
                    # Ensure all values are floats for consistency
                    embeddings.append([float(value) for value in embedding])
                    
            return embeddings
            
        except httpx.TimeoutException:
            raise RuntimeError("Request to Jina API timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling Jina API: {str(e)}")

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "jina"

    @property
    def models(self) -> List[Model]:
        """List available Jina embedding models."""
        return [
            Model(
                id="jina-embeddings-v4",
                owned_by="jina",
                context_window=8192,
                type="embedding"
            ),
            Model(
                id="jina-embeddings-v3", 
                owned_by="jina",
                context_window=8192,
                type="embedding"
            ),
            Model(
                id="jina-embeddings-v2-base-en",
                owned_by="jina",
                context_window=8192,
                type="embedding"
            ),
            Model(
                id="jina-embeddings-v2-small-en",
                owned_by="jina",
                context_window=8192,
                type="embedding"
            ),
            Model(
                id="jina-embeddings-v2-base-multilingual",
                owned_by="jina",
                context_window=8192,
                type="embedding"
            ),
            Model(
                id="jina-clip-v1",
                owned_by="jina",
                context_window=None,  # Multimodal model
                type="embedding"
            ),
            Model(
                id="jina-clip-v2",
                owned_by="jina",
                context_window=None,  # Multimodal model
                type="embedding"
            ),
        ]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "jina-embeddings-v3"