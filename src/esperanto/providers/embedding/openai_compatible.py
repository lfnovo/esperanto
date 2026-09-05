"""OpenAI-compatible Embedding provider implementation."""

from typing import Any, ClassVar, Dict, List, Optional

import httpx

from esperanto.common_types import Model
from esperanto.providers.llm.profiles import OpenAICompatibleProfile
from esperanto.providers.profile_mixin import ProfileAwareMixin
from esperanto.utils import validate_and_decode_embedding
from esperanto.utils.logging import logger

from .base import EmbeddingModel


class OpenAICompatibleEmbeddingModel(ProfileAwareMixin, EmbeddingModel):
    """OpenAI-compatible Embedding provider implementation for custom endpoints.

    This provider extends OpenAI's embedding implementation to work with any OpenAI-compatible
    embedding endpoint, providing graceful fallback for features that may not be supported
    by all endpoints.

    Example:
        >>> from esperanto import AIFactory
        >>> embedder = AIFactory.create_embedding(
        ...     "openai-compatible",
        ...     model_name="nomic-embed-text",
        ...     config={
        ...         "base_url": "http://localhost:1234/v1",
        ...         "timeout": 120
        ...     }
        ... )
        >>> embeddings = embedder.embed(["Hello world", "How are you?"])
    """

    # Mirror OpenAI's per-request ceiling; a profile may lower it if its backend
    # needs a smaller batch, and users can override via config="embed_batch_size".
    MAX_BATCH_SIZE: ClassVar[int] = 2048

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize OpenAI-compatible embedding provider.

        Args:
            model_name: Name of the model to use
            api_key: API key for the provider. If not provided, will try to get from environment
            base_url: Base URL for the OpenAI-compatible endpoint
            config: Additional configuration options including:
                - timeout: Request timeout in seconds (default: 120)
            **kwargs: Additional configuration options
        """
        # Merge config and kwargs
        config = config or {}
        config.update(kwargs)

        # Resolve provider profile (None when not profile-driven) and configuration
        # via the shared precedence chain (ProfileAwareMixin).
        self._profile: Optional[OpenAICompatibleProfile] = self._resolve_profile(
            config, "embedding"
        )
        self.base_url = self._resolve_base_url(
            "embedding", self._profile, base_url, config
        )
        self.api_key = self._resolve_api_key(
            "embedding", self._profile, api_key, config
        )

        if self._profile:
            self.api_key = self._finalize_profile_credentials(
                self._profile, self.base_url, self.api_key
            )
        else:
            # Validation
            if not self.base_url:
                raise ValueError(
                    "OpenAI-compatible base URL is required. "
                    "Set OPENAI_COMPATIBLE_BASE_URL_EMBEDDING or OPENAI_COMPATIBLE_BASE_URL "
                    "environment variable or provide base_url in config."
                )
            # Use a default API key if none is provided (some endpoints don't require authentication)
            if not self.api_key:
                self.api_key = "not-required"

        # Ensure base_url doesn't end with trailing slash for consistency
        if self.base_url and self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

        # Get timeout configuration (default to 120 seconds for embedding operations)
        self.timeout = config.get("timeout", 120.0)

        # Remove base_url, api_key, timeout, and profile marker from config to avoid duplication
        clean_config = {
            k: v
            for k, v in config.items()
            if k not in ["base_url", "api_key", "timeout", "_profile_name"]
        }

        # Initialize attributes for dataclass
        self.model_name = model_name or self._get_default_model()
        self.config = clean_config

        # Call parent's __init__ to set up configuration
        super().__init__(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            config=self.config,
        )

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI-compatible API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses with graceful degradation."""
        if response.status_code >= 400:
            # Log original response for debugging
            logger.debug(f"OpenAI-compatible endpoint error: {response.text}")

            # Try to parse error message from multiple common formats
            try:
                error_data = response.json()
                # Try multiple error message formats
                error_message = (
                    error_data.get("error", {}).get("message")
                    or error_data.get("detail", {}).get("message")  # Some APIs use this
                    or error_data.get("message")  # Direct message field
                    or f"HTTP {response.status_code}"
                )
            except Exception:
                # Fall back to HTTP status code
                error_message = f"HTTP {response.status_code}: {response.text}"

            raise RuntimeError(
                f"OpenAI-compatible embedding endpoint error: {error_message}"
            )

    def _get_models(self) -> List[Model]:
        """List all available models for this provider.

        Note: This attempts to fetch models from the /models endpoint.
        If the endpoint doesn't support this, it will return an empty list.
        """
        try:
            response = self.client.get(
                f"{self.base_url}/models", headers=self._get_headers()
            )
            self._handle_error(response)

            models_data = response.json()
            return [
                Model(
                    id=model["id"],
                    owned_by=model.get("owned_by", "custom"),
                    context_window=model.get("context_window", None),
                )
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            # Log the error but don't fail completely
            logger.info(
                f"Models endpoint not supported by OpenAI-compatible embedding endpoint: {e}"
            )
            return []

    def _get_default_model(self) -> str:
        """Get the default model name.

        Returns the profile's embedding default if a profile is active,
        otherwise a generic default users should override.
        """
        return self._resolve_default_model(
            "embedding", getattr(self, "_profile", None), "text-embedding-3-small"
        )

    @property
    def provider(self) -> str:
        """Get the provider name (the profile name when profile-driven)."""
        if self._profile is not None:
            return self._profile.name
        return "openai-compatible"

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts using OpenAI-compatible Embedding API.

        Args:
            texts: List of texts to create embeddings for
            **kwargs: Additional parameters to pass to the API

        Returns:
            List of embeddings, one for each input text

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Clean texts using enhanced text cleaning
            texts = [self._clean_text(text) for text in texts]

            results: List[List[float]] = []
            for batch in self._iter_embed_batches(texts):
                # Prepare request payload using OpenAI standard format
                payload = {
                    "input": batch,
                    "model": self.get_model_name(),
                    "encoding_format": "float",
                    **{**self._get_api_kwargs(), **kwargs},
                }

                # Generate embeddings
                response = self.client.post(
                    f"{self.base_url}/embeddings", headers=self._get_headers(), json=payload
                )
                self._handle_error(response)

                # Parse response
                response_data = response.json()
                for idx, data in enumerate(response_data["data"], start=len(results)):
                    raw = data.get("embedding")
                    results.append(validate_and_decode_embedding(idx, raw))
            return results

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts using OpenAI-compatible Embedding API asynchronously.

        Args:
            texts: List of texts to create embeddings for
            **kwargs: Additional parameters to pass to the API

        Returns:
            List of embeddings, one for each input text

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Clean texts using enhanced text cleaning
            texts = [self._clean_text(text) for text in texts]

            results: List[List[float]] = []
            for batch in self._iter_embed_batches(texts):
                # Prepare request payload using OpenAI standard format
                payload = {
                    "input": batch,
                    "model": self.get_model_name(),
                    "encoding_format": "float",
                    **{**self._get_api_kwargs(), **kwargs},
                }

                # Generate embeddings
                response = await self.async_client.post(
                    f"{self.base_url}/embeddings", headers=self._get_headers(), json=payload
                )
                self._handle_error(response)

                # Parse response
                response_data = response.json()
                for idx, data in enumerate(response_data["data"], start=len(results)):
                    raw = data.get("embedding")
                    results.append(validate_and_decode_embedding(idx, raw))
            return results

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e
