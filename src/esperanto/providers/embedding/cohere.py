"""Cohere embedding model provider (native v2 API)."""

import os
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List

import httpx

from esperanto.providers.embedding.base import EmbeddingModel, Model
from esperanto.utils import validate_and_decode_embedding

if TYPE_CHECKING:
    from langchain_cohere import CohereEmbeddings

# Cohere's /v2/embed accepts a maximum of 96 texts per request.
_VALID_INPUT_TYPES = (
    "search_document",
    "search_query",
    "classification",
    "clustering",
    "image",
)


class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embedding model implementation using the native v2 embed API."""

    MAX_BATCH_SIZE: ClassVar[int] = 96

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get API key
        self.api_key = self.api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cohere API key not found. Set the COHERE_API_KEY environment variable."
            )

        # Set base URL (endpoints carry their own version segment)
        self.base_url = (self.base_url or "https://api.cohere.com").rstrip("/")

        # input_type is required by Cohere; default to search_document.
        self.input_type = self._config.get("input_type", "search_document")

        if "model_name" in kwargs:
            self._config["model_name"] = kwargs["model_name"]

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Cohere API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("message") or error_data.get(
                    "error", f"HTTP {response.status_code}"
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Cohere API error: {error_message}")

    def _build_payload(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Build the request payload for a single batch of texts."""
        input_type = kwargs.pop("input_type", self.input_type)
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        # embed-v4+ supports output_dimension; pass through when configured.
        if self.output_dimensions:
            payload["output_dimension"] = self.output_dimensions
        payload.update(kwargs)
        return payload

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Texts are automatically batched in groups of ``MAX_BATCH_SIZE`` (96) to
        respect Cohere's per-request limit.
        """
        texts = [self._clean_text(text) for text in texts]

        results: List[List[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[start : start + self.MAX_BATCH_SIZE]
            payload = self._build_payload(batch, **kwargs)

            response = self.client.post(
                f"{self.base_url}/v2/embed",
                headers=self._get_headers(),
                json=payload,
            )
            self._handle_error(response)

            results.extend(self._parse_response(response.json()))
        return results

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously."""
        texts = [self._clean_text(text) for text in texts]

        results: List[List[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[start : start + self.MAX_BATCH_SIZE]
            payload = self._build_payload(batch, **kwargs)

            response = await self.async_client.post(
                f"{self.base_url}/v2/embed",
                headers=self._get_headers(),
                json=payload,
            )
            self._handle_error(response)

            results.extend(self._parse_response(response.json()))
        return results

    def _parse_response(self, response_data: Dict[str, Any]) -> List[List[float]]:
        """Extract float embeddings from a Cohere embed response."""
        float_embeddings = response_data.get("embeddings", {}).get("float", [])
        return [
            validate_and_decode_embedding(idx, raw)
            for idx, raw in enumerate(float_embeddings)
        ]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "embed-v4.0"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "cohere"

    def _get_models(self) -> List[Model]:
        """List all available embedding models for this provider."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/models",
                headers=self._get_headers(),
                params={"endpoint": "embed"},
            )
            self._handle_error(response)
            models_data = response.json()
            return [
                Model(
                    id=model["name"],
                    owned_by="cohere",
                    context_window=model.get("context_length"),
                    type="embedding",
                )
                for model in models_data.get("models", [])
            ]
        except Exception:
            return [
                Model(
                    id="embed-v4.0",
                    owned_by="cohere",
                    context_window=128000,
                    type="embedding",
                ),
            ]

    def to_langchain(self) -> "CohereEmbeddings":
        """Convert to a LangChain embeddings model.

        Raises:
            ImportError: If langchain_cohere is not installed.
        """
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_cohere. "
                "Install with: uv add langchain_cohere or pip install langchain_cohere"
            ) from e

        return CohereEmbeddings(  # type: ignore[call-arg]
            model=self.get_model_name(),
            cohere_api_key=self.api_key,  # type: ignore[arg-type]
        )
