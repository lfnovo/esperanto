"""Base reranker model interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import httpx

from esperanto.common_types import Model
from esperanto.common_types.reranker import RerankResponse


@dataclass
class RerankerModel(ABC):
    """Base class for all reranker providers."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    _config: Dict[str, Any] = field(default_factory=dict)
    client: Any = None
    async_client: Any = None

    def __post_init__(self):
        """Initialize configuration after dataclass initialization."""
        # Initialize config with default values
        self._config = {
            "model_name": self.model_name,
        }

        # Update with any provided config
        if hasattr(self, "config") and self.config:
            self._config.update(self.config)

            # Update instance attributes from config
            for key, value in self._config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Set model name if not provided
        if not self.model_name:
            self.model_name = self._get_default_model()

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider name."""
        pass

    @property
    @abstractmethod
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        pass

    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model name."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """Rerank documents based on relevance to query.

        Args:
            query: The search query to rank documents against.
            documents: List of documents to rerank.
            top_k: Maximum number of results to return. If None, returns all.
            **kwargs: Additional arguments specific to the provider.

        Returns:
            RerankResponse: Standardized response with ranked results.
        """
        pass

    @abstractmethod
    async def arerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """Async rerank documents based on relevance to query.

        Args:
            query: The search query to rank documents against.
            documents: List of documents to rerank.
            top_k: Maximum number of results to return. If None, returns all.
            **kwargs: Additional arguments specific to the provider.

        Returns:
            RerankResponse: Standardized response with ranked results.
        """
        pass

    @abstractmethod
    def to_langchain(self):
        """Convert to LangChain-compatible reranker."""
        pass

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            str: The model name.
        """
        # First try to get from config
        model_name = self._config.get("model_name")
        if model_name:
            return model_name

        # If not in config, use default
        return self._get_default_model()

    def _validate_inputs(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int]
    ) -> Tuple[str, List[str], int]:
        """Validate and normalize inputs.

        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Maximum number of results to return.

        Returns:
            Tuple of validated (query, documents, top_k).

        Raises:
            ValueError: If inputs are invalid.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if any(not isinstance(doc, str) for doc in documents):
            raise ValueError("All documents must be strings")

        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            if top_k > len(documents):
                top_k = len(documents)
        else:
            top_k = len(documents)

        return query.strip(), documents, top_k

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization.

        Args:
            scores: List of raw scores from the provider.

        Returns:
            List of normalized scores in 0-1 range.
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # Handle case where all scores are the same
        if max_score == min_score:
            return [0.5] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from config dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Cleaned configuration dictionary.
        """
        return {k: v for k, v in config.items() if v is not None}