"""Base embedding model interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingModel(ABC):
    """Base class for all embedding models."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    organization: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    _config: Dict[str, Any] = field(default_factory=dict)

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

    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
        pass

    @abstractmethod
    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for.
            **kwargs: Additional arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each input text.
        """
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

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider name."""
        pass

    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model name.

        Returns:
            str: The default model name.
        """
        pass
