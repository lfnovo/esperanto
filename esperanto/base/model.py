from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """Base class for all AI models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = {}):
        """Initialize the base model.

        Args:
            model_name: The name of the model to use
            config: Optional configuration dictionary for the model
        """
        self.model_name = model_name
        self.config = config or {}

    @abstractmethod
    def validate_config(self) -> None:
        """Validate the model configuration."""
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name for this model."""
        pass
