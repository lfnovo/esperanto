"""Base embedding model interface."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from esperanto.common_types import Model
from esperanto.common_types.task_type import EmbeddingTaskType


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
        
        # Extract task-aware settings from config
        self.task_type = self._config.get("task_type")
        self.late_chunking = self._config.get("late_chunking", False)
        self.output_dimensions = self._config.get("output_dimensions")
        self.truncate_at_max_length = self._config.get("truncate_at_max_length", True)
        
        # Convert string task_type to enum if needed
        if self.task_type and isinstance(self.task_type, str):
            try:
                self.task_type = EmbeddingTaskType(self.task_type)
            except ValueError:
                # Try with underscore to dot conversion for backward compatibility
                try:
                    normalized_task = self.task_type.replace("_", ".")
                    self.task_type = EmbeddingTaskType(normalized_task)
                except ValueError:
                    # Invalid task type, use default behavior
                    self.task_type = None

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

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for embedding.
        
        Based on Microsoft Azure OpenAI best practices but useful for all providers.
        Normalizes spacing, removes unwanted characters, and cleans up punctuation.
        
        Args:
            text: The text to clean and normalize.
            
        Returns:
            The cleaned and normalized text.
        """
        # Normalize spacing - replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,])', r'\1', text)
        
        # Remove repeated punctuation (multiple dots)
        text = re.sub(r'\.{2,}', '.', text)
        
        # Replace newlines and carriage returns with spaces
        text = re.sub(r'[\n\r]+', ' ', text)
        
        # Strip to clean up after replacements
        return text.strip()
    
    def _apply_task_optimization(self, texts: List[str]) -> List[str]:
        """Apply task-specific optimization to texts (base implementation).
        
        This default implementation adds task-specific prefixes for providers
        that don't have native task optimization support. Providers with native
        support should override this method to return texts unchanged.
        
        Args:
            texts: List of texts to optimize.
            
        Returns:
            List of optimized texts.
        """
        if not self.task_type or self.task_type == EmbeddingTaskType.DEFAULT:
            return texts
            
        # Default implementation: add task-specific prefix
        prefix_map = {
            EmbeddingTaskType.RETRIEVAL_QUERY: "query: ",
            EmbeddingTaskType.RETRIEVAL_DOCUMENT: "passage: ",
            EmbeddingTaskType.SIMILARITY: "similarity: ",
            EmbeddingTaskType.CLASSIFICATION: "classify: ",
            EmbeddingTaskType.CLUSTERING: "cluster: ",
            EmbeddingTaskType.CODE_RETRIEVAL: "code: ",
            EmbeddingTaskType.QUESTION_ANSWERING: "question: ",
            EmbeddingTaskType.FACT_VERIFICATION: "verify: "
        }
        
        prefix = prefix_map.get(self.task_type, "")
        if prefix:
            return [prefix + text for text in texts]
        return texts
    
    def _apply_late_chunking(self, texts: List[str], max_chunk_size: int = 512) -> List[str]:
        """Apply late chunking if enabled (base implementation).
        
        This is a simple implementation for providers without native support.
        Providers with sophisticated chunking should override this method.
        
        Args:
            texts: List of texts to chunk.
            max_chunk_size: Maximum size of each chunk in characters.
            
        Returns:
            List of chunked texts.
        """
        if not self.late_chunking:
            return texts
            
        chunked = []
        for text in texts:
            # Simple chunking by sentence boundaries or max length
            if len(text) <= max_chunk_size:
                chunked.append(text)
            else:
                # Split by sentences first
                sentences = text.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # Add period back if it was removed
                    if not sentence.endswith('.'):
                        sentence += '.'
                        
                    # Check if adding this sentence would exceed max size
                    if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                        chunked.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk = (current_chunk + " " + sentence).strip()
                
                # Add the last chunk
                if current_chunk:
                    chunked.append(current_chunk.strip())
                    
        return chunked
    
    def _log_unsupported_feature(self, feature: str) -> None:
        """Log when a feature isn't supported by this provider.
        
        Args:
            feature: Name of the unsupported feature.
        """
        # Silent logging - providers can override if they want to log
        pass
    
    def _serialize_config_for_api(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize config values for API calls.
        
        Converts enum values to their string representations for JSON serialization.
        
        Args:
            config: Configuration dictionary that may contain enum values.
            
        Returns:
            Serialized configuration dictionary.
        """
        serialized = {}
        for key, value in config.items():
            if isinstance(value, EmbeddingTaskType):
                # Convert enum to string value
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized
    
    def _filter_unsupported_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out parameters not supported by this provider.
        
        Args:
            kwargs: Configuration dictionary with all parameters.
            
        Returns:
            Filtered dictionary with only supported parameters.
        """
        # Define known advanced features
        advanced_features = ["task_type", "late_chunking", "output_dimensions", "truncate_at_max_length"]
        
        # If provider doesn't explicitly support advanced features, remove them
        supported_features = getattr(self.__class__, 'SUPPORTED_FEATURES', [])
        if not supported_features:
            for feature in advanced_features:
                kwargs.pop(feature, None)
        else:
            # Remove features not in the supported list
            for feature in advanced_features:
                if feature not in supported_features:
                    kwargs.pop(feature, None)
        
        return kwargs
    
    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args.
        
        This is a base implementation that providers can override or use directly.
        
        Returns:
            Filtered and serialized kwargs ready for API calls.
        """
        # Start with a copy of the config
        kwargs = self._config.copy()
        
        # Remove common provider-specific kwargs
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        kwargs.pop("organization", None)
        
        # Filter out unsupported advanced features
        kwargs = self._filter_unsupported_params(kwargs)
        
        # Serialize enums to string values
        kwargs = self._serialize_config_for_api(kwargs)
        
        return kwargs

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
        """Get the default model name.

        Returns:
            str: The default model name.
        """
        pass
