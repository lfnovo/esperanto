"""Base language model interface."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from httpx import Client, AsyncClient

from esperanto.common_types import ChatCompletion, ChatCompletionChunk, Model
from esperanto.utils.connect import HttpConnectionMixin


@dataclass
class LanguageModel(HttpConnectionMixin, ABC):
    """Base class for all language models."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 850
    temperature: float = 1.0
    streaming: bool = False
    top_p: float = 0.9
    structured: Optional[Dict[str, Any]] = None
    organization: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    _config: Dict[str, Any] = field(default_factory=dict)
    client: Optional[Client] = None
    async_client: Optional[AsyncClient] = None

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.

        .. deprecated:: 2.8.0
            The `.models` property is deprecated and will be removed in version 3.0.
            Use `AIFactory.get_provider_models(provider_name)` instead for static
            model discovery without creating provider instances.

        Returns:
            List[Model]: List of available models
        """
        warnings.warn(
            f"The `.models` property is deprecated and will be removed in version 3.0. "
            f"Use AIFactory.get_provider_models('{self.provider}') instead for static "
            f"model discovery without creating provider instances.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._get_models()

    @abstractmethod
    def _get_models(self) -> List[Model]:
        """Internal method to get available models.

        This method should be implemented by providers. The public `.models` property
        will emit a deprecation warning and call this method.

        Returns:
            List[Model]: List of available models
        """
        pass

    def __post_init__(self):
        """Initialize configuration after dataclass initialization."""
        # Initialize config with default values
        self._config = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
            "structured": self.structured,
        }

        # Update with any provided config
        if hasattr(self, "config") and self.config:
            self._config.update(self.config)

            # Update instance attributes from config
            for key, value in self._config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from config dictionary."""
        return {k: v for k, v in config.items() if v is not None}

    @abstractmethod
    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation.
            stream: Whether to stream the response. If None, uses the instance's streaming setting.

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks if streaming.
        """
        pass

    @abstractmethod
    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request.

        Args:
            messages: List of messages in the conversation.
            stream: Whether to stream the response. If None, uses the instance's streaming setting.

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks if streaming.
        """
        pass

    def clean_config(self) -> Dict[str, Any]:
        """Clean the configuration dictionary.

        Returns:
            Dict[str, Any]: The cleaned configuration.
        """
        config = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and value is not None:
                config[key] = value
        return config

    def get_completion_kwargs(
        self, override_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get kwargs for completion API calls."""
        kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
        }

        if override_kwargs:
            kwargs.update(override_kwargs)

        return kwargs

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

    def _get_provider_type(self) -> str:
        """Return provider type for timeout configuration.

        Returns:
            str: "language" for LLM providers
        """
        return "language"

    @abstractmethod
    def to_langchain(self) -> Any:
        """Convert to a LangChain chat model.

        Returns:
            BaseChatModel: A LangChain chat model instance specific to the provider.

        Raises:
            ImportError: If langchain_core is not installed.
        """
        pass
