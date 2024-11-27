"""Factory module for creating AI service instances."""

import importlib
from typing import Any, Dict, Optional, Type

from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.llm.base import LanguageModel


class AIFactory:
    """Factory class for creating AI service instances."""

    # Provider module mappings
    _provider_modules = {
        "llm": {
            "openai": "esperanto.providers.llm.openai:OpenAILanguageModel",
            "anthropic": "esperanto.providers.llm.anthropic:AnthropicLanguageModel",
            "gemini": "esperanto.providers.llm.gemini:GeminiLanguageModel",
            "groq": "esperanto.providers.llm.groq:GroqLanguageModel",
            "ollama": "esperanto.providers.llm.ollama:OllamaLanguageModel",
            "openrouter": "esperanto.providers.llm.openrouter:OpenRouterLanguageModel",
            "xai": "esperanto.providers.llm.xai:XAILanguageModel",
        },
        "embedding": {
            "openai": "esperanto.providers.embedding.openai:OpenAIEmbeddingModel",
            "gemini": "esperanto.providers.embedding.gemini:GeminiEmbeddingModel",
            "ollama": "esperanto.providers.embedding.ollama:OllamaEmbeddingModel",
            "vertex": "esperanto.providers.embedding.vertex:VertexEmbeddingModel",
        },
    }

    @classmethod
    def _import_provider_class(cls, service_type: str, provider: str) -> Type:
        """Dynamically import provider class.

        Args:
            service_type: Type of service (llm, stt, tts)
            provider: Provider name

        Returns:
            Provider class

        Raises:
            ValueError: If provider is not supported
            ImportError: If provider module is not installed
        """
        if service_type not in cls._provider_modules:
            raise ValueError(f"Invalid service type: {service_type}")

        provider = provider.lower()
        if provider not in cls._provider_modules[service_type]:
            raise ValueError(
                f"Provider '{provider}' not supported for {service_type}. "
                f"Supported providers: {list(cls._provider_modules[service_type].keys())}"
            )

        module_path = cls._provider_modules[service_type][provider]
        module_name, class_name = module_path.split(":")

        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except ImportError as e:
            # Get the provider package name from the module path
            provider_package = module_name.split(".")[
                3
            ]  # e.g., openai, anthropic, etc.
            raise ImportError(
                f"Provider '{provider}' requires additional dependencies. "
                f"Install them with: pip install esperanto[{provider_package}] "
                f"or poetry add esperanto[{provider_package}]"
            ) from e

    @classmethod
    def create_llm(
        cls, provider: str, model_name: str, config: Optional[Dict[str, Any]] = None
    ) -> LanguageModel:
        """Create a language model instance.

        Args:
            provider: Provider name
            model_name: Name of the model to use
            config: Optional configuration for the model

        Returns:
            Language model instance
        """
        provider_class = cls._import_provider_class("llm", provider)
        return provider_class(model_name=model_name, config=config or {})

    @classmethod
    def create_embedding(
        cls, provider: str, model_name: str, config: Optional[Dict[str, Any]] = None
    ) -> EmbeddingModel:
        """Create an embedding model instance.

        Args:
            provider: Provider name
            model_name: Name of the model to use
            config: Optional configuration for the model

        Returns:
            Embedding model instance
        """
        provider_class = cls._import_provider_class("embedding", provider)
        return provider_class(model_name=model_name, config=config or {})
