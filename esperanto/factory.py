"""Factory module for creating AI service instances."""

import importlib
import warnings
from typing import Any, Dict, List, Optional, Type

from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.stt.base import SpeechToTextModel
from esperanto.providers.tts.base import TextToSpeechModel


class AIFactory:
    """Factory class for creating AI service instances."""

    # Provider module mappings
    _provider_modules = {
        "language": {
            "openai": "esperanto.providers.llm.openai:OpenAILanguageModel",
            "anthropic": "esperanto.providers.llm.anthropic:AnthropicLanguageModel",
            "google": "esperanto.providers.llm.google:GoogleLanguageModel",
            "groq": "esperanto.providers.llm.groq:GroqLanguageModel",
            "ollama": "esperanto.providers.llm.ollama:OllamaLanguageModel",
            "openrouter": "esperanto.providers.llm.openrouter:OpenRouterLanguageModel",
            "xai": "esperanto.providers.llm.xai:XAILanguageModel",
        },
        "embedding": {
            "openai": "esperanto.providers.embedding.openai:OpenAIEmbeddingModel",
            "google": "esperanto.providers.embedding.google:GoogleEmbeddingModel",
            "ollama": "esperanto.providers.embedding.ollama:OllamaEmbeddingModel",
            "vertex": "esperanto.providers.embedding.vertex:VertexEmbeddingModel",
        },
        "speech_to_text": {
            "openai": "esperanto.providers.stt.openai:OpenAISpeechToTextModel",
            "groq": "esperanto.providers.stt.groq:GroqSpeechToTextModel",
        },
        "text_to_speech": {
            "openai": "esperanto.providers.tts.openai:OpenAITextToSpeechModel",
            "elevenlabs": "esperanto.providers.tts.elevenlabs:ElevenLabsTextToSpeechModel",
            "google": "esperanto.providers.tts.google:GoogleTextToSpeechModel",
        },
    }

    @classmethod
    def _import_provider_class(cls, service_type: str, provider: str) -> Type:
        """Dynamically import provider class.

        Args:
            service_type: Type of service (language, embedding, speech_to_text, text_to_speech)
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
    def get_available_providers(cls) -> Dict[str, List[str]]:
        """Get a dictionary of available providers for each model type.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are model types (language, embedding, speech_to_text, text_to_speech)
                and values are lists of available provider names.
        """
        return {
            model_type: list(providers.keys())
            for model_type, providers in cls._provider_modules.items()
        }

    @classmethod
    def _create_instance(
        cls,
        service_type: str,
        provider: str,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        provider_class = cls._import_provider_class(service_type, provider)
        return provider_class(model_name=model_name, **kwargs)

    @classmethod
    def create_language(
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
        provider_class = cls._import_provider_class("language", provider)
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

    @classmethod
    def create_speech_to_text(
        cls, provider: str, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> SpeechToTextModel:
        """Create a speech-to-text model instance.

        Args:
            provider: Provider name
            model_name: Optional name of the model to use
            config: Optional configuration for the model

        Returns:
            Speech-to-text model instance
        """
        config = config or {}
        return cls._create_instance("speech_to_text", provider, model_name=model_name, **config)

    @classmethod
    def create_text_to_speech(
        cls,
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> TextToSpeechModel:
        """Create a text-to-speech model instance.

        Args:
            provider: Provider name (openai, elevenlabs, google)
            model_name: Name of the model to use
            api_key: API key for the provider
            base_url: Optional base URL for the API
            **kwargs: Additional provider-specific configuration

        Returns:
            TextToSpeechModel instance

        Raises:
            ValueError: If provider is not supported
            ImportError: If provider module is not installed
        """
        provider_class = cls._import_provider_class("text_to_speech", provider)
        return provider_class(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )

    @classmethod
    def create_stt(
        cls, provider: str, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> SpeechToTextModel:
        """Create a speech-to-text model instance (alias for create_speech_to_text).

        Args:
            provider: Provider name
            model_name: Optional name of the model to use
            config: Optional configuration for the model

        Returns:
            Speech-to-text model instance
        """
        warnings.warn(
            "create_stt() is deprecated and will be removed in a future version. "
            "Use create_speech_to_text() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.create_speech_to_text(provider, model_name=model_name, config=config)

    @classmethod
    def create_tts(
        cls,
        provider: str,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> TextToSpeechModel:
        """Create a text-to-speech model instance (alias for create_text_to_speech).

        Args:
            provider: Provider name
            model_name: Optional name of the model to use
            config: Optional configuration for the model
            api_key: Optional API key for authentication

        Returns:
            Text-to-speech model instance
        """
        warnings.warn(
            "create_tts() is deprecated and will be removed in a future version. "
            "Use create_text_to_speech() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.create_text_to_speech(provider, model_name=model_name, config=config, api_key=api_key)

    @classmethod
    def create_llm(
        cls, provider: str, model_name: str, config: Optional[Dict[str, Any]] = None
    ) -> LanguageModel:
        """Create a language model instance (alias for create_language).

        Args:
            provider: Provider name
            model_name: Name of the model to use
            config: Optional configuration for the model

        Returns:
            Language model instance
        """
        warnings.warn(
            "create_llm() is deprecated and will be removed in a future version. "
            "Use create_language() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.create_language(provider, model_name=model_name, config=config)
