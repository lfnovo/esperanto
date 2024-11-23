"""Factory module for creating AI service instances."""

from typing import Any, Dict, Optional, Type, Union

from esperanto.providers.llm import (
    AnthropicLanguageModel,
    GeminiLanguageModel,
    GroqLanguageModel,
    LiteLLMLanguageModel,
    OllamaLanguageModel,
    OpenAILanguageModel,
    OpenRouterLanguageModel,
    VertexAILanguageModel,
    VertexAnthropicLanguageModel,
    XAILanguageModel,
)
from esperanto.providers.speech_to_text import (
    GroqSpeechToTextModel,
    OpenAISpeechToTextModel,
)
from esperanto.providers.text_to_speech import (
    ElevenLabsTextToSpeechModel,
    GeminiTextToSpeechModel,
    OpenAITextToSpeechModel,
)


class AIFactory:
    """Factory class for creating AI service instances."""

    # LLM provider mapping
    _llm_providers = {
        "openai": OpenAILanguageModel,
        "anthropic": AnthropicLanguageModel,
        "gemini": GeminiLanguageModel,
        "groq": GroqLanguageModel,
        "litellm": LiteLLMLanguageModel,
        "ollama": OllamaLanguageModel,
        "openrouter": OpenRouterLanguageModel,
        "vertex": VertexAILanguageModel,
        "vertex_anthropic": VertexAnthropicLanguageModel,
        "xai": XAILanguageModel,
    }

    # Speech-to-text provider mapping
    _stt_providers = {
        "openai": OpenAISpeechToTextModel,
        "groq": GroqSpeechToTextModel,
    }

    # Text-to-speech provider mapping
    _tts_providers = {
        "elevenlabs": ElevenLabsTextToSpeechModel,
        "gemini": GeminiTextToSpeechModel,
        "openai": OpenAITextToSpeechModel,
    }

    @classmethod
    def create_llm(
        cls,
        provider: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a language model instance.

        Args:
            provider: The provider name (e.g., 'openai', 'anthropic', 'gemini')
            model_name: The name of the model to use
            config: Optional configuration dictionary

        Returns:
            A language model instance

        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider.lower()
        if provider not in cls._llm_providers:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {list(cls._llm_providers.keys())}"
            )

        model_class = cls._llm_providers[provider]
        return model_class(model_name=model_name, config=config or {})

    @classmethod
    def create_stt(
        cls,
        provider: str,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a speech-to-text model instance.

        Args:
            provider: The provider name (e.g., 'openai', 'groq')
            model_name: Optional model name
            config: Optional configuration dictionary

        Returns:
            A speech-to-text model instance

        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider.lower()
        if provider not in cls._stt_providers:
            raise ValueError(
                f"Unsupported speech-to-text provider: {provider}. "
                f"Supported providers: {list(cls._stt_providers.keys())}"
            )

        model_class = cls._stt_providers[provider]
        return model_class(model_name=model_name, config=config or {})

    @classmethod
    def create_tts(
        cls,
        provider: str,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a text-to-speech model instance.

        Args:
            provider: The provider name (e.g., 'elevenlabs', 'openai', 'gemini')
            model_name: Optional model name
            config: Optional configuration dictionary

        Returns:
            A text-to-speech model instance

        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider.lower()
        if provider not in cls._tts_providers:
            raise ValueError(
                f"Unsupported text-to-speech provider: {provider}. "
                f"Supported providers: {list(cls._tts_providers.keys())}"
            )

        model_class = cls._tts_providers[provider]
        return model_class(model_name=model_name, config=config or {})
