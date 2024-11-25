"""Gemini language model implementation."""

import os
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from esperanto.base.types import LanguageModel


class GeminiLanguageModel(LanguageModel):
    """Gemini language model implementation."""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = {},
    ) -> None:
        """Initialize Gemini language model.

        Args:
            model_name: Name of the model to use.
            config: Configuration for the model.

        Raises:
            ValueError: If the model name is invalid.
        """
        if not model_name or len(model_name.strip()) == 0:
            raise ValueError("model_name must be specified for Gemini language model")

        super().__init__(model_name=model_name, config=config or {})
        self.streaming = True  # Only use streaming for consistency
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.max_tokens = config.get("max_tokens", 850)

        # Handle API key
        api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        self._api_key = api_key

    @property
    def provider(self) -> str:
        """Get the provider name.

        Returns:
            Provider name.
        """
        return "gemini"

    def validate_config(self) -> None:
        """Validate the model configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.model_name or len(self.model_name.strip()) == 0:
            raise ValueError("model_name must be specified for Gemini language model")

        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        if self.max_tokens is not None and not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be an integer")

    def to_langchain(self) -> ChatGoogleGenerativeAI:
        """Convert to LangChain model.

        Returns:
            LangChain model instance.
        """
        self.validate_config()

        model = ChatGoogleGenerativeAI(
            model=f"models/{self.model_name}",
            api_key=SecretStr(self._api_key),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        return model
