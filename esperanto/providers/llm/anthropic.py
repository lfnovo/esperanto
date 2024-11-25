"""Anthropic language model implementation."""

import os
from typing import Any, ClassVar, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from esperanto.base.types import LanguageModel


class AnthropicLanguageModel(LanguageModel):
    """Anthropic language model implementation."""

    DEFAULT_MAX_TOKENS: ClassVar[int] = 850
    DEFAULT_TEMPERATURE: ClassVar[float] = 1.0
    DEFAULT_TOP_P: ClassVar[float] = 0.9
    MIN_TEMPERATURE: ClassVar[float] = 0.0
    MAX_TEMPERATURE: ClassVar[float] = 1.0
    MIN_TOP_P: ClassVar[float] = 0.0
    MAX_TOP_P: ClassVar[float] = 1.0

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = {},
    ):
        """Initialize Anthropic language model.

        Args:
            model_name: Name of the model to use. Defaults to claude-3-opus-20240229.
            config: Configuration dictionary. Can include:
                - api_key: Anthropic API key
                - base_url: Optional API base URL
                - max_tokens: Maximum number of tokens to generate (default: 850)
                - temperature: Sampling temperature (default: 1.0)
                - top_p: Nucleus sampling parameter (default: 0.9)
                - streaming: Whether to stream responses (default: True)
        """
        super().__init__(model_name=model_name, config=config)

        # Model configuration
        self._max_tokens = self.config.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        self._temperature = self.config.get("temperature", self.DEFAULT_TEMPERATURE)
        self._top_p = self.config.get("top_p", self.DEFAULT_TOP_P)
        self._streaming = self.config.get("streaming", True)

        # API configuration
        api_key = self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        self._api_key = (
            api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        )
        self._base_url = self.config.get("base_url") or os.environ.get(
            "ANTHROPIC_API_BASE"
        )

        self.validate_config()

    @property
    def provider(self) -> str:
        """Get provider name."""
        return "anthropic"

    @property
    def max_tokens(self) -> int:
        """Get maximum tokens setting."""
        return self._max_tokens

    @property
    def temperature(self) -> float:
        """Get temperature setting."""
        return self._temperature

    @property
    def top_p(self) -> float:
        """Get top_p setting."""
        return self._top_p

    @property
    def streaming(self) -> bool:
        """Get streaming setting."""
        return self._streaming

    @property
    def api_key(self) -> SecretStr:
        """Get API key."""
        return self._api_key

    @property
    def base_url(self) -> Optional[str]:
        """Get base URL."""
        return self._base_url

    def validate_config(self) -> None:
        """Validate Anthropic configuration.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if not self.model_name:
            raise ValueError(
                "model_name must be specified for Anthropic language model"
            )
        if not self.api_key.get_secret_value():
            raise ValueError(
                "api_key must be specified in config or ANTHROPIC_API_KEY environment variable"
            )
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not self.MIN_TEMPERATURE <= self.temperature <= self.MAX_TEMPERATURE:
            raise ValueError(
                f"temperature must be between {self.MIN_TEMPERATURE} and {self.MAX_TEMPERATURE}"
            )
        if not self.MIN_TOP_P <= self.top_p <= self.MAX_TOP_P:
            raise ValueError(
                f"top_p must be between {self.MIN_TOP_P} and {self.MAX_TOP_P}"
            )

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model.

        Returns:
            BaseChatModel: LangChain chat model instance.

        Raises:
            ValueError: If configuration is invalid.
        """
        try:
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=None,
                max_retries=2,
                anthropic_api_key=self.api_key,
                streaming=self.streaming,
                top_p=self.top_p,
            )

        except Exception as e:
            raise ValueError(f"Failed to create LangChain model: {str(e)}") from e
