from typing import Any, Dict

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel

from esperanto.base.types import LanguageModel


class LiteLLMLanguageModel(LanguageModel):
    """LiteLLM language model implementation."""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = {},
    ):
        super().__init__(model_name, config)
        if not model_name:
            raise ValueError("model_name must be specified for LiteLLM language model")
        self.max_tokens = self.config.get("max_tokens", 850)
        self.temperature = self.config.get("temperature", 1.0)
        self.streaming = self.config.get("streaming", True)
        self.top_p = self.config.get("top_p", 0.9)
        self.api_base = self.config.get("api_base")

    @property
    def provider(self) -> str:
        return "litellm"

    def validate_config(self) -> None:
        """Validate LiteLLM configuration."""
        if not self.model_name:
            raise ValueError("model_name must be specified for LiteLLM language model")

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        config = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "streaming": self.streaming,
            "top_p": self.top_p,
        }
        if self.api_base:
            config["api_base"] = self.api_base

        return ChatLiteLLM(**config)
