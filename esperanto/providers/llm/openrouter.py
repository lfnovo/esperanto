import os
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from esperanto.base.types import LanguageModel


class OpenRouterLanguageModel(LanguageModel):
    """OpenRouter language model implementation."""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = {},
    ):
        super().__init__(model_name, config)
        self.max_tokens = self.config.get("max_tokens", 850)
        self.temperature = self.config.get("temperature", 1.0)
        self.streaming = self.config.get("streaming", True)
        self.top_p = self.config.get("top_p", 0.9)

    @property
    def provider(self) -> str:
        return "openrouter"

    def validate_config(self) -> None:
        """Validate OpenRouter configuration."""
        if not self.model_name:
            raise ValueError(
                "model_name must be specified for OpenRouter language model"
            )

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            max_tokens=self.max_tokens,
            streaming=self.streaming,
            api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY", "openrouter")),
            top_p=self.top_p,
        )
