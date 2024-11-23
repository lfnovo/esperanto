import os
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from esperanto.base.types import LanguageModel


class GroqLanguageModel(LanguageModel):
    """Groq language model implementation."""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = {},
    ):
        if not model_name:
            raise ValueError("model_name must be specified for Groq language model")
        super().__init__(model_name, config)
        self.api_key = self.config.get("api_key", os.environ.get("GROQ_API_KEY"))
        self.max_tokens = self.config.get("max_tokens", 850)
        self.temperature = self.config.get("temperature", 1.0)
        self.streaming = self.config.get("streaming", True)
        self.top_p = self.config.get("top_p", 0.9)

    @property
    def provider(self) -> str:
        return "groq"

    def validate_config(self) -> None:
        """Validate Groq configuration."""
        if not self.model_name:
            raise ValueError("model_name must be specified for Groq language model")

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        return ChatGroq(
            api_key=self.api_key,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            streaming=self.streaming,
            model_kwargs={"top_p": self.top_p},
        )
