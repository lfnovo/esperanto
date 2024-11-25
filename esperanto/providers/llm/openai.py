import os
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from esperanto.base.types import LanguageModel


class OpenAILanguageModel(LanguageModel):
    """OpenAI language model implementation."""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = {},
    ):
        if not model_name:
            raise ValueError("model_name must be specified for OpenAI language model")
        super().__init__(model_name, config)
        self.max_tokens: int = self.config.get("max_tokens", 850)
        self.temperature: float = self.config.get("temperature", 1.0)
        self.streaming: bool = self.config.get("streaming", True)
        self.top_p: float = self.config.get("top_p", 0.9)
        self.json_mode: bool = self.config.get("json", False)

        # Handle API configuration
        api_key = self.config.get("api_key") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        self.api_key: SecretStr = SecretStr(str(api_key)) if api_key else SecretStr("")
        self.base_url: str | None = self.config.get("openai_api_base") or os.environ.get(
            "OPENAI_API_BASE", None
        )
        self.organization: str | None = self.config.get("organization") or os.environ.get(
            "OPENAI_ORGANIZATION", None
        )

    @property
    def provider(self) -> str:
        return "openai"

    def validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.model_name:
            raise ValueError("model_name must be specified for OpenAI language model")
        if not self.api_key.get_secret_value():
            raise ValueError(
                "api_key must be specified in config or OPENAI_API_KEY environment variable"
            )

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        model_kwargs: Dict[str, Any] = {
            "response_format": {"type": "json"} if self.json_mode else None
        }

        return ChatOpenAI(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            streaming=self.streaming,
            top_p=self.top_p,
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            model_kwargs=model_kwargs,
        )
