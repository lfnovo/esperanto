import os
from typing import Any, Dict, Optional

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
        self.max_tokens = self.config.get("max_tokens", 850)
        self.temperature = self.config.get("temperature", 1.0)
        self.streaming = self.config.get("streaming", True)
        self.top_p = self.config.get("top_p", 0.9)
        self.json_mode = self.config.get("json", False)

        # Handle API configuration
        api_key = self.config.get("api_key") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        self.api_key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self.base_url = self.config.get("openai_api_base") or os.environ.get(
            "OPENAI_API_BASE", None
        )
        self.organization = self.config.get("organization") or os.environ.get(
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
        if self.json_mode:
            model_kwargs = {"response_format": {"type": "json"}}
        else:
            model_kwargs = {"response_format": None}

        kwargs = {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "streaming": self.streaming,
            "top_p": self.top_p,
            "openai_api_key": self.api_key.get_secret_value(),
            "model_kwargs": model_kwargs,
        }

        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.organization:
            kwargs["organization"] = self.organization

        return ChatOpenAI(**kwargs)
