import os
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama.chat_models import ChatOllama

from esperanto.base.types import LanguageModel


class OllamaLanguageModel(LanguageModel):
    """Ollama language model implementation."""

    def __init__(self, model_name: str, config: Dict[str, Any] = {}):
        super().__init__(model_name, config)
        self.max_tokens = self.config.get("max_tokens", 850)
        self.temperature = self.config.get("temperature", 1.0)
        self.streaming = self.config.get("streaming", False)
        self.top_p = self.config.get("top_p", 0.9)
        self.base_url = self.config.get("base_url") or os.environ.get(
            "OLLAMA_API_BASE", "http://localhost:11434"
        )

    @property
    def provider(self) -> str:
        return "ollama"

    def validate_config(self) -> None:
        """Validate Ollama configuration."""
        if not self.model_name:
            raise ValueError("model_name must be specified for Ollama language model")
        if not self.base_url or self.base_url.strip() == "":
            raise ValueError(
                "base_url must be specified in config or OLLAMA_API_BASE environment variable"
            )

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            stream=self.streaming,  # ChatOllama uses stream instead of streaming
            base_url=self.base_url,
        )
