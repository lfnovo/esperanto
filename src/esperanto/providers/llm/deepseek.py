"""DeepSeek language model implementation."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from esperanto.common_types import Model
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger

if TYPE_CHECKING:
    from langchain_deepseek import ChatDeepSeek


@dataclass
class DeepSeekLanguageModel(OpenAILanguageModel):
    """DeepSeek language model implementation using OpenAI-compatible API."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: str = "deepseek-chat"

    @property
    def provider(self) -> str:
        return "deepseek"

    def __post_init__(self):
        # Extract api_key and base_url from config dict first (before parent sets OpenAI defaults)
        if hasattr(self, "config") and self.config:
            if "api_key" in self.config:
                self.api_key = self.config["api_key"]
            if "base_url" in self.config:
                self.base_url = self.config["base_url"]

        # Initialize DeepSeek-specific configuration
        self.base_url = self.base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model_name = self.model_name or "deepseek-chat"

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Set the DEEPSEEK_API_KEY environment variable."
            )

        # Call parent's post_init (won't overwrite since values are already set)
        super().__post_init__()

        # DeepSeek supports JSON mode like OpenAI (handled by parent)
        # If any DeepSeek-specific warnings or configs are needed, add here
