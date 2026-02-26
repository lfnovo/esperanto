"""Zaps.ai language model implementation."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from esperanto.common_types import Model
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@dataclass
class ZapsLanguageModel(OpenAILanguageModel):
    """Zaps.ai language model implementation using OpenAI-compatible API.

    Zaps.ai acts as a privacy gateway — requests pass through Zaps for PII
    redaction before being forwarded to the upstream provider.
    """

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: str = "gpt-4o"

    @property
    def provider(self) -> str:
        return "zaps"

    def __post_init__(self):
        # Extract api_key and base_url from config dict first (before parent sets OpenAI defaults)
        if hasattr(self, "config") and self.config:
            if "api_key" in self.config:
                self.api_key = self.config["api_key"]
            if "base_url" in self.config:
                self.base_url = self.config["base_url"]

        # Initialize Zaps-specific configuration
        self.base_url = self.base_url or os.getenv(
            "ZAPS_BASE_URL", "https://api.zaps.ai/v1"
        )
        self.api_key = self.api_key or os.getenv("ZAPS_API_KEY")
        self.model_name = self.model_name or "gpt-4o"

        if not self.api_key:
            raise ValueError(
                "Zaps API key not found. Set the ZAPS_API_KEY environment variable."
            )

        # Call parent's post_init (won't overwrite since values are already set)
        super().__post_init__()

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "gpt-4o"

    def to_langchain(self) -> "ChatOpenAI":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_openai is not installed.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_openai. "
                "Install with: uv add langchain_openai or pip install langchain_openai"
            ) from e

        langchain_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "organization": self.organization,
            "model": self.get_model_name(),
            "model_kwargs": {},
        }

        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")
        langchain_kwargs["model"] = model_name

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
