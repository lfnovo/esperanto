"""XAI language model implementation."""

import os
from dataclasses import dataclass
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI

from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger


@dataclass
class XAILanguageModel(OpenAILanguageModel):
    """XAI language model implementation using OpenAI-compatible API."""

    base_url: str = None
    api_key: str = None

    def __post_init__(self):
        # Initialize XAI-specific configuration
        self.base_url = self.base_url or os.getenv(
            "XAI_BASE_URL", "https://api.x.ai/v1"
        )
        self.api_key = self.api_key or os.getenv("XAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "XAI API key not found. Set the XAI_API_KEY environment variable."
            )

        # Call parent's post_init to set up normalized response handling
        super().__post_init__()

        if self.structured:
            logger.warning("Structured output not supported for X.AI.")

        # Initialize OpenAI clients with XAI configuration
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
        )

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args.

        Note: XAI doesn't support response_format parameter.
        """
        kwargs = super()._get_api_kwargs(exclude_stream)

        # Remove response_format as XAI doesn't support it
        kwargs.pop("response_format", None)

        return kwargs

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "grok-beta"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "xai"

    def to_langchain(self) -> ChatOpenAI:
        """Convert to a LangChain chat model."""

        langchain_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "organization": self.organization,
            "model": self.get_model_name(),
            "model_kwargs": {},  # XAI doesn't support response_format
        }

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
