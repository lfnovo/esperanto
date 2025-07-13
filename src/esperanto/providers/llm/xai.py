"""XAI language model implementation."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # Added Optional

from openai import AsyncOpenAI, OpenAI

from esperanto.common_types import Model
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@dataclass
class XAILanguageModel(OpenAILanguageModel):
    """XAI language model implementation using OpenAI-compatible API."""

    base_url: Optional[str] = None  # Changed type hint
    api_key: Optional[str] = None  # Changed type hint

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

        # Call parent's post_init to set up HTTP clients
        super().__post_init__()


    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args.

        Note: XAI doesn't support response_format parameter.
        """
        kwargs = super()._get_api_kwargs(exclude_stream)

        # Remove response_format as XAI doesn't support it
        kwargs.pop("response_format", None)

        return kwargs

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        response = self.client.get(
            f"{self.base_url}/models",
            headers=self._get_headers()
        )
        self._handle_error(response)
        
        models_data = response.json()
        return [
            Model(
                id=model["id"],
                owned_by="X.AI",
                context_window=model.get("context_window", None),
                type="language",
            )
            for model in models_data["data"]
            if model["id"].startswith("grok")  # Only include Grok models
        ]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "grok-2-latest"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "xai"

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
            "api_key": self.api_key,  # Pass raw string
            "base_url": self.base_url,
            "organization": self.organization,
            "model": self.get_model_name(),
            "model_kwargs": {},  # XAI doesn't support response_format
        }

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")
        langchain_kwargs["model"] = model_name  # Update model name in kwargs

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
