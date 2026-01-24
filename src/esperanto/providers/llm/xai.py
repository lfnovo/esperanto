"""XAI language model implementation."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from esperanto.common_types import Model
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger

if TYPE_CHECKING:
    from langchain_xai import ChatXAI
    from langchain_openai import ChatOpenAI


@dataclass
class XAILanguageModel(OpenAILanguageModel):
    """XAI language model implementation using OpenAI-compatible API."""

    base_url: Optional[str] = None  # Changed type hint
    api_key: Optional[str] = None  # Changed type hint

    def __post_init__(self):
        # Extract api_key and base_url from config dict first (before parent sets OpenAI defaults)
        if hasattr(self, "config") and self.config:
            if "api_key" in self.config:
                self.api_key = self.config["api_key"]
            if "base_url" in self.config:
                self.base_url = self.config["base_url"]

        # Initialize XAI-specific configuration
        self.base_url = self.base_url or os.getenv(
            "XAI_BASE_URL", "https://api.x.ai/v1"
        )
        self.api_key = self.api_key or os.getenv("XAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "XAI API key not found. Set the XAI_API_KEY environment variable."
            )

        # Call parent's post_init (won't overwrite since values are already set)
        super().__post_init__()


    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args.

        Note: XAI doesn't support response_format parameter.
        """
        kwargs = super()._get_api_kwargs(exclude_stream)

        # Remove response_format as XAI doesn't support it
        kwargs.pop("response_format", None)

        return kwargs

    def _get_models(self) -> List[Model]:
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

    def to_langchain(self) -> Union["ChatXAI", "ChatOpenAI"]:
        """Convert to a LangChain chat model.

        Prefers the dedicated langchain-xai package if installed,
        falls back to langchain-openai with a deprecation warning.

        Raises:
            ImportError: If neither langchain_xai nor langchain_openai is installed.
        """
        import warnings

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")

        # Try dedicated xAI package first
        try:
            from langchain_xai import ChatXAI

            langchain_kwargs = {
                "model": model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "xai_api_key": self.api_key,
            }
            return ChatXAI(**self._clean_config(langchain_kwargs))

        except ImportError:
            pass

        # Fall back to langchain-openai with deprecation warning
        try:
            from langchain_openai import ChatOpenAI

            warnings.warn(
                "Using langchain-openai for xAI is deprecated and will be removed in Esperanto 3.0. "
                "Install langchain-xai for better support: pip install langchain-xai",
                DeprecationWarning,
                stacklevel=2,
            )

            langchain_kwargs = {
                "model": model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "streaming": self.streaming,
                "api_key": self.api_key,
                "base_url": self.base_url,
                "organization": self.organization,
                "model_kwargs": {},
            }
            return ChatOpenAI(**self._clean_config(langchain_kwargs))

        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain-xai (recommended) or langchain-openai. "
                "Install with: pip install langchain-xai"
            ) from e
