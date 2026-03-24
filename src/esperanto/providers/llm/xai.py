"""XAI language model implementation."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # Added Optional

from esperanto.common_types import Model
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.llm.structured_output import (
    ResolvedStructuredOutput,
    resolve_structured_output,
)
from esperanto.utils.logging import logger

if TYPE_CHECKING:
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


    def _get_api_kwargs(
        self,
        exclude_stream: bool = False,
        resolved_structured: Optional[ResolvedStructuredOutput] = None,
    ) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args.

        For legacy json/json_object, keep previous behavior and avoid sending
        response_format. For schema mode, pass through response_format and let
        upstream support determine success/failure (fail-fast).
        """
        kwargs = super()._get_api_kwargs(
            exclude_stream,
            resolved_structured=resolved_structured,
        )

        if resolved_structured is None:
            resolved_structured = resolve_structured_output(
                self.structured,
                allow_string_json_alias=True,
            )

        # Keep fail-fast pass-through for schema mode. Preserve legacy behavior
        # of removing response_format for json/json_object.
        if "response_format" in kwargs and not (
            resolved_structured and resolved_structured.is_schema_mode
        ):
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
            "model_kwargs": {},
        }

        resolved_structured = resolve_structured_output(
            self.structured,
            allow_string_json_alias=True,
        )
        if resolved_structured and resolved_structured.is_schema_mode:
            langchain_kwargs["model_kwargs"]["response_format"] = (
                resolved_structured.response_format
            )

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")
        langchain_kwargs["model"] = model_name  # Update model name in kwargs

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
