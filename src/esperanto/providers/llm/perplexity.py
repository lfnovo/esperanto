"""Perplexity AI language model implementation."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI

from esperanto.common_types import Model
from esperanto.providers.llm.base import LanguageModel  # Import the base class
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger


@dataclass
class PerplexityLanguageModel(OpenAILanguageModel):
    """Perplexity AI language model implementation using OpenAI-compatible API."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    search_domain_filter: Optional[List[str]] = field(default=None)
    return_images: Optional[bool] = field(default=None)
    return_related_questions: Optional[bool] = field(default=None)
    search_recency_filter: Optional[str] = field(default=None)
    web_search_options: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        # Initialize Perplexity-specific configuration
        self.base_url = self.base_url or os.getenv(
            "PERPLEXITY_BASE_URL", "https://api.perplexity.ai/chat/completions"
        )
        self.api_key = self.api_key or os.getenv("PERPLEXITY_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Perplexity API key not found. Set the PERPLEXITY_API_KEY environment variable."
            )
        # Ensure api_key is not None after fetching from env
        assert self.api_key is not None, "PERPLEXITY_API_KEY must be set"
        # Ensure base_url is not None after fetching from env or default
        assert self.base_url is not None, "Base URL could not be determined"

        # Call grandparent's post_init to handle config, skipping OpenAI's __post_init__
        LanguageModel.__post_init__(self)

        # Initialize OpenAI clients with Perplexity configuration
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
        """Get kwargs for API calls, including Perplexity-specific args."""
        # Start with kwargs from the parent class (OpenAI)
        kwargs = super()._get_api_kwargs(exclude_stream=exclude_stream)

        # Add Perplexity-specific parameters if they are set
        if self.search_domain_filter is not None:
            kwargs["search_domain_filter"] = self.search_domain_filter
        if self.return_images is not None:
            kwargs["return_images"] = self.return_images
        if self.return_related_questions is not None:
            kwargs["return_related_questions"] = self.return_related_questions
        if self.search_recency_filter is not None:
            kwargs["search_recency_filter"] = self.search_recency_filter
        if self.web_search_options is not None:
            kwargs["web_search_options"] = self.web_search_options

        # Perplexity supports response_format, so no need to remove it like in XAI

        return kwargs

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.
        Note: Perplexity API docs don't specify a models endpoint.
        Hardcoding based on known models from docs.
        """
        # TODO: Check if Perplexity adds a models endpoint later
        known_models = [
            "sonar-small-chat",
            "sonar-small-online",
            "sonar-medium-chat",
            "sonar-medium-online",
            "codellama-70b-instruct",
            "llama-3-sonar-small-32k-chat",
            "llama-3-sonar-small-32k-online",
            "llama-3-sonar-large-32k-chat",
            "llama-3-sonar-large-32k-online",
            "llama-3-8b-instruct",
            "llama-3-70b-instruct",
            "mixtral-8x7b-instruct",
        ]
        return [
            Model(
                id=model_id,
                owned_by="Perplexity",
                context_window=None,  # Context window info not readily available
                type="language",
            )
            for model_id in known_models
        ]

    def _get_default_model(self) -> str:
        """Get the default model name."""
        # Using sonar-medium-online as a reasonable default with web access
        return "llama-3-sonar-large-32k-online"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "perplexity"

    def to_langchain(self) -> ChatOpenAI:
        """Convert to a LangChain chat model."""

        model_kwargs: Dict[str, Any] = {}
        if self.structured and isinstance(self.structured, dict):
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                model_kwargs["response_format"] = {"type": "json_object"}

        # Add Perplexity-specific parameters to model_kwargs
        if self.search_domain_filter is not None:
            model_kwargs["search_domain_filter"] = self.search_domain_filter
        if self.return_images is not None:
            model_kwargs["return_images"] = self.return_images
        if self.return_related_questions is not None:
            model_kwargs["return_related_questions"] = self.return_related_questions
        if self.search_recency_filter is not None:
            model_kwargs["search_recency_filter"] = self.search_recency_filter
        if self.web_search_options is not None:
            model_kwargs["web_search_options"] = self.web_search_options

        langchain_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "organization": self.organization,
            "model": self.get_model_name(),
            "model_kwargs": model_kwargs,
        }

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
