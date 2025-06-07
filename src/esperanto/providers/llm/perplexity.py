"""Perplexity AI language model implementation."""

import os
from dataclasses import dataclass, field

# Add Union, Generator, AsyncGenerator, TYPE_CHECKING to imports
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

from esperanto.utils.openai_http import AsyncOpenAIHTTPClient, OpenAIHTTPClient

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Model,
)  # Import necessary types
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


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

        # Initialize HTTP clients with Perplexity configuration
        self.client = OpenAIHTTPClient(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
        )
        self.async_client = AsyncOpenAIHTTPClient(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
        )

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get standard API kwargs compatible with the openai library signature."""
        # Get kwargs from the parent class (OpenAI), which handles standard params
        # This ensures we only include args the 'create' method signature expects.
        kwargs = super()._get_api_kwargs(exclude_stream=exclude_stream)
        return kwargs

    def _get_perplexity_extra_body(self) -> Dict[str, Any]:
        """Get Perplexity-specific parameters for the 'extra_body'."""
        # Initialize directly as Dict[str, Any]
        extra_body: Dict[str, Any] = {}
        if self.search_domain_filter is not None:
            extra_body["search_domain_filter"] = self.search_domain_filter
        if self.return_images is not None:
            extra_body["return_images"] = self.return_images
        if self.return_related_questions is not None:
            extra_body["return_related_questions"] = self.return_related_questions
        if self.search_recency_filter is not None:
            extra_body["search_recency_filter"] = self.search_recency_filter
        if self.web_search_options is not None:
            extra_body["web_search_options"] = self.web_search_options
        return extra_body

    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request, including Perplexity params."""
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        api_kwargs = self._get_api_kwargs(exclude_stream=True)
        extra_body = self._get_perplexity_extra_body()

        # Note: Perplexity doesn't seem to have special message transformations like o1
        response = self.client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=should_stream,
            extra_body=extra_body,  # Pass Perplexity params here
            **api_kwargs,
        )

        if should_stream:
            return (self._normalize_chunk(chunk) for chunk in response)
        return self._normalize_response(response)

    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request, including Perplexity params."""
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        api_kwargs = self._get_api_kwargs(exclude_stream=True)
        extra_body = self._get_perplexity_extra_body()

        # Note: Perplexity doesn't seem to have special message transformations like o1
        response = await self.async_client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=should_stream,
            extra_body=extra_body,  # Pass Perplexity params here
            **api_kwargs,
        )

        if should_stream:

            async def generate():
                async for chunk in response:
                    yield self._normalize_chunk(chunk)

            return generate()
        return self._normalize_response(response)

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
                "Install with: uv add esperanto[perplexity,langchain] or pip install esperanto[perplexity,langchain]"
            ) from e

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
            "api_key": self.api_key,  # Pass raw string
            "base_url": self.base_url,
            "organization": self.organization,
            "model": self.get_model_name(),
            "model_kwargs": model_kwargs,
        }

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")
        langchain_kwargs["model"] = model_name  # Update model name in kwargs

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
