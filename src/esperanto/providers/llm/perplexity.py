"""Perplexity AI language model implementation."""

import json
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

import httpx

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    Message,
    Model,
    StreamChoice,
    Usage,
)
from esperanto.providers.llm.base import LanguageModel
from esperanto.utils.logging import logger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@dataclass
class PerplexityLanguageModel(LanguageModel):
    """Perplexity AI language model implementation using httpx."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    search_domain_filter: Optional[List[str]] = field(default=None)
    return_images: Optional[bool] = field(default=None)
    return_related_questions: Optional[bool] = field(default=None)
    search_recency_filter: Optional[str] = field(default=None)
    web_search_options: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Initialize Perplexity-specific configuration
        self.base_url = self.base_url or os.getenv(
            "PERPLEXITY_BASE_URL", "https://api.perplexity.ai"
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

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Perplexity API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Perplexity API error: {error_message}")

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse Server-Sent Events stream from Perplexity chat completions."""
        for chunk in response.iter_text():
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    async def _parse_sse_stream_async(self, response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
        """Parse Server-Sent Events stream from Perplexity chat completions asynchronously."""
        async for chunk in response.aiter_text():
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip() == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = {}
        config = self.get_completion_kwargs()

        # Only include non-provider-specific args that were explicitly set
        for key, value in config.items():
            if key not in [
                "model_name",
                "api_key",
                "base_url",
                "organization",
                "structured",
            ]:
                # Skip max_tokens if it's the default value (850)
                if key == "max_tokens" and value == 850:
                    continue
                kwargs[key] = value

        # Handle streaming parameter
        if exclude_stream:
            kwargs.pop("streaming", None)
        elif "streaming" in kwargs:
            kwargs["stream"] = kwargs.pop("streaming")

        # Handle structured output - Perplexity uses "text" for JSON
        if self.structured:
            if not isinstance(self.structured, dict):
                raise TypeError("structured parameter must be a dictionary")
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                kwargs["response_format"] = {"type": "text"}

        return kwargs

    def _get_perplexity_params(self) -> Dict[str, Any]:
        """Get Perplexity-specific parameters."""
        params: Dict[str, Any] = {}
        if self.search_domain_filter is not None:
            params["search_domain_filter"] = self.search_domain_filter
        if self.return_images is not None:
            params["return_images"] = self.return_images
        if self.return_related_questions is not None:
            params["return_related_questions"] = self.return_related_questions
        if self.search_recency_filter is not None:
            params["search_recency_filter"] = self.search_recency_filter
        if self.web_search_options is not None:
            params["web_search_options"] = self.web_search_options
        return params

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Perplexity response to our format."""
        return ChatCompletion(
            id=response_data["id"],
            choices=[
                Choice(
                    index=choice["index"],
                    message=Message(
                        content=choice["message"]["content"] or "",
                        role=choice["message"]["role"],
                    ),
                    finish_reason=choice["finish_reason"],
                )
                for choice in response_data["choices"]
            ],
            created=response_data["created"],
            model=response_data["model"],
            provider=self.provider,
            usage=Usage(
                completion_tokens=response_data.get("usage", {}).get("completion_tokens", 0),
                prompt_tokens=response_data.get("usage", {}).get("prompt_tokens", 0),
                total_tokens=response_data.get("usage", {}).get("total_tokens", 0),
            ),
        )

    def _normalize_chunk(self, chunk_data: Dict[str, Any]) -> ChatCompletionChunk:
        """Normalize Perplexity stream chunk to our format."""
        return ChatCompletionChunk(
            id=chunk_data["id"],
            choices=[
                StreamChoice(
                    index=choice["index"],
                    delta=DeltaMessage(
                        content=choice.get("delta", {}).get("content", ""),
                        role=choice.get("delta", {}).get("role", "assistant"),
                        function_call=choice.get("delta", {}).get("function_call"),
                        tool_calls=choice.get("delta", {}).get("tool_calls"),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
                for choice in chunk_data["choices"]
            ],
            created=chunk_data["created"],
            model=chunk_data.get("model", ""),
        )

    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request, including Perplexity params."""
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        api_kwargs = self._get_api_kwargs(exclude_stream=True)
        perplexity_params = self._get_perplexity_params()

        # Prepare request payload with Perplexity-specific parameters
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": should_stream,
            **api_kwargs,
            **perplexity_params,  # Add Perplexity params directly to payload
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            return (self._normalize_chunk(chunk_data) for chunk_data in self._parse_sse_stream(response))
        
        response_data = response.json()
        return self._normalize_response(response_data)

    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request, including Perplexity params."""
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        api_kwargs = self._get_api_kwargs(exclude_stream=True)
        perplexity_params = self._get_perplexity_params()

        # Prepare request payload with Perplexity-specific parameters
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": should_stream,
            **api_kwargs,
            **perplexity_params,  # Add Perplexity params directly to payload
        }

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            async def generate():
                async for chunk_data in self._parse_sse_stream_async(response):
                    yield self._normalize_chunk(chunk_data)

            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.
        Note: Perplexity API docs don't specify a models endpoint.
        Hardcoding based on known models from docs.
        """
        # TODO: Check if Perplexity adds a models endpoint later
        known_models = [
            "sonar-deep-research",
            "sonar-reasoning-pro",
            "sonar-reasoning",
            "sonar-pro",
            "sonar",
            "r1-1776",
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
                "Install with: uv add langchain_openai or pip install langchain_openai"
            ) from e

        model_kwargs: Dict[str, Any] = {}
        if self.structured and isinstance(self.structured, dict):
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                model_kwargs["response_format"] = {"type": "text"}

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

        return ChatOpenAI(**langchain_kwargs)
