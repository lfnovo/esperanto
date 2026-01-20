"""Perplexity AI language model implementation."""

import json
import os
from dataclasses import dataclass, field

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
    FunctionCall,
    Message,
    Model,
    StreamChoice,
    Tool,
    ToolCall,
    Usage,
)
from esperanto.common_types.validation import (
    validate_tool_calls as _validate_tool_calls,
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

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

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
                # Tool-related fields are handled separately in chat_complete()
                "tools",
                "tool_choice",
                "parallel_tool_calls",
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

    def _convert_tools_to_openai(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Esperanto tools to OpenAI format.

        Note: Perplexity's tool support is uncertain. This method is provided
        for interface consistency.

        Args:
            tools: List of Esperanto Tool objects.

        Returns:
            List of tools in OpenAI API format, or None if no tools provided.
        """
        if not tools:
            return None
        result = []
        for tool in tools:
            tool_dict: Dict[str, Any] = {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                },
            }
            result.append(tool_dict)
        return result

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
        choices = []
        for choice in response_data["choices"]:
            message_data = choice.get("message", {})

            # Extract tool_calls if present (Perplexity may support this in future)
            tool_calls = None
            if "tool_calls" in message_data and message_data["tool_calls"]:
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in message_data["tool_calls"]
                ]

            choices.append(
                Choice(
                    index=choice["index"],
                    message=Message(
                        content=message_data.get("content") or "",
                        role=message_data.get("role", "assistant"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            )

        return ChatCompletion(
            id=response_data["id"],
            choices=choices,
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
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request.

        Note: Perplexity's tool support is uncertain. The tool parameters are
        included for interface consistency but may not work.

        Args:
            messages: List of messages in the conversation. Messages can include
                tool call results with role="tool" and tool_call_id.
            stream: Whether to stream the response. If None, uses the instance's
                streaming setting.
            tools: List of tools the model can call. If None, uses instance tools.
            tool_choice: Controls tool usage. Values:
                - "auto": Model decides whether to call tools (default)
                - "required": Model must call at least one tool
                - "none": Model cannot call tools
                - {"type": "function", "function": {"name": "..."}}: Force specific tool
            parallel_tool_calls: Whether to allow multiple tool calls in one response.
                None uses provider default (usually True). Set False to force single
                tool call per response.
            validate_tool_calls: If True, validate tool call arguments against the
                tool's JSON schema. Raises ToolCallValidationError on validation
                failure. Requires jsonschema package.

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks
            if streaming. When the model calls tools, the response message will
            have tool_calls populated.
        """
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        api_kwargs = self._get_api_kwargs(exclude_stream=True)
        perplexity_params = self._get_perplexity_params()

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        resolved_parallel = self._resolve_parallel_tool_calls(parallel_tool_calls)

        # Prepare request payload with Perplexity-specific parameters
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": should_stream,
            **api_kwargs,
            **perplexity_params,  # Add Perplexity params directly to payload
        }

        # Add tool-related parameters if configured
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_openai(resolved_tools)
        if resolved_tool_choice is not None:
            payload["tool_choice"] = resolved_tool_choice
        if resolved_parallel is not None:
            payload["parallel_tool_calls"] = resolved_parallel

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
        result = self._normalize_response(response_data)

        # Validate tool calls if requested
        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result

    async def achat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request.

        Note: Perplexity's tool support is uncertain. The tool parameters are
        included for interface consistency but may not work.

        Args:
            messages: List of messages in the conversation. Messages can include
                tool call results with role="tool" and tool_call_id.
            stream: Whether to stream the response. If None, uses the instance's
                streaming setting.
            tools: List of tools the model can call. If None, uses instance tools.
            tool_choice: Controls tool usage. Values:
                - "auto": Model decides whether to call tools (default)
                - "required": Model must call at least one tool
                - "none": Model cannot call tools
                - {"type": "function", "function": {"name": "..."}}: Force specific tool
            parallel_tool_calls: Whether to allow multiple tool calls in one response.
                None uses provider default (usually True). Set False to force single
                tool call per response.
            validate_tool_calls: If True, validate tool call arguments against the
                tool's JSON schema. Raises ToolCallValidationError on validation
                failure. Requires jsonschema package.

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks
            if streaming. When the model calls tools, the response message will
            have tool_calls populated.
        """
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        api_kwargs = self._get_api_kwargs(exclude_stream=True)
        perplexity_params = self._get_perplexity_params()

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        resolved_parallel = self._resolve_parallel_tool_calls(parallel_tool_calls)

        # Prepare request payload with Perplexity-specific parameters
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": should_stream,
            **api_kwargs,
            **perplexity_params,  # Add Perplexity params directly to payload
        }

        # Add tool-related parameters if configured
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_openai(resolved_tools)
        if resolved_tool_choice is not None:
            payload["tool_choice"] = resolved_tool_choice
        if resolved_parallel is not None:
            payload["parallel_tool_calls"] = resolved_parallel

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
        result = self._normalize_response(response_data)

        # Validate tool calls if requested
        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result

    def _get_models(self) -> List[Model]:
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

        # Pass SSL-configured httpx clients to LangChain
        # This ensures SSL verification settings are respected
        # Only pass if they are real httpx clients (not mocks from tests)
        try:
            if hasattr(self, "client") and isinstance(self.client, httpx.Client):
                langchain_kwargs["http_client"] = self.client
            if hasattr(self, "async_client") and isinstance(self.async_client, httpx.AsyncClient):
                langchain_kwargs["http_async_client"] = self.async_client
        except TypeError:
            # httpx types might be mocked in tests, skip passing clients
            pass

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")
        langchain_kwargs["model"] = model_name  # Update model name in kwargs

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
