"""OpenRouter language model implementation."""

import json
import os
from dataclasses import dataclass
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

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Model,
    Tool,
)
from esperanto.common_types.validation import (
    validate_tool_calls as _validate_tool_calls,
)
from esperanto.providers.llm.openai import OpenAILanguageModel

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@dataclass
class OpenRouterLanguageModel(OpenAILanguageModel):
    """OpenRouter language model implementation using OpenAI-compatible API."""

    base_url: Optional[str] = None  # Changed type hint
    api_key: Optional[str] = None  # Changed type hint

    def __post_init__(self):
        # Extract api_key and base_url from config dict first (before parent sets OpenAI defaults)
        if hasattr(self, "config") and self.config:
            if "api_key" in self.config:
                self.api_key = self.config["api_key"]
            if "base_url" in self.config:
                self.base_url = self.config["base_url"]

        # Initialize OpenRouter-specific configuration
        self.base_url = self.base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable."
            )

        # Call parent's post_init (won't overwrite since values are already set)
        super().__post_init__()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests with required headers."""
        headers = super()._get_headers()
        # Add OpenRouter-specific required headers
        headers.update({
            "HTTP-Referer": "https://github.com/lfnovo/esperanto",
            "X-Title": "Esperanto",
        })
        return headers

    def _handle_error(self, response) -> None:
        """Handle HTTP error responses with detailed OpenRouter logging."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception as e:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI API error: {error_message}")

    def _make_http_request(self, payload: Dict[str, Any]) -> Any:
        """Make HTTP request in OpenRouter's expected format."""
        # OpenRouter expects data as JSON string, not json parameter
        headers = self._get_headers()
        
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload)  # Use data= instead of json=
        )
        self._handle_error(response)
        return response

    async def _make_async_http_request(self, payload: Dict[str, Any]) -> Any:
        """Make async HTTP request in OpenRouter's expected format."""
        # OpenRouter expects data as JSON string, not json parameter
        response = await self.async_client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            data=json.dumps(payload)  # Use data= instead of json=
        )
        self._handle_error(response)
        return response

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args.

        Note: OpenRouter doesn't support JSON response format for non-OpenAI models.
        """
        kwargs = super()._get_api_kwargs(exclude_stream)

        # Remove response_format for non-OpenAI models
        model = self.get_model_name().lower()
        if "response_format" in kwargs and not model.startswith(("openai/", "gpt-")):
            kwargs.pop("response_format")

        return kwargs

    def chat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request using OpenRouter-specific HTTP format.

        Args:
            messages: List of messages in the conversation. Messages can include
                tool call results with role="tool" and tool_call_id.
            stream: Whether to stream the response. If None, uses the instance's
                streaming setting.
            tools: List of tools the model can call. If None, uses instance tools.
                Note: Tool support depends on the underlying model in OpenRouter.
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
        # Warn if validate_tool_calls is used with streaming
        self._warn_if_validate_with_streaming(validate_tool_calls, stream)

        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        is_reasoning_model = model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4")

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        resolved_parallel = self._resolve_parallel_tool_calls(parallel_tool_calls)

        # Transform messages for o1 models
        if is_reasoning_model:
            messages = self._transform_messages_for_o1(
                [{**msg} for msg in messages]
            )

        # Prepare request payload
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": should_stream,
            **self._get_api_kwargs(exclude_stream=True),
        }

        # Add tool-related parameters if configured
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_openai(resolved_tools)
        if resolved_tool_choice is not None:
            payload["tool_choice"] = resolved_tool_choice
        if resolved_parallel is not None:
            payload["parallel_tool_calls"] = resolved_parallel

        # Make HTTP request using OpenRouter format
        response = self._make_http_request(payload)

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
        """Send an async chat completion request using OpenRouter-specific HTTP format.

        Args:
            messages: List of messages in the conversation. Messages can include
                tool call results with role="tool" and tool_call_id.
            stream: Whether to stream the response. If None, uses the instance's
                streaming setting.
            tools: List of tools the model can call. If None, uses instance tools.
                Note: Tool support depends on the underlying model in OpenRouter.
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
        # Warn if validate_tool_calls is used with streaming
        self._warn_if_validate_with_streaming(validate_tool_calls, stream)

        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        is_reasoning_model = model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4")

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        resolved_parallel = self._resolve_parallel_tool_calls(parallel_tool_calls)

        # Transform messages for o1 models
        if is_reasoning_model:
            messages = self._transform_messages_for_o1(
                [{**msg} for msg in messages]
            )

        # Prepare request payload
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": should_stream,
            **self._get_api_kwargs(exclude_stream=True),
        }

        # Add tool-related parameters if configured
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_openai(resolved_tools)
        if resolved_tool_choice is not None:
            payload["tool_choice"] = resolved_tool_choice
        if resolved_parallel is not None:
            payload["parallel_tool_calls"] = resolved_parallel

        # Make async HTTP request using OpenRouter format
        response = await self._make_async_http_request(payload)

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

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "anthropic/claude-2"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openrouter"

    def _get_models(self) -> List[Model]:
        """List all available models for this provider."""
        headers = self._get_headers()
        
        response = self.client.get(
            f"{self.base_url}/models",
            headers=headers
        )
        self._handle_error(response)
        
        models_data = response.json()
        return [
            Model(
                id=model["id"],
                owned_by=model["id"].split("/")[0] if "/" in model["id"] else "OpenRouter",
                context_window=model.get("context_window", None),
            )
            for model in models_data["data"]
            if not any(
                model["id"].startswith(prefix)
                for prefix in [
                    "text-embedding",  # Exclude embedding models
                    "whisper",  # Exclude speech models
                    "tts",  # Exclude text-to-speech models
                ]
            )
        ]

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

        model_kwargs = {}
        if self.structured and isinstance(self.structured, dict):
            structured_type = self.structured.get("type")
            if structured_type in [
                "json",
                "json_object",
            ] and self.get_model_name().lower().startswith(("openai/", "gpt-")):
                model_kwargs["response_format"] = {"type": "json_object"}

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
            "default_headers": {
                "HTTP-Referer": "https://github.com/lfnovo/esperanto",  # Required by OpenRouter
                "X-Title": "Esperanto",  # Required by OpenRouter
            },
        }

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")
        langchain_kwargs["model"] = model_name  # Update model name in kwargs

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
