"""Groq language model provider."""

import json
import os
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

if TYPE_CHECKING:
    from langchain_groq import ChatGroq


class GroqLanguageModel(LanguageModel):
    """Groq language model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Get API key
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found")

        # Set base URL for Groq's OpenAI-compatible API
        self.base_url = "https://api.groq.com/openai/v1"

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Groq API requests."""
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
            raise RuntimeError(f"Groq API error: {error_message}")

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
                owned_by="Groq",
                context_window=128000,  # All Groq models currently support 128k context
            )
            for model in models_data["data"]
        ]

    def _convert_tools_to_openai(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Esperanto tools to OpenAI/Groq format.

        Args:
            tools: List of Esperanto Tool objects.

        Returns:
            List of tools in OpenAI/Groq API format, or None if no tools provided.
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
            # Add strict mode if specified (OpenAI-specific, but include for compat)
            if tool.function.strict is not None:
                tool_dict["function"]["strict"] = tool.function.strict
            result.append(tool_dict)
        return result

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Groq response to our format."""
        choices = []
        for choice in response_data["choices"]:
            message_data = choice.get("message", {})

            # Extract tool_calls if present
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
        """Normalize Groq stream chunk to our format."""
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

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse Server-Sent Events stream from Groq chat completions."""
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
        """Parse Server-Sent Events stream from Groq chat completions asynchronously."""
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

        # Handle structured output
        if self.structured:
            if not isinstance(self.structured, dict):
                raise TypeError("structured parameter must be a dictionary")
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                kwargs["response_format"] = {"type": "json_object"}

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
        """Send a chat completion request.

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
        # Warn if validate_tool_calls is used with streaming
        self._warn_if_validate_with_streaming(validate_tool_calls, stream)

        should_stream = stream if stream is not None else self.streaming

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        resolved_parallel = self._resolve_parallel_tool_calls(parallel_tool_calls)

        # Prepare request payload
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
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
        # Warn if validate_tool_calls is used with streaming
        self._warn_if_validate_with_streaming(validate_tool_calls, stream)

        should_stream = stream if stream is not None else self.streaming

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        resolved_parallel = self._resolve_parallel_tool_calls(parallel_tool_calls)

        # Prepare request payload
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
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

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "mixtral-8x7b-32768"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "groq"

    def to_langchain(self) -> "ChatGroq":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_groq is not installed.
        """
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_groq. "
                "Install with: uv add langchain_groq or pip install langchain_groq"
            ) from e

        # SecretStr import removed, rely on ChatGroq internal handling

        # Ensure model name is a string
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name must be set to use Langchain integration.")

        langchain_kwargs = {
            "model": model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # top_p=self.top_p, # Still not supported
            "streaming": self.streaming,
            "api_key": self.api_key,  # Pass the raw API key string
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

        return ChatGroq(**self._clean_config(langchain_kwargs))
