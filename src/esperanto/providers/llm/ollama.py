"""Ollama language model provider."""

import json
import os
import time
import uuid
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
    from langchain_ollama import ChatOllama


class OllamaLanguageModel(LanguageModel):
    """Ollama language model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Set default base URL if not provided
        self.base_url = (
            self.base_url or os.getenv("OLLAMA_BASE_URL")  or os.getenv("OLLAMA_API_BASE") or "http://localhost:11434"
        )

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Ollama API requests."""
        return {
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Ollama API error: {error_message}")

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = {}
        config = self.get_completion_kwargs()
        options = {}

        # Only include non-provider-specific args that were explicitly set
        for key, value in config.items():
            if key not in [
                "model_name",
                "base_url",
                "streaming",
                # Tool-related fields are handled separately in chat_complete()
                "tools",
                "tool_choice",
                "parallel_tool_calls",
            ]:
                if key in ["temperature", "top_p"]:
                    options[key] = value
                elif key == "max_tokens":
                    # Convert max_tokens to num_predict for Ollama
                    options["num_predict"] = value
                else:
                    kwargs[key] = value

        # Handle Ollama-specific options from _config (num_ctx for context window)
        # Default to 128000 tokens if not specified (Ollama's default of 2048 is too small)
        options["num_ctx"] = self._config.get("num_ctx", 128000)

        # Handle keep_alive (top-level parameter, not in options)
        # Only set if explicitly provided - don't force memory usage on users
        keep_alive = self._config.get("keep_alive")
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive

        # Handle JSON format if structured output is requested
        if self.structured:
            if not isinstance(self.structured, dict):
                raise TypeError("structured parameter must be a dictionary")
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                kwargs["format"] = "json"

        # Add options if any were set
        if options:
            kwargs["options"] = options

        return kwargs

    def _convert_tools_to_ollama(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Esperanto tools to Ollama format.

        Ollama uses the same tool format as OpenAI.

        Note: Not all Ollama models support tool calling. This method is provided
        for interface consistency, but behavior depends on the model.

        Args:
            tools: List of Esperanto Tool objects.

        Returns:
            List of tools in Ollama API format, or None if no tools provided.
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

    def _convert_messages_for_ollama(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert messages to Ollama's expected format.

        Ollama expects tool-related messages in a specific format:
        - Assistant messages with tool_calls need the tool_calls converted
        - Tool result messages need proper formatting

        Args:
            messages: List of messages in OpenAI-style format.

        Returns:
            List of messages in Ollama format.
        """
        converted = []
        for msg in messages:
            role = msg.get("role")

            if role == "assistant" and msg.get("tool_calls"):
                # Convert assistant message with tool_calls
                tool_calls = []
                for tc in msg["tool_calls"]:
                    # Parse arguments from JSON string back to dict for Ollama
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    tool_calls.append({
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": args,  # Ollama expects dict, not string
                        }
                    })

                converted.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": tool_calls,
                })

            elif role == "tool":
                # Convert tool result message
                # Ollama expects the content as a string (can be JSON string)
                content = msg.get("content", "")
                converted.append({
                    "role": "tool",
                    "content": content,
                })

            else:
                # Pass through other messages as-is
                converted.append(msg)

        return converted

    def _parse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse streaming response from Ollama."""
        for line in response.iter_lines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def _parse_stream_async(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse streaming response from Ollama asynchronously."""
        async for line in response.aiter_lines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

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

        Note: Not all Ollama models support tool calling. This method provides
        tool parameters for interface consistency, but behavior depends on the
        model being used.

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
                Note: Not all Ollama models support this parameter.
            validate_tool_calls: If True, validate tool call arguments against the
                tool's JSON schema. Raises ToolCallValidationError on validation
                failure. Requires jsonschema package.

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks
            if streaming. When the model calls tools, the response message will
            have tool_calls populated.
        """
        should_stream = stream if stream is not None else self.streaming

        if not messages:
            raise ValueError("Messages cannot be empty")

        # Validate message format - allow missing content for tool messages
        for message in messages:
            if "role" not in message:
                raise ValueError("Missing role in message")
            if message["role"] not in ["user", "assistant", "system", "tool"]:
                raise ValueError("Invalid role in message")
            # Allow messages without content if they have tool_calls (assistant) or are tool results
            if "content" not in message and message["role"] not in ["assistant", "tool"]:
                raise ValueError("Missing content in message")

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        # Note: parallel_tool_calls is resolved but Ollama may not support it
        _ = self._resolve_parallel_tool_calls(parallel_tool_calls)

        # Convert messages to Ollama format (handles tool-related message conversions)
        converted_messages = self._convert_messages_for_ollama(messages)

        # Prepare request payload
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "messages": converted_messages,
            "stream": should_stream,
            **self._get_api_kwargs(),
        }

        # Add tool-related parameters if configured
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_ollama(resolved_tools)

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/api/chat",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            return (self._normalize_chunk(chunk) for chunk in self._parse_stream(response))

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

        Note: Not all Ollama models support tool calling. This method provides
        tool parameters for interface consistency, but behavior depends on the
        model being used.

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
                Note: Not all Ollama models support this parameter.
            validate_tool_calls: If True, validate tool call arguments against the
                tool's JSON schema. Raises ToolCallValidationError on validation
                failure. Requires jsonschema package.

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks
            if streaming. When the model calls tools, the response message will
            have tool_calls populated.
        """
        should_stream = stream if stream is not None else self.streaming

        if not messages:
            raise ValueError("Messages cannot be empty")

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        # Note: parallel_tool_calls is resolved but Ollama may not support it
        _ = self._resolve_parallel_tool_calls(parallel_tool_calls)

        # Convert messages to Ollama format (handles tool-related message conversions)
        converted_messages = self._convert_messages_for_ollama(messages)

        # Prepare request payload
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "messages": converted_messages,
            "stream": should_stream,
            **self._get_api_kwargs(),
        }

        # Add tool-related parameters if configured
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_ollama(resolved_tools)

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/api/chat",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            async def generate():
                async for chunk in self._parse_stream_async(response):
                    yield self._normalize_chunk(chunk)

            return generate()

        response_data = response.json()
        result = self._normalize_response(response_data)

        # Validate tool calls if requested
        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result


    def _normalize_response(self, response: Dict[str, Any]) -> ChatCompletion:
        """Normalize a chat completion response."""
        message = response.get("message", {})

        # Extract tool_calls if present (Ollama returns them in message.tool_calls)
        tool_calls = None
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = []
            for tc in message["tool_calls"]:
                # Ollama format: {"function": {"name": "...", "arguments": {...}}}
                func_info = tc.get("function", {})
                args = func_info.get("arguments", {})
                # Ollama may return arguments as dict or string
                if isinstance(args, dict):
                    arguments_str = json.dumps(args)
                else:
                    arguments_str = str(args)

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=func_info.get("name", ""),
                            arguments=arguments_str,
                        ),
                    )
                )

        # Determine finish_reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"

        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=message.get("role", "assistant"),
                        content=message.get("content") or "",
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            model=response.get("model", self.get_model_name()),
            provider=self.provider,
            created=int(time.time()),
            usage=Usage(
                completion_tokens=response.get("eval_count", 0),
                prompt_tokens=response.get("prompt_eval_count", 0),
                total_tokens=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
            ),
        )

    def _normalize_chunk(self, chunk: Dict[str, Any]) -> ChatCompletionChunk:
        """Normalize a streaming chat completion chunk."""
        message = chunk.get("message", {})

        # Extract tool_calls if present in streaming chunks
        tool_calls_data = None
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls_data = []
            for idx, tc in enumerate(message["tool_calls"]):
                func_info = tc.get("function", {})
                args = func_info.get("arguments", {})
                if isinstance(args, dict):
                    arguments_str = json.dumps(args)
                else:
                    arguments_str = str(args)

                tool_calls_data.append({
                    "index": idx,
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": func_info.get("name", ""),
                        "arguments": arguments_str,
                    },
                })

        # Determine finish_reason
        finish_reason = None
        if chunk.get("done", False):
            finish_reason = "tool_calls" if tool_calls_data else "stop"

        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role=message.get("role", "assistant"),
                        content=message.get("content") or "",
                        tool_calls=tool_calls_data,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            model=chunk.get("model", self.get_model_name()),
            created=int(time.time()),
        )

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "gemma2"  # Default model available on the server

    def _get_models(self) -> List[Model]:
        """List all available models for this provider."""
        response = self.client.get(
            f"{self.base_url}/api/tags",
            headers=self._get_headers()
        )
        self._handle_error(response)
        
        models_data = response.json()
        return [
            Model(
                id=model["name"],
                owned_by="Ollama",
                context_window=32768,  # Default context window for most Ollama models
            )
            for model in models_data.get("models", [])
        ]

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "ollama"

    def to_langchain(self) -> "ChatOllama":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_ollama is not installed.
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_ollama. "
                "Install with: uv add langchain_ollama or pip install langchain_ollama"
            ) from e

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")

        langchain_kwargs = {
            "model": model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "num_ctx": self._config.get("num_ctx", 128000),
            "base_url": self.base_url,
        }

        # Handle keep_alive - only set if explicitly provided
        keep_alive = self._config.get("keep_alive")
        if keep_alive is not None:
            langchain_kwargs["keep_alive"] = keep_alive

        # Handle JSON format if structured output is requested
        if self.structured and isinstance(self.structured, dict):
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                langchain_kwargs["format"] = "json"

        # Pass SSL verification settings to LangChain via client_kwargs
        # ChatOllama uses httpx internally and passes these kwargs to the client
        ssl_verify = self._get_ssl_verify()
        if ssl_verify is not True:  # Only set if SSL is disabled or custom CA bundle
            client_kwargs = {"verify": ssl_verify}
            langchain_kwargs["client_kwargs"] = client_kwargs

        return ChatOllama(**self._clean_config(langchain_kwargs))
