"""Anthropic language model implementation."""

import json
import os
import time
import uuid
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
    from langchain_anthropic import ChatAnthropic


@dataclass
class AnthropicLanguageModel(LanguageModel):
    """Anthropic language model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable."
            )

        # Set base URL
        self.base_url = self.base_url or "https://api.anthropic.com/v1"

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Anthropic API requests."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Anthropic API error: {error_message}")

    def _convert_tools_to_anthropic(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Esperanto tools to Anthropic format.

        Args:
            tools: List of Esperanto Tool objects.

        Returns:
            List of tools in Anthropic API format, or None if no tools provided.
        """
        if not tools:
            return None
        return [
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "input_schema": tool.function.parameters,
            }
            for tool in tools
        ]

    def _convert_tool_choice_to_anthropic(
        self,
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        parallel_tool_calls: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convert tool_choice to Anthropic format.

        Args:
            tool_choice: The tool choice setting (Esperanto/OpenAI format).
            parallel_tool_calls: Whether to allow parallel tool calls.

        Returns:
            Tool choice in Anthropic format, or None if not configured.
        """
        if tool_choice is None and parallel_tool_calls is None:
            return None

        result: Dict[str, Any] = {}

        if tool_choice == "auto":
            result["type"] = "auto"
        elif tool_choice == "required":
            result["type"] = "any"
        elif tool_choice == "none":
            # Anthropic doesn't have a direct "none" equivalent
            # Return None to not send tool_choice (tools will still be sent)
            return None
        elif isinstance(tool_choice, dict):
            # Handle specific tool: {"type": "function", "function": {"name": "..."}}
            func_info = tool_choice.get("function", {})
            tool_name = func_info.get("name")
            if tool_name:
                result["type"] = "tool"
                result["name"] = tool_name
        elif tool_choice is not None:
            # Pass through if it's already in Anthropic format
            result["type"] = str(tool_choice)

        # Handle parallel tool calls - Anthropic uses disable_parallel_tool_use
        if parallel_tool_calls is False:
            result["disable_parallel_tool_use"] = True

        return result if result else None

    def _get_models(self) -> List[Model]:
        """List all available models for this provider."""
        try:
            response = self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            )
            self._handle_error(response)
            
            models_data = response.json()
            return [
                Model(
                    id=model["id"],
                    owned_by="Anthropic",
                    context_window=model.get("max_tokens", 200000),
                )
                for model in models_data.get("data", [])
            ]
        except Exception:
            # Fallback to known models if API call fails
            return [
                Model(
                    id="claude-3-7-sonnet-20250219",
                    owned_by="Anthropic",
                    context_window=200000,
                ),
                Model(
                    id="claude-3-opus-20240229",
                    owned_by="Anthropic",
                    context_window=200000,
                ),
                Model(
                    id="claude-3-sonnet-20240229",
                    owned_by="Anthropic",
                    context_window=200000,
                ),
                Model(
                    id="claude-3-haiku-20240307",
                    owned_by="Anthropic",
                    context_window=200000,
                ),
            ]

    def _prepare_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Handle Anthropic-specific message preparation.

        Converts messages to Anthropic format, including handling:
        - System messages (extracted separately)
        - Tool result messages (converted to tool_result content blocks)
        - Assistant messages with tool_calls (converted to tool_use content blocks)

        Args:
            messages: List of messages in Esperanto/OpenAI format.

        Returns:
            Tuple of (system_message, formatted_messages) for Anthropic API.
        """
        system_message = None
        formatted_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                system_message = msg.get("content")

            elif role == "tool":
                # Convert tool result to Anthropic's tool_result format
                formatted_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": msg.get("content", ""),
                    }]
                })

            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant tool_calls to Anthropic's tool_use format
                content: List[Dict[str, Any]] = []

                # Add text content if present
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})

                # Add tool_use blocks for each tool call
                for tc in msg["tool_calls"]:
                    # Handle both dict and ToolCall objects
                    if isinstance(tc, dict):
                        tc_id = tc.get("id", "")
                        func_info = tc.get("function", {})
                        func_name = func_info.get("name", "")
                        func_args = func_info.get("arguments", "{}")
                    else:
                        # ToolCall object
                        tc_id = tc.id
                        func_name = tc.function.name
                        func_args = tc.function.arguments

                    # Parse arguments JSON to dict (Anthropic expects dict, not JSON string)
                    try:
                        args_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
                    except json.JSONDecodeError:
                        args_dict = {}

                    content.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": func_name,
                        "input": args_dict,
                    })

                formatted_messages.append({"role": "assistant", "content": content})

            else:
                # Regular user or assistant message
                formatted_messages.append({
                    "role": "assistant" if role == "assistant" else "user",
                    "content": msg.get("content", ""),
                })

        return system_message, formatted_messages

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Anthropic response to our format.

        Handles both text content and tool_use blocks from Anthropic responses.
        """
        created = int(time.time())
        content_blocks = response_data.get("content", [])

        # Extract text content and tool calls
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for block in content_blocks:
            block_type = block.get("type")

            if block_type == "text":
                text_parts.append(block.get("text", ""))

            elif block_type == "tool_use":
                # Convert Anthropic tool_use to Esperanto ToolCall
                # Note: Anthropic uses "input" as dict, we store as JSON string
                tool_input = block.get("input", {})
                arguments_str = json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)

                tool_calls.append(
                    ToolCall(
                        id=block.get("id", str(uuid.uuid4())),
                        type="function",
                        function=FunctionCall(
                            name=block.get("name", ""),
                            arguments=arguments_str,
                        ),
                    )
                )

        # Combine text parts
        content_text = "".join(text_parts) if text_parts else None

        # Map Anthropic stop_reason to standard finish_reason
        stop_reason = response_data.get("stop_reason", "stop")
        # Anthropic uses "tool_use" when model wants to call tools
        if stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif stop_reason == "end_turn":
            finish_reason = "stop"
        else:
            finish_reason = stop_reason

        return ChatCompletion(
            id=response_data.get("id", str(uuid.uuid4())),
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        content=content_text,
                        role="assistant",
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            created=created,
            model=response_data.get("model", self.get_model_name()),
            provider=self.provider,
            usage=Usage(
                completion_tokens=response_data.get("usage", {}).get("output_tokens", 0),
                prompt_tokens=response_data.get("usage", {}).get("input_tokens", 0),
                total_tokens=response_data.get("usage", {}).get("input_tokens", 0) + response_data.get("usage", {}).get("output_tokens", 0),
            ),
        )

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse Server-Sent Events stream from Anthropic."""
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
        """Parse Server-Sent Events stream from Anthropic asynchronously."""
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

    def _normalize_stream_event(self, event_data: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """Normalize Anthropic stream event to our format.

        Handles:
        - content_block_delta with text: Regular text streaming
        - content_block_start with tool_use: Start of a tool call
        - content_block_delta with input_json_delta: Tool call argument chunks
        - message_delta: Message completion with stop_reason
        """
        event_type = event_data.get("type")

        # Handle content block start (for tool_use blocks)
        if event_type == "content_block_start":
            content_block = event_data.get("content_block", {})
            block_type = content_block.get("type")

            if block_type == "tool_use":
                # Start of a tool call - emit the tool call info
                block_index = event_data.get("index", 0)
                tool_call_dict = {
                    "index": block_index,
                    "id": content_block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": content_block.get("name", ""),
                        "arguments": "",  # Arguments come in subsequent deltas
                    },
                }
                return ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                content=None,
                                role="assistant",
                                tool_calls=[tool_call_dict],
                            ),
                            finish_reason=None,
                        )
                    ],
                    created=int(time.time()),
                    model=self.get_model_name(),
                )

        # Handle content delta events
        elif event_type == "content_block_delta":
            delta = event_data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta" or "text" in delta:
                # Text content delta
                text_content = delta.get("text", "")
                return ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                content=text_content,
                                role="assistant",
                            ),
                            finish_reason=None,
                        )
                    ],
                    created=int(time.time()),
                    model=self.get_model_name(),
                )

            elif delta_type == "input_json_delta":
                # Tool call arguments delta
                # Note: In streaming, we emit partial tool call updates. The index
                # corresponds to the content block index from Anthropic.
                block_index = event_data.get("index", 0)
                partial_json = delta.get("partial_json", "")
                # We need to include all required fields for ToolCall validation.
                # Use empty id/name for delta updates - these are only valid as
                # incremental updates to be accumulated by the client.
                tool_call_dict = {
                    "index": block_index,
                    "id": "",  # Empty for delta updates
                    "type": "function",
                    "function": {
                        "name": "",  # Empty for delta updates
                        "arguments": partial_json,
                    },
                }
                return ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                content=None,
                                role="assistant",
                                tool_calls=[tool_call_dict],
                            ),
                            finish_reason=None,
                        )
                    ],
                    created=int(time.time()),
                    model=self.get_model_name(),
                )

        # Handle message completion event
        elif event_type == "message_delta":
            delta = event_data.get("delta", {})
            stop_reason = delta.get("stop_reason", "stop")
            # Map Anthropic stop_reason to standard finish_reason
            if stop_reason == "tool_use":
                finish_reason = "tool_calls"
            elif stop_reason == "end_turn":
                finish_reason = "stop"
            else:
                finish_reason = stop_reason

            return ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            content=None,
                            role="assistant",
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                created=int(time.time()),
                model=self.get_model_name(),
            )

        # Ignore other event types (message_start, content_block_stop, message_stop, ping)
        return None

    # Removed the faulty _prepare_api_kwargs method

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = self.get_completion_kwargs()

        # Remove provider-specific kwargs that Anthropic doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        kwargs.pop("organization", None)

        # Handle streaming
        if exclude_stream:
            kwargs.pop("streaming", None)
        elif "streaming" in kwargs:
            kwargs["stream"] = kwargs.pop("streaming")

        # Handle temperature - Anthropic expects 0-1 range
        if "temperature" in kwargs:
            temp = kwargs["temperature"]
            if temp is not None:
                kwargs["temperature"] = max(0.0, min(1.0, float(temp)))

        # Handle max_tokens - required by Anthropic
        if "max_tokens" in kwargs:
            max_tokens = kwargs["max_tokens"]
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)

        return kwargs

    def get_model_name(self) -> str:
        """Get the model name to use."""
        return self.model_name or self._get_default_model()

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "claude-3-7-sonnet-20250219"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "anthropic"

    def _create_request_payload(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create request payload for Anthropic API.

        Args:
            messages: List of messages in the conversation.
            stream: Whether to stream the response.
            tools: List of tools the model can call.
            tool_choice: Controls tool usage.
            parallel_tool_calls: Whether to allow parallel tool calls.

        Returns:
            Request payload dict for Anthropic API.
        """
        system_message, formatted_messages = self._prepare_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "messages": formatted_messages,
            "max_tokens": self.max_tokens or 1024,
        }

        if system_message:
            payload["system"] = system_message

        # Anthropic does not allow both temperature and top_p to be set
        # Prioritize temperature if both are provided
        if self.temperature is not None:
            payload["temperature"] = max(0.0, min(1.0, float(self.temperature)))
        elif self.top_p is not None:
            payload["top_p"] = float(self.top_p)

        if stream:
            payload["stream"] = True

        # Add tools if provided
        if tools:
            payload["tools"] = self._convert_tools_to_anthropic(tools)

        # Add tool_choice if provided
        anthropic_tool_choice = self._convert_tool_choice_to_anthropic(
            tool_choice, parallel_tool_calls
        )
        if anthropic_tool_choice:
            payload["tool_choice"] = anthropic_tool_choice

        return payload

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

        payload = self._create_request_payload(
            messages,
            should_stream,
            tools=resolved_tools,
            tool_choice=resolved_tool_choice,
            parallel_tool_calls=resolved_parallel,
        )

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/messages",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            def generate():
                for event_data in self._parse_sse_stream(response):
                    chunk = self._normalize_stream_event(event_data)
                    if chunk:
                        yield chunk
            return generate()

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

        payload = self._create_request_payload(
            messages,
            should_stream,
            tools=resolved_tools,
            tool_choice=resolved_tool_choice,
            parallel_tool_calls=resolved_parallel,
        )

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/messages",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            async def generate():
                async for event_data in self._parse_sse_stream_async(response):
                    chunk = self._normalize_stream_event(event_data)
                    if chunk:
                        yield chunk
            return generate()

        response_data = response.json()
        result = self._normalize_response(response_data)

        # Validate tool calls if requested
        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result

    def to_langchain(self) -> "ChatAnthropic":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_anthropic is not installed.
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_anthropic. "
                "Install with: uv add langchain_anthropic or pip install langchain_anthropic"
            ) from e


        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")

        # Anthropic does not allow both temperature and top_p to be set
        # Prioritize temperature if both are provided
        kwargs = {
            "model": model_name,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        elif self.top_p is not None:
            kwargs["top_p"] = self.top_p

        return ChatAnthropic(**kwargs)
