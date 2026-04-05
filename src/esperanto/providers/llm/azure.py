"""Azure OpenAI language model provider."""

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
from pydantic import SecretStr

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
    from langchain_openai import AzureChatOpenAI


class AzureLanguageModel(LanguageModel):
    """Azure OpenAI language model implementation using direct HTTP."""

    def __post_init__(self):
        """Initialize with Azure-specific configuration."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Resolve configuration with priority: config dict → modality env var → generic env var
        self.api_key = (
            self.api_key or
            self._config.get("api_key") or
            os.getenv("AZURE_OPENAI_API_KEY_LLM") or
            os.getenv("AZURE_OPENAI_API_KEY")
        )

        self.azure_endpoint = (
            self.base_url or
            self._config.get("azure_endpoint") or
            os.getenv("AZURE_OPENAI_ENDPOINT_LLM") or
            os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # deployment_name is model_name for Azure
        self.deployment_name = self.model_name or self._get_default_model()

        # Validate required parameters
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY_LLM "
                "or AZURE_OPENAI_API_KEY environment variable, or provide in config."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT_LLM "
                "or AZURE_OPENAI_ENDPOINT environment variable, or provide in config."
            )
        if not self.deployment_name:
            raise ValueError("Azure OpenAI deployment name (model_name) not found")

        # Initialize httpx clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Azure API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_url(self, path: str) -> str:
        """Build Azure OpenAI URL."""
        # Remove trailing slash from endpoint
        endpoint = self.azure_endpoint.rstrip('/')
        # Azure URL pattern: {endpoint}/openai/v1/{path}
        return f"{endpoint}/openai/v1/{path}"

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Azure OpenAI API error: {error_message}")

    def _get_models(self) -> List[Model]:
        """List available models for this provider.

        Note: Azure doesn't have a models API endpoint - it uses deployments.
        Returns an empty list since model discovery isn't available.
        """
        return []

    def _convert_tools_to_responses(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Esperanto tools to OpenAI Responses API format.

        Args:
            tools: List of Esperanto Tool objects.

        Returns:
            List of tools in Responses API format (flattened), or None if no tools provided.
        """
        if not tools:
            return None
        result = []
        for tool in tools:
            tool_dict: Dict[str, Any] = {
                "type": tool.type,
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            }
            # Add strict mode if specified
            if tool.function.strict is not None:
                tool_dict["strict"] = tool.function.strict
            result.append(tool_dict)
        return result

    def _status_to_finish_reason(self, status: str) -> str:
        """Convert Responses API status to chat completions finish_reason."""
        mapping = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "error",
        }
        return mapping.get(status, status)

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Responses API output to ChatCompletion format."""
        # Extract content and tool calls from output items
        content = ""
        tool_calls = []

        for item in response_data.get("output", []):
            item_type = item.get("type")
            if item_type == "message":
                # Concatenate text from all content parts
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content += part.get("text", "")
            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id", ""),
                        type="function",
                        function=FunctionCall(
                            name=item.get("name", ""),
                            arguments=item.get("arguments", ""),
                        ),
                    )
                )

        status = response_data.get("status", "completed")
        finish_reason = "tool_calls" if tool_calls else self._status_to_finish_reason(status)

        choices = [
            Choice(
                index=0,
                message=Message(
                    content=content,
                    role="assistant",
                    tool_calls=tool_calls if tool_calls else None,
                ),
                finish_reason=finish_reason,
            )
        ]

        usage_data = response_data.get("usage") or {}

        return ChatCompletion(
            id=response_data["id"],
            choices=choices,
            created=response_data.get("created_at"),
            model=response_data.get("model", ""),
            provider=self.provider,
            usage=Usage(
                completion_tokens=usage_data.get("output_tokens", 0),
                prompt_tokens=usage_data.get("input_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )

    def _parse_sse_events(self, response: httpx.Response) -> Generator[ChatCompletionChunk, None, None]:
        """Parse Responses API SSE stream and yield ChatCompletionChunks."""
        response_id = ""
        response_model = ""
        created_at = 0
        # Track function call metadata from output_item.added events
        # Maps output_index -> {"name": str, "call_id": str}
        function_calls: Dict[int, Dict[str, str]] = {}

        for chunk in response.iter_text():
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Responses API sends: event: <type>\ndata: <json>
                # We only care about the data lines
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    event_data = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event_data.get("type", "")

                # Capture response metadata from lifecycle events
                if event_type in ("response.created", "response.in_progress"):
                    resp = event_data.get("response", {})
                    response_id = resp.get("id", response_id)
                    response_model = resp.get("model", response_model)
                    created_at = resp.get("created_at", created_at)
                    continue

                # Track function call identity from output_item.added
                if event_type == "response.output_item.added":
                    item = event_data.get("item", {})
                    if item.get("type") == "function_call":
                        idx = event_data.get("output_index", 0)
                        function_calls[idx] = {
                            "name": item.get("name", ""),
                            "call_id": item.get("call_id", ""),
                        }
                    continue

                if event_type == "response.output_text.delta":
                    yield ChatCompletionChunk(
                        id=response_id,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    content=event_data.get("delta", ""),
                                    role="assistant",
                                ),
                                finish_reason=None,
                            )
                        ],
                        created=created_at,
                        model=response_model,
                    )
                elif event_type == "response.function_call_arguments.delta":
                    output_index = event_data.get("output_index", 0)
                    fc_meta = function_calls.get(output_index, {})
                    tool_call_dict: Dict[str, Any] = {
                        "index": output_index,
                        "id": fc_meta.get("call_id", event_data.get("item_id", "")),
                        "type": "function",
                        "function": {
                            "name": fc_meta.get("name", ""),
                            "arguments": event_data.get("delta", ""),
                        },
                    }
                    yield ChatCompletionChunk(
                        id=response_id,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    content="",
                                    role="assistant",
                                    tool_calls=[tool_call_dict],
                                ),
                                finish_reason=None,
                            )
                        ],
                        created=created_at,
                        model=response_model,
                    )
                elif event_type == "response.completed":
                    resp = event_data.get("response", {})
                    response_id = resp.get("id", response_id)
                    yield ChatCompletionChunk(
                        id=response_id,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(content="", role="assistant"),
                                finish_reason="stop",
                            )
                        ],
                        created=created_at,
                        model=response_model,
                    )
                    return
                # Ignore other lifecycle events

    async def _parse_sse_events_async(self, response: httpx.Response) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Parse Responses API SSE stream asynchronously and yield ChatCompletionChunks."""
        response_id = ""
        response_model = ""
        created_at = 0
        function_calls: Dict[int, Dict[str, str]] = {}

        async for chunk in response.aiter_text():
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    event_data = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event_data.get("type", "")

                if event_type in ("response.created", "response.in_progress"):
                    resp = event_data.get("response", {})
                    response_id = resp.get("id", response_id)
                    response_model = resp.get("model", response_model)
                    created_at = resp.get("created_at", created_at)
                    continue

                if event_type == "response.output_item.added":
                    item = event_data.get("item", {})
                    if item.get("type") == "function_call":
                        idx = event_data.get("output_index", 0)
                        function_calls[idx] = {
                            "name": item.get("name", ""),
                            "call_id": item.get("call_id", ""),
                        }
                    continue

                if event_type == "response.output_text.delta":
                    yield ChatCompletionChunk(
                        id=response_id,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    content=event_data.get("delta", ""),
                                    role="assistant",
                                ),
                                finish_reason=None,
                            )
                        ],
                        created=created_at,
                        model=response_model,
                    )
                elif event_type == "response.function_call_arguments.delta":
                    output_index = event_data.get("output_index", 0)
                    fc_meta = function_calls.get(output_index, {})
                    tool_call_dict: Dict[str, Any] = {
                        "index": output_index,
                        "id": fc_meta.get("call_id", event_data.get("item_id", "")),
                        "type": "function",
                        "function": {
                            "name": fc_meta.get("name", ""),
                            "arguments": event_data.get("delta", ""),
                        },
                    }
                    yield ChatCompletionChunk(
                        id=response_id,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    content="",
                                    role="assistant",
                                    tool_calls=[tool_call_dict],
                                ),
                                finish_reason=None,
                            )
                        ],
                        created=created_at,
                        model=response_model,
                    )
                elif event_type == "response.completed":
                    resp = event_data.get("response", {})
                    response_id = resp.get("id", response_id)
                    yield ChatCompletionChunk(
                        id=response_id,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(content="", role="assistant"),
                                finish_reason="stop",
                            )
                        ],
                        created=created_at,
                        model=response_model,
                    )
                    return

    def _is_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model (o1, o3, o4, gpt-5 series)."""
        model_name = self.deployment_name.lower()
        return (model_name.startswith("o1") or
                model_name.startswith("o3") or
                model_name.startswith("o4") or
                model_name.startswith("gpt-5"))

    def _get_api_kwargs(
        self, override_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get kwargs for Azure API calls, using current instance attributes and overrides."""
        is_reasoning_model = self._is_reasoning_model()

        effective_kwargs = {
            "model": self.deployment_name,
            "store": False,
        }

        # Handle token parameters — Responses API uses max_output_tokens
        if is_reasoning_model:
            # Skip if it's the default value (850) for reasoning models
            if self.max_tokens != 850:
                effective_kwargs["max_output_tokens"] = self.max_tokens
        else:
            effective_kwargs["max_output_tokens"] = self.max_tokens

        # Handle temperature and top_p - reasoning models don't support these
        if not is_reasoning_model:
            effective_kwargs["temperature"] = self.temperature
            effective_kwargs["top_p"] = self.top_p

        effective_kwargs["stream"] = self.streaming

        if self.structured is not None:
            is_json_mode = False
            if isinstance(self.structured, dict):
                struct_type = self.structured.get("type")
                if struct_type == "json_object" or struct_type == "json":
                    is_json_mode = True
                else:
                    raise TypeError(
                        f"Invalid 'type' in structured_output dictionary: {struct_type}. Expected 'json' or 'json_object'."
                    )
            elif isinstance(self.structured, str):
                if self.structured == "json":
                    is_json_mode = True
                else:
                    raise TypeError(
                        f"Invalid string for structured_output: '{self.structured}'. Expected 'json'."
                    )
            else:
                raise TypeError(
                    f"Invalid type for structured_output: {type(self.structured)}. Expected dict or str 'json'."
                )

            if is_json_mode:
                effective_kwargs["text"] = {"format": {"type": "json_object"}}

        if override_kwargs:
            effective_kwargs.update(override_kwargs)

        return {k: v for k, v in effective_kwargs.items() if v is not None}

    def _chat_complete_streaming(
        self, messages: List[Dict[str, Any]], api_kwargs: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Handle streaming chat completion."""
        url = self._build_url("responses")
        with self.client.stream(
            "POST",
            url,
            headers=self._get_headers(),
            json={"input": messages, "stream": True, **api_kwargs},
        ) as response:
            self._handle_error(response)
            for chunk in self._parse_sse_events(response):
                yield chunk

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

        call_override_kwargs: Dict[str, Any] = {}
        if stream is not None:
            call_override_kwargs["stream"] = stream

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        resolved_parallel = self._resolve_parallel_tool_calls(parallel_tool_calls)

        api_kwargs = self._get_api_kwargs(override_kwargs=call_override_kwargs)
        effective_stream_setting = api_kwargs.pop("stream", False)

        # Add tool-related parameters if configured
        if resolved_tools:
            api_kwargs["tools"] = self._convert_tools_to_responses(resolved_tools)
        if resolved_tool_choice is not None:
            api_kwargs["tool_choice"] = resolved_tool_choice
        if resolved_parallel is not None:
            api_kwargs["parallel_tool_calls"] = resolved_parallel

        if effective_stream_setting:
            # Return streaming generator
            return self._chat_complete_streaming(messages, api_kwargs)
        else:
            # Non-streaming request
            url = self._build_url("responses")
            response = self.client.post(
                url,
                headers=self._get_headers(),
                json={"input": messages, "stream": False, **api_kwargs},
            )
            self._handle_error(response)
            result = self._normalize_response(response.json())

            # Validate tool calls if requested
            if validate_tool_calls and resolved_tools:
                for choice in result.choices:
                    if choice.message.tool_calls:
                        _validate_tool_calls(choice.message.tool_calls, resolved_tools)

            return result

    async def _achat_complete_streaming(
        self, messages: List[Dict[str, Any]], api_kwargs: Dict[str, Any]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Handle async streaming chat completion."""
        url = self._build_url("responses")
        async with self.async_client.stream(
            "POST",
            url,
            headers=self._get_headers(),
            json={"input": messages, "stream": True, **api_kwargs},
        ) as response:
            self._handle_error(response)
            async for chunk in self._parse_sse_events_async(response):
                yield chunk

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

        call_override_kwargs: Dict[str, Any] = {}
        if stream is not None:
            call_override_kwargs["stream"] = stream

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        resolved_parallel = self._resolve_parallel_tool_calls(parallel_tool_calls)

        api_kwargs = self._get_api_kwargs(override_kwargs=call_override_kwargs)
        effective_stream_setting = api_kwargs.pop("stream", False)

        # Add tool-related parameters if configured
        if resolved_tools:
            api_kwargs["tools"] = self._convert_tools_to_responses(resolved_tools)
        if resolved_tool_choice is not None:
            api_kwargs["tool_choice"] = resolved_tool_choice
        if resolved_parallel is not None:
            api_kwargs["parallel_tool_calls"] = resolved_parallel

        if effective_stream_setting:
            # Return async streaming generator
            return self._achat_complete_streaming(messages, api_kwargs)
        else:
            # Non-streaming async request
            url = self._build_url("responses")
            response = await self.async_client.post(
                url,
                headers=self._get_headers(),
                json={"input": messages, "stream": False, **api_kwargs},
            )
            self._handle_error(response)
            result = self._normalize_response(response.json())

            # Validate tool calls if requested
            if validate_tool_calls and resolved_tools:
                for choice in result.choices:
                    if choice.message.tool_calls:
                        _validate_tool_calls(choice.message.tool_calls, resolved_tools)

            return result

    def to_langchain(
        self, **kwargs: Any
    ) -> "AzureChatOpenAI":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_openai is not installed.
        """
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise ImportError(
                "LangChain or langchain-openai not installed. "
                "Please install with `pip install langchain_openai`"
            ) from e

        model_kwargs = {}
        if self.structured is not None:
            # Handle different structured formats
            if isinstance(self.structured, dict):
                struct_type = self.structured.get("type")
                if struct_type == "json_object" or struct_type == "json":
                    model_kwargs["response_format"] = {"type": "json_object"}
            elif self.structured == "json":
                model_kwargs["response_format"] = {"type": "json_object"}

        is_reasoning_model = self._is_reasoning_model()

        langchain_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "api_key": SecretStr(self.api_key) if self.api_key else None,
            "azure_deployment": self.deployment_name,
            "azure_endpoint": self.azure_endpoint,
            "api_version": "2025-04-01-preview",
            "use_responses_api": True,
            "model_kwargs": model_kwargs,
        }

        # Create new HTTP clients for LangChain with same SSL/timeout/proxy config
        # We create fresh clients instead of sharing ours because when this Esperanto
        # model is garbage collected, __del__ closes our clients - which would break
        # LangChain if it was sharing them. Fresh clients give LangChain ownership.
        try:
            sync_client, async_client = self._create_langchain_http_clients()
            langchain_kwargs["http_client"] = sync_client
            langchain_kwargs["http_async_client"] = async_client
        except (TypeError, AttributeError):
            # httpx types might be mocked in tests, skip passing clients
            pass

        if is_reasoning_model:
            # For reasoning models, put max_completion_tokens in model_kwargs
            if self.max_tokens != 850:
                model_kwargs["max_completion_tokens"] = self.max_tokens
            langchain_kwargs["temperature"] = 1
            langchain_kwargs["top_p"] = None
        else:
            langchain_kwargs["max_tokens"] = self.max_tokens

        langchain_kwargs.update(kwargs)

        final_lc_kwargs = {k: v for k, v in langchain_kwargs.items() if v is not None or k == "api_key"}

        # Remove model_kwargs if it's empty and not explicitly passed
        if not final_lc_kwargs.get("model_kwargs") and "model_kwargs" not in kwargs:
            final_lc_kwargs.pop("model_kwargs", None)

        return AzureChatOpenAI(**self._clean_config(final_lc_kwargs))

    @property
    def provider(self) -> str:
        return "azure"

    def _get_default_model(self) -> str:
        """Get the default model name (deployment name for Azure).

        For Azure, model_name (deployment name) is required for actual usage.
        Returns empty string if no model is configured - the validation in __post_init__
        will catch this and raise an error.
        """
        return ""
