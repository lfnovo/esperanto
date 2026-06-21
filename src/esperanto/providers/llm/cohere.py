"""Cohere language model implementation (native v2 API)."""

import json
import logging
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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_cohere import ChatCohere

# Cohere-specific request fields that pass through from config without an analog
# on other Esperanto providers (built-in RAG / citations / knowledge connectors).
_COHERE_PASSTHROUGH_FIELDS = ("documents", "citation_options", "connectors")


@dataclass
class CohereLanguageModel(LanguageModel):
    """Cohere language model implementation using the native v2 chat API."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("COHERE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Cohere API key not found. Set the COHERE_API_KEY environment variable."
            )

        # Set base URL (Cohere uses unversioned host; endpoints carry the version)
        self.base_url = (self.base_url or "https://api.cohere.com").rstrip("/")

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Cohere API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("message") or error_data.get(
                    "error", f"HTTP {response.status_code}"
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Cohere API error: {error_message}")

    def _convert_tools_to_cohere(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Esperanto tools to Cohere v2 format.

        Cohere v2 uses an OpenAI-style tool shape:
        ``{"type": "function", "function": {name, description, parameters}}``.
        """
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                },
            }
            for tool in tools
        ]

    def _convert_tool_choice_to_cohere(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[str]:
        """Convert tool_choice to Cohere format.

        Cohere v2 only supports ``REQUIRED`` and ``NONE`` (uppercase). "auto" maps
        to omitting the field (Cohere's default behavior). Specific-tool forcing is
        not supported, so we fall back to ``REQUIRED``.
        """
        if tool_choice is None or tool_choice == "auto":
            return None
        if tool_choice == "required":
            return "REQUIRED"
        if tool_choice == "none":
            return "NONE"
        # Specific tool dict — Cohere can't force a named tool; require any tool.
        if isinstance(tool_choice, dict):
            return "REQUIRED"
        return str(tool_choice).upper()

    def _format_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert Esperanto/OpenAI messages to Cohere v2 messages.

        Cohere v2 uses an OpenAI-compatible message shape, so most messages pass
        through. Assistant tool calls and tool results need light normalization to
        ensure arguments are JSON strings and roles map correctly.
        """
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "tool":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id"),
                    "content": msg.get("content", ""),
                })

            elif role == "assistant" and msg.get("tool_calls"):
                tool_calls: List[Dict[str, Any]] = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id", "")
                        func_info = tc.get("function", {})
                        func_name = func_info.get("name", "")
                        func_args = func_info.get("arguments", "{}")
                    else:
                        tc_id = tc.id
                        func_name = tc.function.name
                        func_args = tc.function.arguments

                    # Cohere expects arguments as a JSON string (same as Esperanto).
                    if not isinstance(func_args, str):
                        func_args = json.dumps(func_args)

                    tool_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": func_name, "arguments": func_args},
                    })

                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                # Cohere uses tool_plan to carry the assistant's pre-tool reasoning.
                if msg.get("content"):
                    assistant_msg["tool_plan"] = msg["content"]
                formatted.append(assistant_msg)

            else:
                formatted.append({
                    "role": "assistant" if role == "assistant" else role or "user",
                    "content": msg.get("content", ""),
                })

        return formatted

    def _create_request_payload(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create request payload for the Cohere v2 chat API."""
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )
        effective_top_p = top_p if top_p is not None else self.top_p

        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "messages": self._format_messages(messages),
        }

        if effective_max_tokens is not None:
            payload["max_tokens"] = int(effective_max_tokens)
        if effective_temperature is not None:
            payload["temperature"] = float(effective_temperature)
        # Cohere uses "p" for nucleus sampling (not "top_p").
        if effective_top_p is not None:
            payload["p"] = float(effective_top_p)

        if stream:
            payload["stream"] = True

        if tools:
            payload["tools"] = self._convert_tools_to_cohere(tools)
            cohere_tool_choice = self._convert_tool_choice_to_cohere(tool_choice)
            if cohere_tool_choice:
                payload["tool_choice"] = cohere_tool_choice

        # JSON mode via structured output config.
        if self.structured and self.structured.get("type") in ("json", "json_object"):
            payload["response_format"] = {"type": "json_object"}

        # Pass through Cohere-specific fields (documents/citations/connectors) from
        # config without altering the universal chat_complete signature.
        for field in _COHERE_PASSTHROUGH_FIELDS:
            if field in self._config and self._config[field] is not None:
                payload[field] = self._config[field]

        return payload

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize a Cohere v2 chat response into a ChatCompletion."""
        created = int(time.time())
        message_data = response_data.get("message", {})
        content_blocks = message_data.get("content", []) or []

        text_parts: List[str] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        tool_calls: List[ToolCall] = []
        for tc in message_data.get("tool_calls", []) or []:
            func = tc.get("function", {})
            arguments = func.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", str(uuid.uuid4())),
                    type="function",
                    function=FunctionCall(
                        name=func.get("name", ""),
                        arguments=arguments,
                    ),
                )
            )

        content_text = "".join(text_parts) if text_parts else None
        finish_reason = self._map_finish_reason(response_data.get("finish_reason"))

        usage_data = response_data.get("usage", {}).get("tokens", {}) or {}
        prompt_tokens = int(usage_data.get("input_tokens", 0) or 0)
        completion_tokens = int(usage_data.get("output_tokens", 0) or 0)

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
            model=self.get_model_name(),
            provider=self.provider,
            usage=Usage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @staticmethod
    def _map_finish_reason(cohere_reason: Optional[str]) -> str:
        """Map a Cohere finish_reason to the Esperanto convention."""
        mapping = {
            "COMPLETE": "stop",
            "STOP_SEQUENCE": "stop",
            "MAX_TOKENS": "length",
            "TOOL_CALL": "tool_calls",
        }
        if not cohere_reason:
            return "stop"
        return mapping.get(cohere_reason, cohere_reason.lower())

    def _parse_sse_stream(
        self, response: httpx.Response
    ) -> Generator[Dict[str, Any], None, None]:
        """Parse the Cohere SSE stream into event dicts."""
        for chunk in response.iter_text():
            for line in chunk.split("\n"):
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    async def _parse_sse_stream_async(
        self, response: httpx.Response
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Parse the Cohere SSE stream asynchronously into event dicts."""
        async for chunk in response.aiter_text():
            for line in chunk.split("\n"):
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    def _normalize_stream_event(
        self, event_data: Dict[str, Any]
    ) -> Optional[ChatCompletionChunk]:
        """Normalize a Cohere stream event into a ChatCompletionChunk.

        Handles:
        - content-delta: text token (delta.message.content.text)
        - tool-call-start: start of a tool call (delta.message.tool_calls)
        - tool-call-delta: tool argument chunks (delta.message.tool_calls.function.arguments)
        - message-end: completion with finish_reason (delta.finish_reason)
        """
        event_type = event_data.get("type")
        delta = event_data.get("delta", {}) or {}
        message = delta.get("message", {}) or {}
        index = event_data.get("index", 0)

        if event_type == "content-delta":
            text_content = message.get("content", {}).get("text", "")
            return self._text_chunk(text_content)

        if event_type == "tool-call-start":
            tc = message.get("tool_calls", {}) or {}
            func = tc.get("function", {}) or {}
            return self._tool_chunk(
                tc_id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=func.get("arguments", ""),
                index=index,
            )

        if event_type == "tool-call-delta":
            func = message.get("tool_calls", {}).get("function", {}) or {}
            return self._tool_chunk(
                tc_id="",
                name="",
                arguments=func.get("arguments", ""),
                index=index,
            )

        if event_type == "message-end":
            finish_reason = self._map_finish_reason(delta.get("finish_reason"))
            return ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=None, role="assistant"),
                        finish_reason=finish_reason,
                    )
                ],
                created=int(time.time()),
                model=self.get_model_name(),
            )

        # Ignore message-start, content-start, content-end, tool-call-end, etc.
        return None

    def _text_chunk(self, text_content: str) -> ChatCompletionChunk:
        """Build a text-delta streaming chunk."""
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=text_content, role="assistant"),
                    finish_reason=None,
                )
            ],
            created=int(time.time()),
            model=self.get_model_name(),
        )

    def _tool_chunk(
        self, tc_id: str, name: str, arguments: str, index: int
    ) -> ChatCompletionChunk:
        """Build a tool-call streaming chunk."""
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id=tc_id,
                                type="function",
                                function=FunctionCall(
                                    name=name, arguments=arguments
                                ),
                                index=index,
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            created=int(time.time()),
            model=self.get_model_name(),
        )

    def get_model_name(self) -> str:
        """Get the model name to use."""
        return self.model_name or self._get_default_model()

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "command-a-03-2025"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "cohere"

    def _get_models(self) -> List[Model]:
        """List all available language models for this provider."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/models",
                headers=self._get_headers(),
                params={"endpoint": "chat"},
            )
            self._handle_error(response)
            models_data = response.json()
            return [
                Model(
                    id=model["name"],
                    owned_by="cohere",
                    context_window=model.get("context_length"),
                    type="language",
                )
                for model in models_data.get("models", [])
            ]
        except Exception:
            return [
                Model(
                    id="command-a-03-2025",
                    owned_by="cohere",
                    context_window=256000,
                    type="language",
                ),
            ]

    def chat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request to the Cohere v2 chat API.

        Cohere-specific RAG features (``documents``, ``citation_options``,
        ``connectors``) are supplied via ``config`` and ride along on the request.
        Citations returned by Cohere are not surfaced on the universal
        ``ChatCompletion`` response (out of scope for v1).
        """
        self._warn_if_validate_with_streaming(validate_tool_calls, stream)

        should_stream = stream if stream is not None else self.streaming

        effective_max_tokens = self._resolve_max_tokens(max_tokens)
        effective_temperature = self._resolve_temperature(temperature)
        effective_top_p = self._resolve_top_p(top_p)

        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)

        payload = self._create_request_payload(
            messages,
            should_stream,
            tools=resolved_tools,
            tool_choice=resolved_tool_choice,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
            top_p=effective_top_p,
        )

        response = self.client.post(
            f"{self.base_url}/v2/chat",
            headers=self._get_headers(),
            json=payload,
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
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request to the Cohere v2 chat API."""
        self._warn_if_validate_with_streaming(validate_tool_calls, stream)

        should_stream = stream if stream is not None else self.streaming

        effective_max_tokens = self._resolve_max_tokens(max_tokens)
        effective_temperature = self._resolve_temperature(temperature)
        effective_top_p = self._resolve_top_p(top_p)

        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)

        payload = self._create_request_payload(
            messages,
            should_stream,
            tools=resolved_tools,
            tool_choice=resolved_tool_choice,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
            top_p=effective_top_p,
        )

        response = await self.async_client.post(
            f"{self.base_url}/v2/chat",
            headers=self._get_headers(),
            json=payload,
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

        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result

    def to_langchain(self) -> "ChatCohere":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_cohere is not installed.
        """
        try:
            from langchain_cohere import ChatCohere
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_cohere. "
                "Install with: uv add langchain_cohere or pip install langchain_cohere"
            ) from e

        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "cohere_api_key": self.api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        return ChatCohere(**kwargs)  # type: ignore[arg-type]
