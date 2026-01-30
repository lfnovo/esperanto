"""
Pydantic AI integration for Esperanto.

This module provides an adapter that allows any Esperanto LanguageModel
to be used with Pydantic AI agents.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

try:
    import pydantic_ai as _pydantic_ai_check  # noqa: F401

    PYDANTIC_AI_INSTALLED = True
    _PYDANTIC_AI_IMPORT_ERROR: ImportError | None = None
except ImportError as e:
    PYDANTIC_AI_INSTALLED = False
    _PYDANTIC_AI_IMPORT_ERROR = e
    # Type stubs for when pydantic-ai is not installed
    Model = object  # type: ignore[misc, assignment]
    StreamedResponse = object  # type: ignore[misc, assignment]

if PYDANTIC_AI_INSTALLED:
    try:
        from pydantic_ai.messages import (
            ModelMessage,
            ModelRequest,
            ModelResponse,
            ModelResponseStreamEvent,
            RetryPromptPart,
            SystemPromptPart,
            TextPart,
            ThinkingPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )
        from pydantic_ai.models import Model, StreamedResponse
        from pydantic_ai.settings import ModelSettings
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.usage import RequestUsage
    except ImportError as e:
        # pydantic-ai is installed but missing expected classes
        # This likely means the version is too old
        _PYDANTIC_AI_IMPORT_ERROR = e
        PYDANTIC_AI_INSTALLED = False
        Model = object  # type: ignore[misc, assignment]
        StreamedResponse = object  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestParameters

    from esperanto.common_types import ChatCompletion, ChatCompletionChunk
    from esperanto.providers.llm.base import LanguageModel


def _check_pydantic_ai_installed() -> None:
    """Raise ImportError if pydantic-ai is not installed or incompatible."""
    if not PYDANTIC_AI_INSTALLED:
        base_msg = (
            "Pydantic AI integration requires pydantic-ai>=1.50.0. "
            "Install with: uv add 'pydantic-ai>=1.50.0' or pip install 'pydantic-ai>=1.50.0'"
        )
        if _PYDANTIC_AI_IMPORT_ERROR:
            raise ImportError(
                f"{base_msg}\n\nOriginal error: {_PYDANTIC_AI_IMPORT_ERROR}"
            ) from _PYDANTIC_AI_IMPORT_ERROR
        raise ImportError(base_msg)


def _now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


# Finish reason mapping from Esperanto to Pydantic AI
_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}


@dataclass
class EsperantoPydanticModel(Model):  # type: ignore[misc]
    """
    Pydantic AI Model adapter that wraps any Esperanto LanguageModel.

    This adapter allows Esperanto models to be used with Pydantic AI agents,
    providing a bridge between Esperanto's unified interface and Pydantic AI's
    agent framework.

    Example:
        ```python
        from esperanto import AIFactory
        from pydantic_ai import Agent

        # Create Esperanto model
        model = AIFactory.create_language("openai", "gpt-4o")

        # Use with Pydantic AI
        agent = Agent(model.to_pydantic_ai())
        result = await agent.run("Hello!")
        ```
    """

    _esperanto_model: "LanguageModel"
    _model_name: str = field(default="", init=False)
    _system: str = field(default="", init=False)

    def __init__(self, esperanto_model: "LanguageModel") -> None:
        """
        Initialize the adapter with an Esperanto LanguageModel.

        Args:
            esperanto_model: Any Esperanto LanguageModel instance (OpenAI, Anthropic, etc.)
        """
        _check_pydantic_ai_installed()
        self._esperanto_model = esperanto_model
        self._model_name = esperanto_model.get_model_name()
        self._system = esperanto_model.provider

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    @property
    def system(self) -> str:
        """Return the provider name for telemetry."""
        return self._system

    def _convert_messages(
        self, messages: list[ModelMessage]
    ) -> list[dict[str, Any]]:
        """
        Convert Pydantic AI messages to Esperanto format.

        Args:
            messages: List of Pydantic AI ModelMessage objects

        Returns:
            List of message dicts in Esperanto/OpenAI format
        """
        result: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                # Process request parts
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        result.append({
                            "role": "system",
                            "content": part.content,
                        })
                    elif isinstance(part, UserPromptPart):
                        # Handle string or multimodal content
                        if isinstance(part.content, str):
                            result.append({
                                "role": "user",
                                "content": part.content,
                            })
                        else:
                            # Multimodal content - convert to string for now
                            # TODO: Handle images, audio, etc.
                            content_parts = []
                            for content_item in part.content:
                                if hasattr(content_item, "text"):
                                    content_parts.append(content_item.text)
                                elif isinstance(content_item, str):
                                    content_parts.append(content_item)
                            result.append({
                                "role": "user",
                                "content": " ".join(content_parts) if content_parts else "",
                            })
                    elif isinstance(part, ToolReturnPart):
                        # Tool result message
                        content = part.content
                        if not isinstance(content, str):
                            content = json.dumps(content)
                        result.append({
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": content,
                        })
                    elif isinstance(part, RetryPromptPart):
                        # Retry prompt - convert to user message
                        content = part.content
                        if not isinstance(content, str):
                            content = str(content)
                        result.append({
                            "role": "user",
                            "content": f"Error: {content}. Please try again.",
                        })

            elif isinstance(msg, ModelResponse):
                # Reconstruct assistant message from response parts
                text_content = ""
                tool_calls = []

                for part in msg.parts:
                    if isinstance(part, TextPart):
                        text_content += part.content
                    elif isinstance(part, ThinkingPart):
                        # Include thinking in content with tags
                        text_content = f"<think>{part.content}</think>\n{text_content}"
                    elif isinstance(part, ToolCallPart):
                        # Convert tool call
                        args = part.args
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        elif args is None:
                            args = "{}"
                        tool_calls.append({
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": args,
                            },
                        })

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": text_content or None,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                result.append(assistant_msg)

        return result

    def _convert_tools(
        self, params: "ModelRequestParameters"
    ) -> list[Any] | None:
        """
        Convert Pydantic AI tool definitions to Esperanto Tool format.

        Args:
            params: ModelRequestParameters containing tool definitions

        Returns:
            List of Tool objects in Esperanto format, or None if no tools
        """
        from esperanto.common_types import Tool, ToolFunction

        tools: list[ToolDefinition] = []

        # Get function tools
        if hasattr(params, "function_tools") and params.function_tools:
            tools.extend(params.function_tools)

        # Get output tools (used for structured output)
        if hasattr(params, "output_tools") and params.output_tools:
            tools.extend(params.output_tools)

        if not tools:
            return None

        esperanto_tools: list[Tool] = []
        for tool_def in tools:
            esperanto_tools.append(
                Tool(
                    type="function",
                    function=ToolFunction(
                        name=tool_def.name,
                        description=tool_def.description or "",
                        parameters=tool_def.parameters_json_schema,
                    ),
                )
            )

        return esperanto_tools

    def _convert_usage(self, usage: Any) -> "RequestUsage":
        """
        Convert Esperanto Usage to Pydantic AI RequestUsage.

        Args:
            usage: Esperanto Usage object

        Returns:
            Pydantic AI RequestUsage
        """
        if usage is None:
            return RequestUsage()

        return RequestUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

    def _map_finish_reason(self, reason: str | None) -> str | None:
        """Map Esperanto finish reason to Pydantic AI format."""
        if reason is None:
            return None
        return _FINISH_REASON_MAP.get(reason, reason)

    def _convert_response(
        self, completion: "ChatCompletion"
    ) -> "ModelResponse":
        """
        Convert Esperanto ChatCompletion to Pydantic AI ModelResponse.

        Args:
            completion: Esperanto ChatCompletion response

        Returns:
            Pydantic AI ModelResponse
        """
        parts: list[Any] = []
        message = completion.choices[0].message

        # Handle thinking/reasoning traces
        if message.thinking:
            parts.append(ThinkingPart(content=message.thinking))

        # Handle text content
        cleaned_content = message.cleaned_content
        if cleaned_content:
            parts.append(TextPart(content=cleaned_content))

        # Handle tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments - Pydantic AI accepts str or dict
                args: str | dict[str, Any] = tc.function.arguments
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as string

                parts.append(
                    ToolCallPart(
                        tool_name=tc.function.name,
                        args=args,
                        tool_call_id=tc.id,
                    )
                )

        return ModelResponse(
            parts=parts,
            usage=self._convert_usage(completion.usage),
            model_name=completion.model,
            timestamp=_now_utc(),
            provider_name=self._system,
            finish_reason=self._map_finish_reason(
                completion.choices[0].finish_reason
            ),
        )

    def _apply_settings(
        self, settings: ModelSettings | None
    ) -> dict[str, Any]:
        """
        Extract settings overrides from ModelSettings.

        Args:
            settings: Pydantic AI ModelSettings or None (TypedDict)

        Returns:
            Dict of kwargs to pass to achat_complete
        """
        if settings is None:
            return {}

        kwargs: dict[str, Any] = {}

        # ModelSettings is a TypedDict, so use dict-style access
        if settings.get("max_tokens") is not None:
            kwargs["max_tokens"] = settings["max_tokens"]
        if settings.get("temperature") is not None:
            kwargs["temperature"] = settings["temperature"]
        if settings.get("top_p") is not None:
            kwargs["top_p"] = settings["top_p"]

        return kwargs

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: "ModelRequestParameters",
    ) -> "ModelResponse":
        """
        Make a non-streaming request to the model.

        This is the main method called by Pydantic AI agents.

        Args:
            messages: Conversation history in Pydantic AI format
            model_settings: Optional settings overrides (temperature, max_tokens, etc.)
            model_request_parameters: Tool definitions and output configuration

        Returns:
            ModelResponse with the model's reply
        """
        # Convert messages to Esperanto format
        esperanto_messages = self._convert_messages(messages)

        # Convert tools
        tools = self._convert_tools(model_request_parameters)

        # Get settings overrides
        settings_kwargs = self._apply_settings(model_settings)

        # Make the request via Esperanto
        completion = await self._esperanto_model.achat_complete(
            messages=esperanto_messages,
            stream=False,
            tools=tools,
            **settings_kwargs,
        )

        # Convert response
        return self._convert_response(completion)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: "ModelRequestParameters",
    ) -> AsyncIterator["EsperantoStreamedResponse"]:
        """
        Make a streaming request to the model.

        Args:
            messages: Conversation history in Pydantic AI format
            model_settings: Optional settings overrides
            model_request_parameters: Tool definitions and output configuration

        Yields:
            EsperantoStreamedResponse that can be iterated for events
        """
        # Convert messages to Esperanto format
        esperanto_messages = self._convert_messages(messages)

        # Convert tools
        tools = self._convert_tools(model_request_parameters)

        # Get settings overrides
        settings_kwargs = self._apply_settings(model_settings)

        # Make the streaming request via Esperanto
        stream = await self._esperanto_model.achat_complete(
            messages=esperanto_messages,
            stream=True,
            tools=tools,
            **settings_kwargs,
        )

        # Yield the streamed response wrapper
        yield EsperantoStreamedResponse(
            stream=stream,
            model_name=self._model_name,
            provider_name=self._system,
            model_request_parameters=model_request_parameters,
        )


@dataclass
class EsperantoStreamedResponse(StreamedResponse):  # type: ignore[misc]
    """
    Streaming response handler for Esperanto models.

    Implements Pydantic AI's StreamedResponse interface to handle
    streaming responses from Esperanto models.
    """

    # Additional fields for our implementation
    _stream: Any = field(default=None, init=False)
    _model_name_value: str = field(default="", init=False)
    _provider_name_value: str = field(default="", init=False)
    _timestamp_value: datetime = field(default_factory=_now_utc, init=False)

    def __init__(
        self,
        stream: Any,
        model_name: str,
        provider_name: str,
        model_request_parameters: Any = None,
    ) -> None:
        """Initialize the streamed response."""
        # Import here to avoid circular issues
        if PYDANTIC_AI_INSTALLED:
            from pydantic_ai.models import ModelResponsePartsManager
            from pydantic_ai.usage import RequestUsage as PARequestUsage

            # Initialize parent dataclass fields
            object.__setattr__(self, "model_request_parameters", model_request_parameters)
            object.__setattr__(self, "final_result_event", None)
            object.__setattr__(self, "provider_response_id", None)
            object.__setattr__(self, "provider_details", None)
            object.__setattr__(self, "finish_reason", None)
            object.__setattr__(self, "_parts_manager", ModelResponsePartsManager())
            object.__setattr__(self, "_event_iterator", None)
            object.__setattr__(self, "_usage", PARequestUsage())

        # Initialize our own fields
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_model_name_value", model_name)
        object.__setattr__(self, "_provider_name_value", provider_name)
        object.__setattr__(self, "_timestamp_value", _now_utc())

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name_value

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self._provider_name_value

    @property
    def provider_url(self) -> str | None:
        """Return the provider URL (not available from Esperanto)."""
        return None

    @property
    def timestamp(self) -> datetime:
        """Return the response timestamp."""
        return self._timestamp_value

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """
        Return an async iterator of ModelResponseStreamEvent.

        This is the main streaming implementation required by StreamedResponse.
        """
        text_vendor_id: str | None = None
        tool_vendor_ids: dict[int, str] = {}  # stream index -> vendor id

        async for chunk in self._stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if finish_reason:
                self.finish_reason = _FINISH_REASON_MAP.get(finish_reason, finish_reason)

            # Handle text content
            if delta.content:
                if text_vendor_id is None:
                    text_vendor_id = "text-0"
                    new_part = TextPart(content="")
                    event = self._parts_manager.handle_part(
                        vendor_part_id=text_vendor_id,
                        part=new_part,
                    )
                    yield event

                # Use handle_text_delta to accumulate content
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id=text_vendor_id,
                    content=delta.content,
                ):
                    yield event

            # Handle streaming tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_index = tc.get("index", 0) if isinstance(tc, dict) else getattr(tc, "index", 0)
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    tc_function = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})

                    if tc_id and tc_index not in tool_vendor_ids:
                        # New tool call starting
                        vendor_id = f"tool-{tc_index}-{tc_id}"
                        tool_vendor_ids[tc_index] = vendor_id

                        func_name = tc_function.get("name", "") if isinstance(tc_function, dict) else getattr(tc_function, "name", "")

                        event = self._parts_manager.handle_tool_call_part(
                            vendor_part_id=vendor_id,
                            tool_name=func_name,
                            args="",
                            tool_call_id=tc_id,
                        )
                        yield event

                    # Accumulate arguments
                    if tc_index in tool_vendor_ids:
                        func_args = tc_function.get("arguments", "") if isinstance(tc_function, dict) else getattr(tc_function, "arguments", "")
                        if func_args:
                            vendor_id = tool_vendor_ids[tc_index]
                            event = self._parts_manager.handle_tool_call_delta(
                                vendor_part_id=vendor_id,
                                args=func_args,
                            )
                            if event:
                                yield event
