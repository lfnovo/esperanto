"""Unit tests for the Pydantic AI adapter."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from esperanto.common_types import (
    ChatCompletion,
    Choice,
    FunctionCall,
    Message,
    Tool,
    ToolCall,
    ToolFunction,
    Usage,
)
from esperanto.integrations.pydantic_ai import (
    EsperantoPydanticModel,
    EsperantoStreamedResponse,
)


class TestEsperantoPydanticModelInit:
    """Tests for EsperantoPydanticModel initialization."""

    def test_init_stores_model_reference(self):
        """Test that init stores the Esperanto model reference."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"

        adapter = EsperantoPydanticModel(mock_model)

        assert adapter._esperanto_model is mock_model

    def test_model_name_property(self):
        """Test model_name property returns correct value."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "claude-sonnet-4-20250514"
        mock_model.provider = "anthropic"

        adapter = EsperantoPydanticModel(mock_model)

        assert adapter.model_name == "claude-sonnet-4-20250514"

    def test_system_property(self):
        """Test system property returns provider name."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"

        adapter = EsperantoPydanticModel(mock_model)

        assert adapter.system == "openai"


class TestMessageConversion:
    """Tests for _convert_messages method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter with a mock model."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"
        return EsperantoPydanticModel(mock_model)

    def test_convert_system_prompt(self, adapter):
        """Test converting SystemPromptPart to Esperanto format."""
        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="You are a helpful assistant.")
            ])
        ]

        result = adapter._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."

    def test_convert_user_prompt_string(self, adapter):
        """Test converting UserPromptPart with string content."""
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content="Hello!")
            ])
        ]

        result = adapter._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_convert_model_response_text(self, adapter):
        """Test converting ModelResponse with TextPart."""
        messages = [
            ModelResponse(parts=[
                TextPart(content="Hi there!")
            ])
        ]

        result = adapter._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there!"

    def test_convert_tool_return_part(self, adapter):
        """Test converting ToolReturnPart."""
        messages = [
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="get_weather",
                    content="Sunny, 22C",
                    tool_call_id="call_123"
                )
            ])
        ]

        result = adapter._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"] == "Sunny, 22C"

    def test_convert_tool_call_part(self, adapter):
        """Test converting ModelResponse with ToolCallPart."""
        messages = [
            ModelResponse(parts=[
                ToolCallPart(
                    tool_name="get_weather",
                    args={"city": "Tokyo"},
                    tool_call_id="call_456"
                )
            ])
        ]

        result = adapter._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_456"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_convert_retry_prompt_part(self, adapter):
        """Test converting RetryPromptPart."""
        messages = [
            ModelRequest(parts=[
                RetryPromptPart(content="Invalid format, please try again.")
            ])
        ]

        result = adapter._convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Invalid format" in result[0]["content"]

    def test_convert_mixed_messages(self, adapter):
        """Test converting a conversation with mixed message types."""
        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="You are helpful."),
                UserPromptPart(content="Hello!"),
            ]),
            ModelResponse(parts=[
                TextPart(content="Hi! How can I help?")
            ]),
            ModelRequest(parts=[
                UserPromptPart(content="What's 2+2?")
            ]),
        ]

        result = adapter._convert_messages(messages)

        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"


class TestResponseConversion:
    """Tests for _convert_response method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter with a mock model."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"
        return EsperantoPydanticModel(mock_model)

    def test_convert_text_response(self, adapter):
        """Test converting a simple text response."""
        completion = ChatCompletion(
            id="test-123",
            choices=[
                Choice(
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                    finish_reason="stop"
                )
            ],
            model="gpt-4o",
            provider="openai",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )

        result = adapter._convert_response(completion)

        assert len(result.parts) == 1
        assert isinstance(result.parts[0], TextPart)
        assert result.parts[0].content == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.finish_reason == "stop"

    def test_convert_response_with_tool_calls(self, adapter):
        """Test converting a response with tool calls."""
        completion = ChatCompletion(
            id="test-456",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="call_abc",
                                type="function",
                                function=FunctionCall(
                                    name="get_weather",
                                    arguments='{"city": "Tokyo"}'
                                )
                            )
                        ]
                    ),
                    finish_reason="tool_calls"
                )
            ],
            model="gpt-4o",
            provider="openai",
            usage=Usage(prompt_tokens=15, completion_tokens=10, total_tokens=25)
        )

        result = adapter._convert_response(completion)

        assert len(result.parts) == 1
        assert isinstance(result.parts[0], ToolCallPart)
        assert result.parts[0].tool_name == "get_weather"
        assert result.parts[0].args == {"city": "Tokyo"}
        assert result.parts[0].tool_call_id == "call_abc"
        assert result.finish_reason == "tool_calls"

    def test_convert_response_with_thinking(self, adapter):
        """Test converting a response with thinking tags."""
        completion = ChatCompletion(
            id="test-789",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        content="<think>Let me think...</think>\n\nThe answer is 42.",
                        role="assistant"
                    ),
                    finish_reason="stop"
                )
            ],
            model="gpt-4o",
            provider="openai",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )

        result = adapter._convert_response(completion)

        assert len(result.parts) == 2
        assert isinstance(result.parts[0], ThinkingPart)
        assert result.parts[0].content == "Let me think..."
        assert isinstance(result.parts[1], TextPart)
        assert result.parts[1].content == "The answer is 42."

    def test_convert_response_no_usage(self, adapter):
        """Test converting a response with no usage info."""
        completion = ChatCompletion(
            id="test-000",
            choices=[
                Choice(
                    index=0,
                    message=Message(content="Hello!", role="assistant"),
                    finish_reason="stop"
                )
            ],
            model="gpt-4o",
            provider="openai",
            usage=None
        )

        result = adapter._convert_response(completion)

        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


class TestToolConversion:
    """Tests for _convert_tools method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter with a mock model."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"
        return EsperantoPydanticModel(mock_model)

    def test_convert_tools_empty(self, adapter):
        """Test converting empty tool list."""
        params = MagicMock()
        params.function_tools = []
        params.output_tools = []

        result = adapter._convert_tools(params)

        assert result is None

    def test_convert_tools_single(self, adapter):
        """Test converting a single tool."""
        tool_def = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters_json_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        )
        params = MagicMock()
        params.function_tools = [tool_def]
        params.output_tools = []

        result = adapter._convert_tools(params)

        assert result is not None
        assert len(result) == 1
        # Result is now Tool objects, not dicts
        assert result[0].type == "function"
        assert result[0].function.name == "get_weather"
        assert result[0].function.description == "Get weather for a city"


class TestUsageConversion:
    """Tests for _convert_usage method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter with a mock model."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"
        return EsperantoPydanticModel(mock_model)

    def test_convert_usage_normal(self, adapter):
        """Test converting normal usage."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        result = adapter._convert_usage(usage)

        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_convert_usage_none(self, adapter):
        """Test converting None usage."""
        result = adapter._convert_usage(None)

        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestFinishReasonMapping:
    """Tests for _map_finish_reason method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter with a mock model."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"
        return EsperantoPydanticModel(mock_model)

    @pytest.mark.parametrize("esperanto_reason,expected", [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("content_filter", "content_filter"),
        (None, None),
        ("unknown", "unknown"),
    ])
    def test_map_finish_reason(self, adapter, esperanto_reason, expected):
        """Test mapping various finish reasons."""
        result = adapter._map_finish_reason(esperanto_reason)
        assert result == expected


class TestApplySettings:
    """Tests for _apply_settings method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter with a mock model."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"
        return EsperantoPydanticModel(mock_model)

    def test_apply_settings_none(self, adapter):
        """Test applying None settings."""
        result = adapter._apply_settings(None)
        assert result == {}

    def test_apply_settings_with_values(self, adapter):
        """Test applying settings with values."""
        settings = ModelSettings(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )

        result = adapter._apply_settings(settings)

        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9


class TestRequestMethod:
    """Tests for the request method."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter with a mock model."""
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "gpt-4o"
        mock_model.provider = "openai"

        # Setup async mock for achat_complete
        async def mock_achat_complete(*args, **kwargs):
            return ChatCompletion(
                id="test-request",
                choices=[
                    Choice(
                        index=0,
                        message=Message(content="Test response", role="assistant"),
                        finish_reason="stop"
                    )
                ],
                model="gpt-4o",
                provider="openai",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            )

        mock_model.achat_complete = mock_achat_complete
        return EsperantoPydanticModel(mock_model)

    @pytest.mark.asyncio
    async def test_request_basic(self, adapter):
        """Test basic request flow."""
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content="Hello!")
            ])
        ]
        params = MagicMock()
        params.function_tools = []
        params.output_tools = []

        result = await adapter.request(messages, None, params)

        assert isinstance(result, ModelResponse)
        assert len(result.parts) == 1
        assert isinstance(result.parts[0], TextPart)
        assert result.parts[0].content == "Test response"


class TestEsperantoStreamedResponse:
    """Tests for EsperantoStreamedResponse class."""

    def test_init(self):
        """Test initialization."""
        mock_stream = AsyncMock()

        response = EsperantoStreamedResponse(
            stream=mock_stream,
            model_name="gpt-4o",
            provider_name="openai"
        )

        # Use property accessors since internal names may vary
        assert response.model_name == "gpt-4o"
        assert response.provider_name == "openai"
        # _stream is our internal field
        assert response._stream is mock_stream

    def test_properties(self):
        """Test property methods."""
        mock_stream = AsyncMock()

        response = EsperantoStreamedResponse(
            stream=mock_stream,
            model_name="gpt-4o",
            provider_name="openai"
        )

        assert response.model_name == "gpt-4o"
        assert response.provider_name == "openai"
        assert response.provider_url is None
        assert response.timestamp is not None

    def test_usage_empty(self):
        """Test usage when no usage data."""
        mock_stream = AsyncMock()

        response = EsperantoStreamedResponse(
            stream=mock_stream,
            model_name="gpt-4o",
            provider_name="openai"
        )

        usage = response.usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_get_empty(self):
        """Test get() with no accumulated data."""
        mock_stream = AsyncMock()

        response = EsperantoStreamedResponse(
            stream=mock_stream,
            model_name="gpt-4o",
            provider_name="openai"
        )

        result = response.get()

        assert isinstance(result, ModelResponse)
        assert len(result.parts) == 0

    def test_get_with_parts_manager(self):
        """Test get() with content added via parts manager."""
        mock_stream = AsyncMock()

        response = EsperantoStreamedResponse(
            stream=mock_stream,
            model_name="gpt-4o",
            provider_name="openai"
        )

        # Simulate adding content through the parts manager (as streaming would)
        text_part = TextPart(content="Hello, world!")
        # Use handle_part with keyword arguments
        response._parts_manager.handle_part(vendor_part_id="text-0", part=text_part)

        result = response.get()

        assert len(result.parts) == 1
        assert isinstance(result.parts[0], TextPart)
        assert result.parts[0].content == "Hello, world!"
