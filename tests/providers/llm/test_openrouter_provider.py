import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import Tool, ToolFunction, ToolCall, ToolCallValidationError
from esperanto.providers.llm.openrouter import OpenRouterLanguageModel


def test_provider_name():
    model = OpenRouterLanguageModel(api_key="test-key")
    assert model.provider == "openrouter"

def test_initialization_with_api_key():
    model = OpenRouterLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"
    assert model.base_url == "https://openrouter.ai/api/v1"

def test_initialization_with_env_var():
    with patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "env-test-key",
        "OPENROUTER_BASE_URL": "https://custom.openrouter.ai/v1"
    }):
        model = OpenRouterLanguageModel()
        assert model.api_key == "env-test-key"
        assert model.base_url == "https://custom.openrouter.ai/v1"

def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenRouter API key not found"):
            OpenRouterLanguageModel()

def test_custom_base_url():
    model = OpenRouterLanguageModel(
        api_key="test-key",
        base_url="https://custom.openrouter.ai/v1"
    )
    assert model.base_url == "https://custom.openrouter.ai/v1"


# =============================================================================
# Tool Calling Tests
# =============================================================================


@pytest.fixture
def sample_tools():
    """Sample tools for testing."""
    return [
        Tool(
            function=ToolFunction(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            )
        ),
        Tool(
            function=ToolFunction(
                name="get_time",
                description="Get the current time for a timezone",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "The timezone"}
                    }
                }
            )
        )
    ]


@pytest.fixture
def mock_openrouter_tool_call_response():
    """Mock HTTP response for OpenRouter chat completions with tool calls."""
    return {
        "id": "chatcmpl-tool-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "openai/gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco", "unit": "celsius"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    }


@pytest.fixture
def openrouter_model():
    """Create an OpenRouter model with mocked HTTP client."""
    model = OpenRouterLanguageModel(api_key="test-key", model_name="openai/gpt-4")

    client = Mock()
    async_client = AsyncMock()

    mock_response_data = {
        "id": "chatcmpl-123",
        "created": 1677858242,
        "model": "openai/gpt-4",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }
        ],
        "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30}
    }

    def mock_post(url, **kwargs):
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_response_data
        return response

    async def mock_async_post(url, **kwargs):
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_response_data
        return response

    client.post.side_effect = mock_post
    async_client.post.side_effect = mock_async_post

    model.client = client
    model.async_client = async_client
    return model


@pytest.fixture
def openrouter_model_with_tool_response(mock_openrouter_tool_call_response):
    """Create an OpenRouter model with tool call response mocked."""
    model = OpenRouterLanguageModel(api_key="test-key", model_name="openai/gpt-4")

    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    client.post.return_value = make_response(200, mock_openrouter_tool_call_response)
    async_client.post.return_value = make_response(200, mock_openrouter_tool_call_response)

    model.client = client
    model.async_client = async_client
    return model


class TestToolConversion:
    """Tests for tool conversion to OpenAI format (OpenRouter uses OpenAI-compatible format)."""

    def test_convert_single_tool(self, openrouter_model, sample_tools):
        """Test converting a single tool to OpenAI format."""
        result = openrouter_model._convert_tools_to_openai([sample_tools[0]])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the current weather for a location"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "location" in result[0]["function"]["parameters"]["properties"]

    def test_convert_multiple_tools(self, openrouter_model, sample_tools):
        """Test converting multiple tools to OpenAI format."""
        result = openrouter_model._convert_tools_to_openai(sample_tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "get_time"

    def test_convert_none_tools(self, openrouter_model):
        """Test converting None returns None."""
        result = openrouter_model._convert_tools_to_openai(None)
        assert result is None

    def test_convert_empty_tools(self, openrouter_model):
        """Test converting empty list returns None."""
        result = openrouter_model._convert_tools_to_openai([])
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, openrouter_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = openrouter_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools (OpenRouter uses data= with JSON string, not json=)
        call_args = openrouter_model_with_tool_response.client.post.call_args
        json_payload = json.loads(call_args.kwargs["data"])
        assert "tools" in json_payload
        assert len(json_payload["tools"]) == 2

        # Check response has tool calls
        assert len(response.choices) == 1
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tool_call = response.choices[0].message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id == "call_abc123"
        assert tool_call.function.name == "get_weather"
        assert '"location": "San Francisco"' in tool_call.function.arguments

    def test_chat_complete_with_tool_choice(self, openrouter_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        openrouter_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        # OpenRouter uses data= with JSON string, not json=
        call_args = openrouter_model_with_tool_response.client.post.call_args
        json_payload = json.loads(call_args.kwargs["data"])
        assert json_payload["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, openrouter_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await openrouter_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools (OpenRouter uses data= with JSON string, not json=)
        call_args = openrouter_model_with_tool_response.async_client.post.call_args
        json_payload = json.loads(call_args.kwargs["data"])
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, openrouter_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = openrouter_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, validate_tool_calls=True
        )

        assert response.choices[0].message.tool_calls is not None
