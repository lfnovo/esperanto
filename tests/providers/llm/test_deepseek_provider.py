import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import Tool, ToolFunction, ToolCall, ToolCallValidationError
from esperanto.providers.llm.deepseek import DeepSeekLanguageModel


def test_provider_name():
    model = DeepSeekLanguageModel(api_key="test-key")
    assert model.provider == "deepseek"


def test_client_properties():
    model = DeepSeekLanguageModel(api_key="test-key")
    assert model.client is not None
    assert model.async_client is not None
    # Check HTTP client properties
    assert hasattr(model.client, "post")
    assert hasattr(model.async_client, "post")
    assert hasattr(model.client, "get")
    assert hasattr(model.async_client, "get")


def test_initialization_with_api_key():
    model = DeepSeekLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"


def test_initialization_with_env_var():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-test-key"}):
        model = DeepSeekLanguageModel()
        assert model.api_key == "env-test-key"


def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="DeepSeek API key not found"):
            DeepSeekLanguageModel()


def test_default_model_name():
    model = DeepSeekLanguageModel(api_key="test-key")
    assert model.model_name == "deepseek-chat"


def test_to_langchain():
    model = DeepSeekLanguageModel(api_key="test-key")
    lc = model.to_langchain()
    # Should be ChatDeepSeek or compatible
    assert lc is not None


def test_initialization_with_api_key_in_config():
    """Test that api_key can be passed via config dict (GitHub issue #68)."""
    model = DeepSeekLanguageModel(config={"api_key": "config-test-key"})
    assert model.api_key == "config-test-key"
    assert model.base_url == "https://api.deepseek.com/v1"


def test_initialization_with_base_url_in_config():
    """Test that base_url can be passed via config dict."""
    model = DeepSeekLanguageModel(
        api_key="test-key",
        config={"base_url": "https://custom.deepseek.com/v1"}
    )
    assert model.base_url == "https://custom.deepseek.com/v1"


def test_initialization_with_api_key_and_base_url_in_config():
    """Test that both api_key and base_url can be passed via config dict."""
    model = DeepSeekLanguageModel(
        config={
            "api_key": "config-test-key",
            "base_url": "https://custom.deepseek.com/v1"
        }
    )
    assert model.api_key == "config-test-key"
    assert model.base_url == "https://custom.deepseek.com/v1"


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
def mock_deepseek_tool_call_response():
    """Mock HTTP response for DeepSeek chat completions with tool calls."""
    return {
        "id": "chatcmpl-tool-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepseek-chat",
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
def deepseek_model():
    """Create a DeepSeek model with mocked HTTP client."""
    model = DeepSeekLanguageModel(api_key="test-key", model_name="deepseek-chat")

    client = Mock()
    async_client = AsyncMock()

    mock_response_data = {
        "id": "chatcmpl-123",
        "created": 1677858242,
        "model": "deepseek-chat",
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
def deepseek_model_with_tool_response(mock_deepseek_tool_call_response):
    """Create a DeepSeek model with tool call response mocked."""
    model = DeepSeekLanguageModel(api_key="test-key", model_name="deepseek-chat")

    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    client.post.return_value = make_response(200, mock_deepseek_tool_call_response)
    async_client.post.return_value = make_response(200, mock_deepseek_tool_call_response)

    model.client = client
    model.async_client = async_client
    return model


class TestToolConversion:
    """Tests for tool conversion to OpenAI format (DeepSeek uses OpenAI-compatible format)."""

    def test_convert_single_tool(self, deepseek_model, sample_tools):
        """Test converting a single tool to OpenAI format."""
        result = deepseek_model._convert_tools_to_openai([sample_tools[0]])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the current weather for a location"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "location" in result[0]["function"]["parameters"]["properties"]

    def test_convert_multiple_tools(self, deepseek_model, sample_tools):
        """Test converting multiple tools to OpenAI format."""
        result = deepseek_model._convert_tools_to_openai(sample_tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "get_time"

    def test_convert_none_tools(self, deepseek_model):
        """Test converting None returns None."""
        result = deepseek_model._convert_tools_to_openai(None)
        assert result is None

    def test_convert_empty_tools(self, deepseek_model):
        """Test converting empty list returns None."""
        result = deepseek_model._convert_tools_to_openai([])
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, deepseek_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = deepseek_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = deepseek_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
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

    def test_chat_complete_with_tool_choice(self, deepseek_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        deepseek_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = deepseek_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, deepseek_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await deepseek_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = deepseek_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, deepseek_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = deepseek_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, validate_tool_calls=True
        )

        assert response.choices[0].message.tool_calls is not None

    def test_validation_fails_for_invalid_tool_call(self, sample_tools):
        """Test that validation fails for invalid tool calls."""
        pytest.importorskip("jsonschema")

        # Create a response with invalid tool call (missing required 'location')
        invalid_response = {
            "id": "chatcmpl-invalid",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_invalid",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"unit": "celsius"}'  # Missing required 'location'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

        model = DeepSeekLanguageModel(api_key="test-key", model_name="deepseek-chat")
        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = invalid_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]

        with pytest.raises(ToolCallValidationError):
            model.chat_complete(messages, tools=sample_tools, validate_tool_calls=True)
