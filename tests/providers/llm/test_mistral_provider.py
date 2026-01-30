import os
from unittest.mock import AsyncMock, Mock, patch
import pytest
from esperanto.providers.llm.mistral import MistralLanguageModel
from esperanto.common_types import (
    ChatCompletion, Choice, Message, Usage,
    Tool, ToolFunction, ToolCall, ToolCallValidationError
)

@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for Mistral API."""
    def create_response():
        return {
            "id": "cmpl-123",
            "object": "chat.completion",
            "created": 123,
            "model": "mistral-large-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5
            }
        }
    return create_response

@pytest.fixture
def mistral_model(mock_httpx_response):
    """Create MistralLanguageModel with mocked HTTP client."""
    model = MistralLanguageModel(api_key="test-key", model_name="mistral-large-latest")
    
    # Mock the HTTP clients
    mock_client = Mock()
    mock_async_client = AsyncMock()
    
    def mock_post(url, **kwargs):
        response_data = mock_httpx_response()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        return mock_response
    
    async def mock_async_post(url, **kwargs):
        response_data = mock_httpx_response()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        return mock_response
    
    mock_client.post = mock_post
    mock_async_client.post = mock_async_post
    
    model.client = mock_client
    model.async_client = mock_async_client
    return model

def test_provider_name(mistral_model):
    assert mistral_model.provider == "mistral"

def test_initialization_with_api_key():
    model = MistralLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"

def test_initialization_with_env_var(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "env-test-key")
    model = MistralLanguageModel()
    assert model.api_key == "env-test-key"

def test_initialization_without_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Mistral API key not found"):
        MistralLanguageModel()

def test_chat_complete(mistral_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = mistral_model.chat_complete(messages)
    assert response.choices[0].message.content == "Hello!"

async def test_achat_complete(mistral_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = await mistral_model.achat_complete(messages)
    assert response.choices[0].message.content == "Hello!"

def test_to_langchain(mistral_model):
    # Only run if langchain_mistralai is installed
    try:
        lc = mistral_model.to_langchain()
        assert lc is not None
    except ImportError:
        pytest.skip("langchain_mistralai not installed")


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
def mock_mistral_tool_call_response():
    """Mock HTTP response for Mistral chat completions with tool calls."""
    return {
        "id": "chatcmpl-tool-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "mistral-large-latest",
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
def mistral_model_with_tool_response(mock_mistral_tool_call_response):
    """Create a Mistral model with tool call response mocked."""
    model = MistralLanguageModel(api_key="test-key", model_name="mistral-large-latest")

    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    client.post.return_value = make_response(200, mock_mistral_tool_call_response)
    async_client.post.return_value = make_response(200, mock_mistral_tool_call_response)

    model.client = client
    model.async_client = async_client
    return model


class TestToolConversion:
    """Tests for tool conversion to OpenAI format (Mistral uses OpenAI-compatible format)."""

    def test_convert_single_tool(self, mistral_model, sample_tools):
        """Test converting a single tool to OpenAI format."""
        result = mistral_model._convert_tools_to_openai([sample_tools[0]])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the current weather for a location"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "location" in result[0]["function"]["parameters"]["properties"]

    def test_convert_multiple_tools(self, mistral_model, sample_tools):
        """Test converting multiple tools to OpenAI format."""
        result = mistral_model._convert_tools_to_openai(sample_tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "get_time"

    def test_convert_none_tools(self, mistral_model):
        """Test converting None returns None."""
        result = mistral_model._convert_tools_to_openai(None)
        assert result is None

    def test_convert_empty_tools(self, mistral_model):
        """Test converting empty list returns None."""
        result = mistral_model._convert_tools_to_openai([])
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, mistral_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = mistral_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = mistral_model_with_tool_response.client.post.call_args
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

    def test_chat_complete_with_tool_choice(self, mistral_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        mistral_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = mistral_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, mistral_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await mistral_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = mistral_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestInstanceLevelTools:
    """Tests for instance-level tool configuration."""

    def test_instance_tools_used_when_no_call_tools(self, mock_mistral_tool_call_response, sample_tools):
        """Test that instance-level tools are used when not passed at call time."""
        model = MistralLanguageModel(
            api_key="test-key",
            model_name="mistral-large-latest",
            tools=sample_tools
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_mistral_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload
        assert len(json_payload["tools"]) == 2


class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, mistral_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = mistral_model_with_tool_response.chat_complete(
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
            "model": "mistral-large-latest",
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

        model = MistralLanguageModel(api_key="test-key", model_name="mistral-large-latest")
        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = invalid_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]

        with pytest.raises(ToolCallValidationError):
            model.chat_complete(messages, tools=sample_tools, validate_tool_calls=True)