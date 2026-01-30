import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import Tool, ToolFunction, ToolCall, ToolCallValidationError

try:
    from langchain_groq import ChatGroq

    from esperanto.providers.llm.groq import GroqLanguageModel

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    pytestmark = pytest.mark.skip("Groq not installed")


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_provider_name(groq_model):
    assert groq_model.provider == "groq"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_client_properties(groq_model):
    """Test that client properties are properly initialized."""
    # Verify clients are not None
    assert groq_model.client is not None
    assert groq_model.async_client is not None

    # Verify clients have expected HTTP methods (httpx)
    assert hasattr(groq_model.client, "post")
    assert hasattr(groq_model.async_client, "post")
    
    # Verify API key is set
    assert groq_model.api_key == "test-key"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_initialization_with_api_key():
    model = GroqLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_initialization_with_env_var():
    with patch.dict(os.environ, {"GROQ_API_KEY": "env-test-key"}):
        model = GroqLanguageModel()
        assert model.api_key == "env-test-key"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Groq API key not found"):
            GroqLanguageModel()


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_chat_complete(groq_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = groq_model.chat_complete(messages)

    # Verify the client was called with correct parameters
    groq_model.client.post.assert_called_once()
    call_args = groq_model.client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.groq.com/openai/v1/chat/completions"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["messages"] == messages
    assert json_payload["model"] == "mixtral-8x7b-32768"
    assert json_payload["temperature"] == 1.0
    assert not json_payload["stream"]
    
    # Check response
    assert response.choices[0].message.content == "Test response"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
@pytest.mark.asyncio
async def test_achat_complete(groq_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = await groq_model.achat_complete(messages)

    # Verify the async client was called with correct parameters
    groq_model.async_client.post.assert_called_once()
    call_args = groq_model.async_client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.groq.com/openai/v1/chat/completions"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["messages"] == messages
    assert json_payload["model"] == "mixtral-8x7b-32768"
    assert json_payload["temperature"] == 1.0
    assert not json_payload["stream"]
    
    # Check response
    assert response.choices[0].message.content == "Test response"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_to_langchain(groq_model):
    langchain_model = groq_model.to_langchain()

    assert isinstance(langchain_model, ChatGroq)
    assert langchain_model.model_name == "mixtral-8x7b-32768"
    assert langchain_model.temperature == 1.0
    assert langchain_model.max_tokens == 850
    # assert langchain_model.model_kwargs["top_p"] == 0.9 # top_p is not stored in model_kwargs by default
    assert langchain_model.streaming == False
    assert langchain_model.groq_api_key.get_secret_value() == "test-key"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_response_normalization(groq_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = groq_model.chat_complete(messages)

    assert response.id == "chatcmpl-123"
    assert response.created == 1677858242
    assert response.model == "mixtral-8x7b-32768"
    assert response.provider == "groq"
    assert len(response.choices) == 1

    choice = response.choices[0]
    assert choice.index == 0
    assert choice.message.content == "Test response"
    assert choice.message.role == "assistant"
    assert choice.finish_reason == "stop"

    assert response.usage.completion_tokens == 10
    assert response.usage.prompt_tokens == 8
    assert response.usage.total_tokens == 18


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
def mock_groq_tool_call_response():
    """Mock HTTP response for Groq chat completions with tool calls."""
    return {
        "id": "chatcmpl-tool-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "mixtral-8x7b-32768",
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
def groq_model_with_tool_response(mock_groq_tool_call_response):
    """Create a Groq model with tool call response mocked."""
    model = GroqLanguageModel(api_key="test-key", model_name="mixtral-8x7b-32768")

    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    client.post.return_value = make_response(200, mock_groq_tool_call_response)
    async_client.post.return_value = make_response(200, mock_groq_tool_call_response)

    model.client = client
    model.async_client = async_client
    return model


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
class TestToolConversion:
    """Tests for tool conversion to OpenAI format (Groq uses OpenAI-compatible format)."""

    def test_convert_single_tool(self, groq_model, sample_tools):
        """Test converting a single tool to OpenAI format."""
        result = groq_model._convert_tools_to_openai([sample_tools[0]])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the current weather for a location"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "location" in result[0]["function"]["parameters"]["properties"]

    def test_convert_multiple_tools(self, groq_model, sample_tools):
        """Test converting multiple tools to OpenAI format."""
        result = groq_model._convert_tools_to_openai(sample_tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "get_time"

    def test_convert_none_tools(self, groq_model):
        """Test converting None returns None."""
        result = groq_model._convert_tools_to_openai(None)
        assert result is None

    def test_convert_empty_tools(self, groq_model):
        """Test converting empty list returns None."""
        result = groq_model._convert_tools_to_openai([])
        assert result is None


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, groq_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = groq_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = groq_model_with_tool_response.client.post.call_args
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

    def test_chat_complete_with_tool_choice(self, groq_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        groq_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = groq_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == "required"

    def test_chat_complete_with_parallel_tool_calls_disabled(
        self, groq_model_with_tool_response, sample_tools
    ):
        """Test chat_complete with parallel_tool_calls=False."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        groq_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, parallel_tool_calls=False
        )

        call_args = groq_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, groq_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await groq_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = groq_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
class TestInstanceLevelTools:
    """Tests for instance-level tool configuration."""

    def test_instance_tools_used_when_no_call_tools(self, mock_groq_tool_call_response, sample_tools):
        """Test that instance-level tools are used when not passed at call time."""
        model = GroqLanguageModel(
            api_key="test-key",
            model_name="mixtral-8x7b-32768",
            tools=sample_tools
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_groq_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload
        assert len(json_payload["tools"]) == 2

    def test_call_tools_override_instance_tools(self, mock_groq_tool_call_response, sample_tools):
        """Test that call-time tools override instance-level tools."""
        instance_tool = Tool(
            function=ToolFunction(name="instance_tool", description="Instance tool")
        )
        model = GroqLanguageModel(
            api_key="test-key",
            model_name="mixtral-8x7b-32768",
            tools=[instance_tool]
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_groq_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages, tools=sample_tools)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        # Should have call-time tools, not instance tools
        assert json_payload["tools"][0]["function"]["name"] == "get_weather"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, groq_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = groq_model_with_tool_response.chat_complete(
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
            "model": "mixtral-8x7b-32768",
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

        model = GroqLanguageModel(api_key="test-key", model_name="mixtral-8x7b-32768")
        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = invalid_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]

        with pytest.raises(ToolCallValidationError):
            model.chat_complete(messages, tools=sample_tools, validate_tool_calls=True)
