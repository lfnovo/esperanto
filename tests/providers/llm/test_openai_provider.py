"""Tests for the OpenAI LLM provider."""
import os
from unittest.mock import AsyncMock, Mock, patch
import json

import pytest

from esperanto.providers.llm.openai import OpenAILanguageModel


@pytest.fixture
def mock_openai_chat_response():
    """Mock HTTP response for OpenAI chat completions API."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30
        }
    }


@pytest.fixture
def mock_openai_chat_stream_chunks():
    """Mock SSE chunks for OpenAI streaming chat completions."""
    return [
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        'data: [DONE]'
    ]


@pytest.fixture
def mock_openai_models_response():
    """Mock HTTP response for OpenAI models API."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4",
                "object": "model",
                "owned_by": "openai"
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "owned_by": "openai"
            },
            {
                "id": "whisper-1",
                "object": "model",
                "owned_by": "openai-internal"
            }
        ]
    }


@pytest.fixture
def mock_httpx_clients(mock_openai_chat_response, mock_openai_models_response, mock_openai_chat_stream_chunks):
    """Mock httpx clients for OpenAI LLM."""
    client = Mock()
    async_client = AsyncMock()

    # Mock HTTP response objects
    def make_response(status_code, json_data=None, stream_lines=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        if stream_lines is not None:
            # Mock iter_text() method for streaming
            response.iter_text.return_value = stream_lines
        return response

    def make_async_response(status_code, json_data=None, stream_lines=None):
        response = Mock()  # Use regular Mock, not AsyncMock
        response.status_code = status_code
        if json_data is not None:
            # Make json() synchronous like httpx does
            response.json.return_value = json_data
        if stream_lines is not None:
            async def async_iter():
                for line in stream_lines:
                    yield line
            # Mock aiter_text() method for async streaming
            response.aiter_text = async_iter
        return response

    # Configure responses based on URL and payload
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/chat/completions"):
            json_payload = kwargs.get("json", {})
            if json_payload.get("stream"):
                return make_response(200, stream_lines=mock_openai_chat_stream_chunks)
            else:
                return make_response(200, json_data=mock_openai_chat_response)
        return make_response(404, json_data={"error": "Not found"})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_response(200, json_data=mock_openai_models_response)
        return make_response(404, json_data={"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/chat/completions"):
            json_payload = kwargs.get("json", {})
            if json_payload.get("stream"):
                return make_async_response(200, stream_lines=mock_openai_chat_stream_chunks)
            else:
                return make_async_response(200, json_data=mock_openai_chat_response)
        return make_async_response(404, json_data={"error": "Not found"})

    async def mock_async_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_async_response(200, json_data=mock_openai_models_response)
        return make_async_response(404, json_data={"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    async_client.get.side_effect = mock_async_get_side_effect

    return client, async_client


@pytest.fixture
def openai_model(mock_httpx_clients):
    """Create an OpenAI model instance with mocked HTTP clients."""
    model = OpenAILanguageModel(
        api_key="test-key",
        model_name="gpt-4"
    )
    model.client, model.async_client = mock_httpx_clients
    return model


def test_provider_name(openai_model):
    assert openai_model.provider == "openai"


def test_initialization_with_api_key():
    model = OpenAILanguageModel(api_key="test-key")
    assert model.api_key == "test-key"


def test_initialization_with_env_var():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
        model = OpenAILanguageModel()
        assert model.api_key == "env-test-key"


def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAILanguageModel()


def test_models(openai_model):
    """Test that the models property works with HTTP."""
    models = openai_model.models
    
    # Verify HTTP GET was called
    openai_model.client.get.assert_called_with(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }
    )
    
    # Check that only GPT models are returned
    assert len(models) == 2
    assert models[0].id == "gpt-4"
    assert models[1].id == "gpt-3.5-turbo"
    # Model type is None when not explicitly provided by the API
    assert models[0].type is None
    assert models[1].type is None


def test_chat_complete(openai_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = openai_model.chat_complete(messages)

    # Verify HTTP POST was called
    openai_model.client.post.assert_called_once()
    call_args = openai_model.client.post.call_args

    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"

    # Check JSON payload
    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "gpt-4"
    assert json_payload["messages"] == messages
    assert json_payload["stream"] == False
    assert json_payload["temperature"] == 1.0

    # Verify response structure
    assert response.id == "chatcmpl-123"
    assert response.created == 1677652288
    assert response.model == "gpt-4"
    assert response.provider == "openai"

    # Verify choices
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.index == 0
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hello! How can I help you today?"

    # Verify usage
    assert response.usage.completion_tokens == 10
    assert response.usage.prompt_tokens == 20
    assert response.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_achat_complete(openai_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = await openai_model.achat_complete(messages)

    # Verify async HTTP POST was called
    openai_model.async_client.post.assert_called_once()
    call_args = openai_model.async_client.post.call_args

    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"

    # Check JSON payload
    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "gpt-4"
    assert json_payload["messages"] == messages
    assert json_payload["stream"] == False
    assert json_payload["temperature"] == 1.0

    # Verify response structure
    assert response.id == "chatcmpl-123"
    assert response.created == 1677652288
    assert response.model == "gpt-4"
    assert response.provider == "openai"

    # Verify choices
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.index == 0
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hello! How can I help you today?"

    # Verify usage
    assert response.usage.completion_tokens == 10
    assert response.usage.prompt_tokens == 20
    assert response.usage.total_tokens == 30


def test_chat_complete_streaming(openai_model):
    messages = [{"role": "user", "content": "Hello!"}]
    
    # Test streaming
    chunks = list(openai_model.chat_complete(messages, stream=True))

    # Verify HTTP POST was called with stream=True
    openai_model.client.post.assert_called_once()
    call_args = openai_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["stream"] == True

    # Verify we got chunks
    assert len(chunks) == 3  # 3 chunks before [DONE]
    
    # Check first chunk
    first_chunk = chunks[0]
    assert first_chunk.id == "chatcmpl-123"
    assert first_chunk.model == "gpt-4"
    assert len(first_chunk.choices) == 1
    assert first_chunk.choices[0].delta.role == "assistant"
    assert first_chunk.choices[0].delta.content == "Hello"


@pytest.mark.asyncio
async def test_achat_complete_streaming(openai_model):
    messages = [{"role": "user", "content": "Hello!"}]
    
    # Test async streaming
    chunks = []
    async for chunk in await openai_model.achat_complete(messages, stream=True):
        chunks.append(chunk)

    # Verify async HTTP POST was called with stream=True
    openai_model.async_client.post.assert_called_once()
    call_args = openai_model.async_client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["stream"] == True

    # Verify we got chunks
    assert len(chunks) == 3  # 3 chunks before [DONE]
    
    # Check first chunk
    first_chunk = chunks[0]
    assert first_chunk.id == "chatcmpl-123"
    assert first_chunk.model == "gpt-4"
    assert len(first_chunk.choices) == 1
    assert first_chunk.choices[0].delta.role == "assistant"
    assert first_chunk.choices[0].delta.content == "Hello"


def test_json_structured_output(openai_model):
    openai_model.structured = {"type": "json_object"}
    messages = [{"role": "user", "content": "Hello!"}]

    response = openai_model.chat_complete(messages)

    call_args = openai_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_structured_output_async(openai_model):
    openai_model.structured = {"type": "json_object"}
    messages = [{"role": "user", "content": "Hello!"}]

    response = await openai_model.achat_complete(messages)

    call_args = openai_model.async_client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["response_format"] == {"type": "json_object"}


def test_o1_model_transformations(openai_model):
    """Test that o1 models correctly transform parameters and messages."""
    openai_model.model_name = "o1-model"  # Set model to o1
    openai_model._config["model_name"] = "o1-model"  # Update config as well
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # Test synchronous completion
    response = openai_model.chat_complete(messages)
    call_args = openai_model.client.post.call_args
    json_payload = call_args[1]["json"]

    # Check message transformation
    assert json_payload["messages"] == [
        {"role": "user", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # Check parameter transformations
    assert "temperature" not in json_payload
    assert "top_p" not in json_payload
    assert "max_tokens" not in json_payload
    if "max_completion_tokens" in json_payload:
        assert json_payload["max_completion_tokens"] == openai_model.max_tokens


@pytest.mark.asyncio
async def test_o1_model_transformations_async(openai_model):
    """Test that o1 models correctly transform parameters and messages in async mode."""
    openai_model.model_name = "o1-model"  # Set model to o1
    openai_model._config["model_name"] = "o1-model"  # Update config as well
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # Test async completion
    await openai_model.achat_complete(messages)
    call_args = openai_model.async_client.post.call_args
    json_payload = call_args[1]["json"]

    # Check message transformation
    assert json_payload["messages"] == [
        {"role": "user", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # Check parameter transformations
    assert "temperature" not in json_payload
    assert "top_p" not in json_payload
    assert "max_tokens" not in json_payload
    if "max_completion_tokens" in json_payload:
        assert json_payload["max_completion_tokens"] == openai_model.max_tokens


def test_to_langchain(openai_model):
    # Test with structured output
    openai_model.structured = "json"
    langchain_model = openai_model.to_langchain()
    assert langchain_model.model_kwargs == {"response_format": {"type": "json_object"}}

    # Test model configuration
    assert langchain_model.model_name == "gpt-4"
    assert langchain_model.temperature == 1.0
    # Skip API key check since it's masked in SecretStr


def test_to_langchain_with_base_url(openai_model):
    openai_model.base_url = "https://custom.openai.com"
    langchain_model = openai_model.to_langchain()
    assert langchain_model.openai_api_base == "https://custom.openai.com"


def test_to_langchain_with_organization(openai_model):
    openai_model.organization = "test-org"
    langchain_model = openai_model.to_langchain()
    assert langchain_model.openai_organization == "test-org"


# =============================================================================
# Tool Calling Tests
# =============================================================================

from esperanto.common_types import (
    Tool,
    ToolFunction,
    ToolCall,
    FunctionCall,
    ToolCallValidationError,
)


@pytest.fixture
def mock_openai_tool_call_response():
    """Mock HTTP response for OpenAI chat completions with tool calls."""
    return {
        "id": "chatcmpl-tool-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
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
def mock_openai_parallel_tool_calls_response():
    """Mock HTTP response with multiple tool calls."""
    return {
        "id": "chatcmpl-tool-456",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
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
                                "arguments": '{"location": "San Francisco"}'
                            }
                        },
                        {
                            "id": "call_def456",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": '{"timezone": "PST"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 60,
            "completion_tokens": 30,
            "total_tokens": 90
        }
    }


@pytest.fixture
def sample_tools():
    """Create sample tools for testing."""
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
def openai_model_with_tool_response(mock_openai_tool_call_response):
    """Create an OpenAI model with tool call response mocked."""
    model = OpenAILanguageModel(api_key="test-key", model_name="gpt-4")

    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    def make_async_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    client.post.return_value = make_response(200, mock_openai_tool_call_response)
    async_client.post.return_value = make_async_response(200, mock_openai_tool_call_response)

    model.client = client
    model.async_client = async_client
    return model


class TestToolConversion:
    """Tests for tool conversion to OpenAI format."""

    def test_convert_single_tool(self, openai_model, sample_tools):
        """Test converting a single tool to OpenAI format."""
        result = openai_model._convert_tools_to_openai([sample_tools[0]])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the current weather for a location"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "location" in result[0]["function"]["parameters"]["properties"]

    def test_convert_multiple_tools(self, openai_model, sample_tools):
        """Test converting multiple tools to OpenAI format."""
        result = openai_model._convert_tools_to_openai(sample_tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "get_time"

    def test_convert_tool_with_strict_mode(self, openai_model):
        """Test that strict mode is included when specified."""
        tool = Tool(
            function=ToolFunction(
                name="strict_tool",
                description="A strict tool",
                parameters={"type": "object", "properties": {}},
                strict=True
            )
        )
        result = openai_model._convert_tools_to_openai([tool])

        assert result[0]["function"]["strict"] is True

    def test_convert_tool_without_strict_mode(self, openai_model, sample_tools):
        """Test that strict mode is not included when not specified."""
        result = openai_model._convert_tools_to_openai([sample_tools[0]])

        assert "strict" not in result[0]["function"]

    def test_convert_none_tools(self, openai_model):
        """Test converting None returns None."""
        result = openai_model._convert_tools_to_openai(None)
        assert result is None

    def test_convert_empty_tools(self, openai_model):
        """Test converting empty list returns None."""
        result = openai_model._convert_tools_to_openai([])
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, openai_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = openai_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = openai_model_with_tool_response.client.post.call_args
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

    def test_chat_complete_with_tool_choice(self, openai_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        openai_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = openai_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == "required"

    def test_chat_complete_with_specific_tool_choice(self, openai_model_with_tool_response, sample_tools):
        """Test chat_complete with specific tool choice."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        specific_choice = {"type": "function", "function": {"name": "get_weather"}}

        openai_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice=specific_choice
        )

        call_args = openai_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == specific_choice

    def test_chat_complete_with_parallel_tool_calls_disabled(
        self, openai_model_with_tool_response, sample_tools
    ):
        """Test chat_complete with parallel_tool_calls=False."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        openai_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, parallel_tool_calls=False
        )

        call_args = openai_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, openai_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await openai_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = openai_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestInstanceLevelTools:
    """Tests for instance-level tool configuration."""

    def test_instance_tools_used_when_no_call_tools(self, mock_openai_tool_call_response, sample_tools):
        """Test that instance-level tools are used when not passed at call time."""
        model = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4",
            tools=sample_tools
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_openai_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload
        assert len(json_payload["tools"]) == 2

    def test_call_tools_override_instance_tools(self, mock_openai_tool_call_response, sample_tools):
        """Test that call-time tools override instance-level tools."""
        instance_tool = Tool(
            function=ToolFunction(name="instance_tool", description="Instance tool")
        )
        model = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4",
            tools=[instance_tool]
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_openai_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages, tools=sample_tools)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        # Should have call-time tools, not instance tools
        assert json_payload["tools"][0]["function"]["name"] == "get_weather"

    def test_instance_tool_choice(self, mock_openai_tool_call_response, sample_tools):
        """Test instance-level tool_choice is used."""
        model = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4",
            tools=sample_tools,
            tool_choice="required"
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_openai_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == "required"


class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, openai_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = openai_model_with_tool_response.chat_complete(
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
            "model": "gpt-4",
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

        model = OpenAILanguageModel(api_key="test-key", model_name="gpt-4")
        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = invalid_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]

        with pytest.raises(ToolCallValidationError):
            model.chat_complete(messages, tools=sample_tools, validate_tool_calls=True)


class TestToolResultMessages:
    """Tests for handling tool result messages in conversations."""

    def test_tool_result_message_format(self, openai_model_with_tool_response, sample_tools):
        """Test that tool result messages are passed correctly."""
        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temperature": 72, "condition": "sunny"}'
            }
        ]

        openai_model_with_tool_response.chat_complete(messages, tools=sample_tools)

        call_args = openai_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]

        # Check tool result message is passed through
        assert len(json_payload["messages"]) == 3
        tool_msg = json_payload["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_abc123"


class TestNormalizeResponse:
    """Tests for response normalization with tool calls."""

    def test_normalize_response_with_tool_calls(self, openai_model):
        """Test that _normalize_response correctly extracts tool calls."""
        response_data = {
            "id": "chatcmpl-123",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "test_func",
                                    "arguments": '{"arg": "value"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        result = openai_model._normalize_response(response_data)

        assert len(result.choices) == 1
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1

        tool_call = result.choices[0].message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id == "call_abc"
        assert tool_call.type == "function"
        assert isinstance(tool_call.function, FunctionCall)
        assert tool_call.function.name == "test_func"
        assert tool_call.function.arguments == '{"arg": "value"}'

    def test_normalize_response_without_tool_calls(self, openai_model):
        """Test that _normalize_response handles responses without tool calls."""
        response_data = {
            "id": "chatcmpl-123",
            "created": 1677652288,
            "model": "gpt-4",
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
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        result = openai_model._normalize_response(response_data)

        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].message.tool_calls is None

    def test_normalize_response_with_content_and_tool_calls(self, openai_model):
        """Test response with both content and tool calls."""
        response_data = {
            "id": "chatcmpl-123",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

        result = openai_model._normalize_response(response_data)

        assert result.choices[0].message.content == "I'll check the weather for you."
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1