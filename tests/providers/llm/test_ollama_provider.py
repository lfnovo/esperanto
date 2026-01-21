"""Tests for Ollama LLM provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import (
    ChatCompletion, ChatCompletionChunk,
    Tool, ToolFunction, ToolCall, ToolCallValidationError
)
from esperanto.providers.llm.ollama import OllamaLanguageModel


@pytest.fixture
def mock_ollama_response():
    return {
        "model": "gemma2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": "Test response"},
        "done": True,
        "context": [],
        "total_duration": 100000000,
        "load_duration": 10000000,
        "prompt_eval_duration": 50000000,
        "eval_duration": 40000000,
        "eval_count": 10,
    }


@pytest.fixture
def mock_ollama_stream_response():
    return [
        {
            "model": "gemma2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Test"},
            "done": False,
        },
        {
            "model": "gemma2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": " response"},
            "done": True,
        },
    ]


@pytest.fixture
def ollama_model():
    """Create a test Ollama model with mocked clients."""
    with patch("ollama.Client") as mock_client:
        client_instance = Mock()
        mock_client.return_value = client_instance

        with patch("ollama.AsyncClient") as mock_async_client:
            async_client_instance = AsyncMock()
            mock_async_client.return_value = async_client_instance

            model = OllamaLanguageModel(model_name="gemma2")
            model.client = client_instance
            model.async_client = async_client_instance
            return model


def test_ollama_provider_name(ollama_model):
    """Test provider name."""
    assert ollama_model.provider == "ollama"


def test_ollama_default_model():
    """Test default model name."""
    model = OllamaLanguageModel()
    assert model._get_default_model() == "gemma2"


def test_ollama_initialization_with_base_url():
    """Test initialization with base URL."""
    model = OllamaLanguageModel(base_url="http://custom:11434")
    assert model.base_url == "http://custom:11434"


def test_ollama_initialization_with_env_var():
    """Test initialization with environment variable."""
    with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env:11434"}):
        model = OllamaLanguageModel()
        assert model.base_url == "http://env:11434"


def test_ollama_chat_complete():
    """Test chat completion with httpx mocking."""
    from unittest.mock import Mock
    from esperanto.providers.llm.ollama import OllamaLanguageModel
    
    # Create fresh model instance
    model = OllamaLanguageModel(model_name="gemma2")
    
    messages = [{"role": "user", "content": "Hello"}]

    # Mock Ollama API response data
    mock_response_data = {
        "model": "gemma2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Test response"
        },
        "done": True,
        "total_duration": 1000000000,
        "load_duration": 500000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 100000000,
        "eval_count": 5,
        "eval_duration": 200000000
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    completion = model.chat_complete(messages)

    assert completion.choices[0].message.content == "Test response"
    assert completion.model == "gemma2"
    assert completion.provider == "ollama"


def test_ollama_chat_complete_streaming():
    """Test streaming chat completion with httpx mocking."""
    from unittest.mock import Mock
    from esperanto.providers.llm.ollama import OllamaLanguageModel
    
    # Create fresh model instance
    model = OllamaLanguageModel(model_name="gemma2")
    
    messages = [{"role": "user", "content": "Hello"}]

    # Mock Ollama streaming response - multiple JSONL responses
    stream_data = [
        '{"model":"gemma2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Test"},"done":false}\n',
        '{"model":"gemma2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":" response"},"done":false}\n',
        '{"model":"gemma2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true}\n'
    ]
    
    # Mock HTTP response for streaming
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = stream_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    stream = model.chat_complete(messages, stream=True)
    chunks = list(stream)
    
    assert len(chunks) > 0
    assert chunks[0].choices[0].delta.content == "Test"


@pytest.mark.asyncio
async def test_ollama_achat_complete():
    """Test async chat completion with httpx mocking."""
    from unittest.mock import Mock, AsyncMock
    from esperanto.providers.llm.ollama import OllamaLanguageModel
    
    # Create fresh model instance
    model = OllamaLanguageModel(model_name="gemma2")
    
    messages = [{"role": "user", "content": "Hello"}]

    # Mock Ollama API response data
    mock_response_data = {
        "model": "gemma2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Test response"
        },
        "done": True
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the async client
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    model.async_client = mock_async_client

    completion = await model.achat_complete(messages)

    assert completion.choices[0].message.content == "Test response"
    assert completion.model == "gemma2"
    assert completion.provider == "ollama"


@pytest.mark.asyncio
async def test_ollama_achat_complete_streaming():
    """Test async streaming chat completion with httpx mocking."""
    from unittest.mock import Mock, AsyncMock
    from esperanto.providers.llm.ollama import OllamaLanguageModel
    
    # Create fresh model instance
    model = OllamaLanguageModel(model_name="gemma2")
    
    messages = [{"role": "user", "content": "Hello"}]

    # Mock Ollama streaming response - multiple JSONL responses
    async def mock_aiter_lines():
        yield '{"model":"gemma2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Test"},"done":false}\n'
        yield '{"model":"gemma2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":" response"},"done":false}\n'
        yield '{"model":"gemma2","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true}\n'
    
    # Mock HTTP response for streaming
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines
    
    # Mock the async client
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    model.async_client = mock_async_client

    stream = await model.achat_complete(messages, stream=True)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[0].choices[0].delta.content == "Test"


def test_ollama_to_langchain(ollama_model):
    """Test conversion to LangChain."""
    langchain_model = ollama_model.to_langchain()
    assert langchain_model is not None
    assert hasattr(langchain_model, "invoke")
    assert langchain_model.base_url == ollama_model.base_url
    assert langchain_model.model == "gemma2"


def test_ollama_to_langchain_with_json_format():
    """Test that to_langchain passes format parameter when structured={"type": "json"}."""
    model = OllamaLanguageModel(
        model_name="gemma2",
        structured={"type": "json"}
    )
    langchain_model = model.to_langchain()
    assert langchain_model.format == "json"


def test_ollama_to_langchain_with_json_object_format():
    """Test that to_langchain passes format parameter when structured={"type": "json_object"}."""
    model = OllamaLanguageModel(
        model_name="gemma2",
        structured={"type": "json_object"}
    )
    langchain_model = model.to_langchain()
    assert langchain_model.format == "json"


def test_ollama_to_langchain_without_structured():
    """Test that to_langchain does not set format when structured is not set."""
    model = OllamaLanguageModel(model_name="gemma2")
    langchain_model = model.to_langchain()
    assert langchain_model.format is None


def test_ollama_chat_complete_with_system_message():
    """Test chat completion with system message using httpx mocking."""
    from unittest.mock import Mock
    from esperanto.providers.llm.ollama import OllamaLanguageModel
    
    # Create fresh model instance
    model = OllamaLanguageModel(model_name="gemma2")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]

    # Mock Ollama API response data
    mock_response_data = {
        "model": "gemma2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Test response"
        },
        "done": True,
        "context": [],
        "total_duration": 100000000,
        "load_duration": 10000000,
        "prompt_eval_duration": 50000000,
        "eval_duration": 40000000,
        "eval_count": 10,
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    completion = model.chat_complete(messages)
    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "Test response"


def test_ollama_chat_complete_with_invalid_messages():
    """Test chat completion with invalid messages."""
    model = OllamaLanguageModel()
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        model.chat_complete([])
    with pytest.raises(ValueError, match="Invalid role"):
        model.chat_complete([{"role": "invalid", "content": "test"}])
    with pytest.raises(ValueError, match="Missing content"):
        model.chat_complete([{"role": "user"}])


def test_ollama_model_parameters():
    """Test model parameters are correctly set."""
    model = OllamaLanguageModel(
        model_name="gemma2", temperature=0.7, top_p=0.9, max_tokens=100, streaming=True
    )
    assert model.model_name == "gemma2"
    assert model.temperature == 0.7
    assert model.top_p == 0.9
    assert model.max_tokens == 100
    assert model.streaming is True


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
def mock_ollama_tool_call_response():
    """Mock HTTP response for Ollama chat completions with tool calls."""
    return {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "San Francisco", "unit": "celsius"}
                    }
                }
            ]
        },
        "done": True,
        "done_reason": "tool_call"
    }


@pytest.fixture
def ollama_model_with_tool_response(mock_ollama_tool_call_response):
    """Create an Ollama model with tool call response mocked."""
    model = OllamaLanguageModel(model_name="llama3.2")

    mock_client = Mock()
    mock_async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    mock_client.post.return_value = make_response(200, mock_ollama_tool_call_response)
    mock_async_client.post.return_value = make_response(200, mock_ollama_tool_call_response)

    model.client = mock_client
    model.async_client = mock_async_client
    return model


class TestToolConversion:
    """Tests for tool conversion to Ollama format."""

    def test_convert_single_tool(self, sample_tools):
        """Test converting a single tool to Ollama format."""
        model = OllamaLanguageModel(model_name="llama3.2")
        result = model._convert_tools_to_ollama([sample_tools[0]])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the current weather for a location"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "location" in result[0]["function"]["parameters"]["properties"]

    def test_convert_multiple_tools(self, sample_tools):
        """Test converting multiple tools to Ollama format."""
        model = OllamaLanguageModel(model_name="llama3.2")
        result = model._convert_tools_to_ollama(sample_tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "get_time"

    def test_convert_none_tools(self):
        """Test converting None returns None."""
        model = OllamaLanguageModel(model_name="llama3.2")
        result = model._convert_tools_to_ollama(None)
        assert result is None

    def test_convert_empty_tools(self):
        """Test converting empty list returns None."""
        model = OllamaLanguageModel(model_name="llama3.2")
        result = model._convert_tools_to_ollama([])
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, ollama_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = ollama_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = ollama_model_with_tool_response.client.post.call_args
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

    def test_chat_complete_with_tool_choice(self, ollama_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        ollama_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = ollama_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        # Ollama may or may not support tool_choice - just verify it's passed
        assert "tools" in json_payload

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, ollama_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await ollama_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = ollama_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, ollama_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = ollama_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, validate_tool_calls=True
        )

        assert response.choices[0].message.tool_calls is not None
