"""Tests for the Perplexity AI language model provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_openai import ChatOpenAI

from esperanto.providers.llm.perplexity import PerplexityLanguageModel
from esperanto.common_types import (
    ChatCompletion, Choice, Message, Usage,
    Tool, ToolFunction, ToolCall, ToolCallValidationError
)


@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for Perplexity API."""
    def create_response():
        return {
            "id": "cmpl-123",
            "object": "chat.completion",
            "created": 123,
            "model": "llama-3-sonar-large-32k-online",
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
def perplexity_provider(mock_httpx_response):
    """Fixture for PerplexityLanguageModel."""
    # Set dummy API key for testing
    os.environ["PERPLEXITY_API_KEY"] = "test_api_key"
    provider = PerplexityLanguageModel(model_name="llama-3-sonar-large-32k-online")
    
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
    
    provider.client = mock_client
    provider.async_client = mock_async_client
    
    # Clean up env var after test
    yield provider
    del os.environ["PERPLEXITY_API_KEY"]




def test_perplexity_provider_initialization(perplexity_provider):
    """Test initialization of PerplexityLanguageModel."""
    assert perplexity_provider.provider == "perplexity"
    assert (
        perplexity_provider.get_model_name() == "llama-3-sonar-large-32k-online"
    )  # Default model
    assert perplexity_provider.api_key == "test_api_key"
    assert perplexity_provider.base_url == "https://api.perplexity.ai"


def test_perplexity_provider_initialization_no_api_key():
    """Test initialization raises error if API key is missing."""
    if "PERPLEXITY_API_KEY" in os.environ:
        del os.environ["PERPLEXITY_API_KEY"]  # Ensure key is not set
    with pytest.raises(ValueError, match="Perplexity API key not found"):
        PerplexityLanguageModel(model_name="test-model")


def test_perplexity_get_api_kwargs(perplexity_provider):
    """Test _get_api_kwargs includes standard and perplexity-specific args."""
    perplexity_provider.temperature = 0.8
    perplexity_provider.max_tokens = 500
    perplexity_provider.search_domain_filter = ["example.com"]
    perplexity_provider.return_images = True
    perplexity_provider.web_search_options = {"search_context_size": "medium"}

    kwargs = perplexity_provider._get_api_kwargs()
    perplexity_params = perplexity_provider._get_perplexity_params()

    # Test standard kwargs
    assert kwargs["temperature"] == 0.8
    assert kwargs["max_tokens"] == 500
    # Ensure Perplexity params are NOT in standard kwargs
    assert "search_domain_filter" not in kwargs
    assert "return_images" not in kwargs
    assert "web_search_options" not in kwargs

    # Test perplexity params
    assert perplexity_params["search_domain_filter"] == ["example.com"]
    assert perplexity_params["return_images"] is True
    assert "return_related_questions" not in perplexity_params  # Not set
    assert "search_recency_filter" not in perplexity_params  # Not set
    assert perplexity_params["web_search_options"] == {"search_context_size": "medium"}


def test_perplexity_get_api_kwargs_exclude_stream(perplexity_provider):
    """Test _get_api_kwargs excludes stream when requested."""
    perplexity_provider.streaming = True
    kwargs = perplexity_provider._get_api_kwargs(exclude_stream=True)
    assert "stream" not in kwargs


@pytest.mark.asyncio
async def test_perplexity_async_call(perplexity_provider):
    """Test the asynchronous call method."""
    # Pass messages as dicts, not LangChain objects
    messages = [{"role": "user", "content": "Hello"}]
    expected_response_text = "Hello!"

    response = await perplexity_provider.achat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()


def test_perplexity_call(perplexity_provider):
    """Test the synchronous call method."""
    # Pass messages as dicts, not LangChain objects
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    expected_response_text = "Hello!"

    response = perplexity_provider.chat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()


def test_perplexity_call_with_extra_params(perplexity_provider):
    """Test synchronous call with extra Perplexity parameters."""
    perplexity_provider.search_domain_filter = ["test.com"]
    perplexity_provider.return_images = True
    messages = [{"role": "user", "content": "Hi"}]
    expected_response_text = "Hello!"

    response = perplexity_provider.chat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()
    
    # Test that perplexity params are available
    params = perplexity_provider._get_perplexity_params()
    assert params["search_domain_filter"] == ["test.com"]
    assert params["return_images"] is True


def test_perplexity_to_langchain(perplexity_provider):
    """Test conversion to LangChain model."""
    perplexity_provider.temperature = 0.7
    perplexity_provider.max_tokens = 100
    perplexity_provider.search_domain_filter = ["test.dev"]
    perplexity_provider.return_related_questions = True

    langchain_model = perplexity_provider.to_langchain()

    assert isinstance(langchain_model, ChatOpenAI)
    assert langchain_model.model_name == perplexity_provider.get_model_name()
    # Skip API key and base_url checks since they may be private attributes in LangChain
    assert langchain_model.temperature == 0.7
    assert langchain_model.max_tokens == 100
    assert langchain_model.model_kwargs["search_domain_filter"] == ["test.dev"]
    assert langchain_model.model_kwargs["return_related_questions"] is True
    assert "return_images" not in langchain_model.model_kwargs  # Not set


def test_perplexity_to_langchain_structured(perplexity_provider):
    """Test conversion to LangChain model with structured output."""
    perplexity_provider.structured = {"type": "json_object"}
    langchain_model = perplexity_provider.to_langchain()

    assert langchain_model.model_kwargs["response_format"] == {"type": "text"}


def test_perplexity_models_property(perplexity_provider):
    """Test the models property (currently hardcoded)."""
    models = perplexity_provider.models
    assert isinstance(models, list)
    assert len(models) > 5  # Check if it returns a reasonable number of models
    assert all(model.owned_by == "Perplexity" for model in models)
    # Check for some known models
    model_ids = [m.id for m in models]
    assert "sonar-pro" in model_ids
    assert "sonar" in model_ids


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
def mock_perplexity_tool_call_response():
    """Mock HTTP response for Perplexity chat completions with tool calls."""
    return {
        "id": "chatcmpl-tool-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "llama-3-sonar-large-32k-online",
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
def perplexity_model_with_tool_response(mock_perplexity_tool_call_response):
    """Create a Perplexity model with tool call response mocked."""
    os.environ["PERPLEXITY_API_KEY"] = "test_api_key"
    model = PerplexityLanguageModel(model_name="llama-3-sonar-large-32k-online")

    mock_client = Mock()
    mock_async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    mock_client.post = Mock(return_value=make_response(200, mock_perplexity_tool_call_response))
    mock_async_client.post = AsyncMock(return_value=make_response(200, mock_perplexity_tool_call_response))

    model.client = mock_client
    model.async_client = mock_async_client
    yield model
    del os.environ["PERPLEXITY_API_KEY"]


class TestToolConversion:
    """Tests for tool conversion to OpenAI format (Perplexity uses OpenAI-compatible format)."""

    def test_convert_single_tool(self, perplexity_provider, sample_tools):
        """Test converting a single tool to OpenAI format."""
        result = perplexity_provider._convert_tools_to_openai([sample_tools[0]])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the current weather for a location"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "location" in result[0]["function"]["parameters"]["properties"]

    def test_convert_multiple_tools(self, perplexity_provider, sample_tools):
        """Test converting multiple tools to OpenAI format."""
        result = perplexity_provider._convert_tools_to_openai(sample_tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "get_time"

    def test_convert_none_tools(self, perplexity_provider):
        """Test converting None returns None."""
        result = perplexity_provider._convert_tools_to_openai(None)
        assert result is None

    def test_convert_empty_tools(self, perplexity_provider):
        """Test converting empty list returns None."""
        result = perplexity_provider._convert_tools_to_openai([])
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, perplexity_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = perplexity_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = perplexity_model_with_tool_response.client.post.call_args
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

    def test_chat_complete_with_tool_choice(self, perplexity_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        perplexity_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = perplexity_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, perplexity_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await perplexity_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = perplexity_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, perplexity_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = perplexity_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, validate_tool_calls=True
        )

        assert response.choices[0].message.tool_calls is not None
