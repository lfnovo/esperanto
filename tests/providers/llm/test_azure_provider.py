"""Tests for the Azure OpenAI language model provider."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Tool,
    ToolFunction,
    ToolCall,
    ToolCallValidationError,
)
from esperanto.providers.llm.azure import AzureLanguageModel

# --- Fixtures ---

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock necessary environment variables for Azure."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_azure_api_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-endpoint.openai.azure.com/")
    # AZURE_OPENAI_DEPLOYMENT_NAME is passed as model_name

@pytest.fixture
def azure_model(mock_env_vars):
    """Return an AzureLanguageModel instance with mocked httpx clients."""
    # Mock httpx response for non-streaming
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "resp-test123",
        "object": "response",
        "created_at": 1677652288,
        "status": "completed",
        "model": "gpt-35-turbo",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from Azure!"
                    }
                ]
            }
        ],
        "usage": {"output_tokens": 5, "input_tokens": 10, "total_tokens": 15}
    }

    # Mock the httpx clients
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.stream = MagicMock()  # For streaming tests
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.stream = MagicMock()  # For streaming tests

    # Create model and replace clients before any HTTP calls
    with patch('httpx.Client', return_value=mock_client), \
         patch('httpx.AsyncClient', return_value=mock_async_client):
        model = AzureLanguageModel(model_name="test-deployment")

    return model

# --- Test Cases ---

def test_provider_name(azure_model):
    assert azure_model.provider == "azure"

def test_initialization_success(mock_env_vars):
    """Test successful initialization with environment variables."""
    model = AzureLanguageModel(model_name="test-deployment")
    assert model.api_key == "test_azure_api_key"
    assert model.azure_endpoint == "https://test-endpoint.openai.azure.com/"
    assert model.model_name == "test-deployment" # Deployment name
    assert model.client is not None
    assert model.async_client is not None

def test_initialization_with_direct_params():
    """Test successful initialization with direct parameters."""
    model = AzureLanguageModel(
        model_name="direct-deployment",
        api_key="direct_key",
        config={
            "azure_endpoint": "https://direct-endpoint.com/",
        }
    )
    assert model.api_key == "direct_key"
    assert model.azure_endpoint == "https://direct-endpoint.com/"
    assert model.model_name == "direct-deployment"

@pytest.mark.parametrize(
    "missing_var, error_msg_part",
    [
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI API key not found"),
        ("AZURE_OPENAI_ENDPOINT", "Azure OpenAI endpoint not found"),
    ],
)
def test_initialization_missing_env_vars(monkeypatch, missing_var, error_msg_part):
    """Test initialization failure when an environment variable is missing."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-endpoint.com/")
    
    monkeypatch.delenv(missing_var, raising=False)
    
    with pytest.raises(ValueError, match=error_msg_part):
        AzureLanguageModel(model_name="test-deployment")

def test_initialization_missing_model_name(mock_env_vars):
    """Test initialization failure if model_name (deployment_name) is missing."""
    with pytest.raises(ValueError, match="Azure OpenAI deployment name \(model_name\) not found"):
        AzureLanguageModel(model_name=None)

def test_models_property(azure_model):
    """Test the 'models' property."""
    # Currently returns empty list or a placeholder for the configured deployment
    # For now, let's assume it's an empty list as per implementation
    assert azure_model.models == [] 
    # If implementation changes to return current deployment:
    # assert azure_model.models == [Model(id="test-deployment", owned_by="azure", type="language")]

def test_chat_complete_non_streaming(azure_model):
    messages = [{"role": "user", "content": "Hello Azure!"}]
    # Explicitly pass stream=False to avoid returning a generator
    response = azure_model.chat_complete(messages, stream=False)

    azure_model.client.post.assert_called_once()
    call_args = azure_model.client.post.call_args

    # Check URL was built correctly (Responses API)
    assert call_args[0][0].endswith("/openai/v1/responses")

    # Check request payload uses Responses API format
    _, kwargs = call_args
    assert kwargs['json']['input'] == messages
    assert kwargs['json']['stream'] == False

    assert isinstance(response, ChatCompletion)
    assert response.id == "resp-test123"
    assert response.choices[0].message.content == "Hello from Azure!"
    assert response.model == "gpt-35-turbo" # Underlying model from response
    assert response.provider == "azure"
    assert response.usage.total_tokens == 15

@pytest.mark.asyncio
async def test_achat_complete_non_streaming(azure_model):
    messages = [{"role": "user", "content": "Hello async Azure!"}]
    response = await azure_model.achat_complete(messages)

    azure_model.async_client.post.assert_called_once()
    call_args = azure_model.async_client.post.call_args

    # Check URL was built correctly (Responses API)
    assert call_args[0][0].endswith("/openai/v1/responses")

    # Check request payload uses Responses API format
    _, kwargs = call_args
    assert kwargs['json']['input'] == messages
    assert kwargs['json']['stream'] == False

    assert isinstance(response, ChatCompletion)
    assert response.id == "resp-test123"
    assert response.choices[0].message.content == "Hello from Azure!"
    assert response.model == "gpt-35-turbo"
    assert response.provider == "azure"
    assert response.usage.total_tokens == 15


def test_chat_complete_streaming(azure_model):
    messages = [{"role": "user", "content": "Stream Azure hello!"}]

    # Mock streaming response using Responses API SSE format
    mock_stream_response = MagicMock()
    mock_stream_response.status_code = 200
    mock_stream_response.iter_text.return_value = [
        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp-stream-test","model":"gpt-35-turbo","created_at":1677652290}}\n\n',
        'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hello ","item_id":"msg_123","output_index":0,"content_index":0}\n\n',
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp-stream-test","model":"gpt-35-turbo","created_at":1677652290,"status":"completed"}}\n\n',
    ]

    # Mock the stream context manager
    mock_stream_context = MagicMock()
    mock_stream_context.__enter__.return_value = mock_stream_response
    mock_stream_context.__exit__.return_value = None
    azure_model.client.stream.return_value = mock_stream_context

    response_gen = azure_model.chat_complete(messages, stream=True)
    responses = list(response_gen)

    azure_model.client.stream.assert_called_once()
    call_args = azure_model.client.stream.call_args

    # Check it was called with POST and correct URL (Responses API)
    assert call_args[0][0] == "POST"
    assert call_args[0][1].endswith("/openai/v1/responses")

    # Should have text delta + completed chunks
    assert len(responses) == 2
    chunk = responses[0]
    assert isinstance(chunk, ChatCompletionChunk)
    assert chunk.id == "resp-stream-test"
    assert chunk.choices[0].delta.content == "Hello "
    assert chunk.model == "gpt-35-turbo"

@pytest.mark.asyncio
async def test_achat_complete_streaming(azure_model):
    messages = [{"role": "user", "content": "Stream async Azure hello!"}]

    # Mock async streaming response using Responses API SSE format
    mock_stream_response = MagicMock()
    mock_stream_response.status_code = 200

    async def mock_aiter_text():
        yield 'event: response.created\ndata: {"type":"response.created","response":{"id":"resp-asyncstream-test","model":"gpt-35-turbo","created_at":1677652291}}\n\n'
        yield 'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Async Hello ","item_id":"msg_123","output_index":0,"content_index":0}\n\n'
        yield 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp-asyncstream-test","model":"gpt-35-turbo","created_at":1677652291,"status":"completed"}}\n\n'

    mock_stream_response.aiter_text.return_value = mock_aiter_text()

    # Mock the async stream context manager
    mock_stream_context = MagicMock()
    mock_stream_context.__aenter__ = AsyncMock(return_value=mock_stream_response)
    mock_stream_context.__aexit__ = AsyncMock(return_value=None)
    azure_model.async_client.stream.return_value = mock_stream_context

    response_gen = await azure_model.achat_complete(messages, stream=True)
    responses = [chunk async for chunk in response_gen]

    azure_model.async_client.stream.assert_called_once()
    call_args = azure_model.async_client.stream.call_args

    # Check it was called with POST and correct URL (Responses API)
    assert call_args[0][0] == "POST"
    assert call_args[0][1].endswith("/openai/v1/responses")

    # Should have text delta + completed chunks
    assert len(responses) == 2
    chunk = responses[0]
    assert isinstance(chunk, ChatCompletionChunk)
    assert chunk.id == "resp-asyncstream-test"
    assert chunk.choices[0].delta.content == "Async Hello "
    assert chunk.model == "gpt-35-turbo"

def test_get_api_kwargs(azure_model):
    azure_model.temperature = 0.7
    azure_model.max_tokens = 100
    kwargs = azure_model._get_api_kwargs()
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_output_tokens"] == 100
    assert kwargs.get("model") == "test-deployment" # model_name is used
    assert kwargs["store"] == False

def test_get_api_kwargs_streaming(azure_model):
    azure_model.streaming = True
    kwargs = azure_model._get_api_kwargs()
    assert kwargs["stream"]

def test_get_api_kwargs_json_mode(azure_model):
    azure_model.structured = {"type": "json_object"}
    kwargs = azure_model._get_api_kwargs()
    assert kwargs["text"] == {"format": {"type": "json_object"}}

    azure_model.structured = {"type": "json"} # Alias
    kwargs = azure_model._get_api_kwargs()
    assert kwargs["text"] == {"format": {"type": "json_object"}}

    with pytest.raises(TypeError):
        azure_model.structured = "not_a_dict"
        azure_model._get_api_kwargs()


@patch("langchain_openai.AzureChatOpenAI")
def test_to_langchain(MockAzureChatOpenAI, azure_model, mock_env_vars):
    azure_model.temperature = 0.8
    azure_model.max_tokens = 150
    # model_name is the deployment name
    lc_model = azure_model.to_langchain(another_param="test_val")

    MockAzureChatOpenAI.assert_called_once()
    call_kwargs = MockAzureChatOpenAI.call_args[1]

    assert call_kwargs["azure_deployment"] == "test-deployment"
    assert call_kwargs["api_key"].get_secret_value() == "test_azure_api_key"
    assert call_kwargs["azure_endpoint"] == "https://test-endpoint.openai.azure.com/"
    assert call_kwargs["api_version"] == "2025-04-01-preview"
    assert call_kwargs["use_responses_api"] == True
    assert call_kwargs["temperature"] == 0.8
    assert call_kwargs["max_tokens"] == 150
    assert call_kwargs["another_param"] == "test_val" # Kwargs passed directly

@patch("langchain_openai.AzureChatOpenAI")
def test_to_langchain_json_mode(MockAzureChatOpenAI, azure_model, mock_env_vars):
    azure_model.structured = {"type": "json"}
    lc_model = azure_model.to_langchain()

    MockAzureChatOpenAI.assert_called_once()
    call_kwargs = MockAzureChatOpenAI.call_args[1]
    assert call_kwargs["model_kwargs"] == {"response_format": {"type": "json_object"}}

@patch.dict(os.environ, {}, clear=True)
@patch.dict(sys.modules, {"langchain_openai": None})
def test_to_langchain_import_error(azure_model):
    # Simulate langchain_openai not being installed by patching sys.modules

    # Ensure necessary attributes are set on azure_model if __post_init_post_parse__ relies on them
    # However, the import error should be raised before these are deeply checked by LangChain
    azure_model.api_key = "temp_key"
    azure_model.azure_endpoint = "temp_endpoint"

    with pytest.raises(ImportError, match="LangChain or langchain-openai not installed"):
        azure_model.to_langchain()


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
def mock_azure_tool_call_response():
    """Mock HTTP response for Azure OpenAI Responses API with tool calls."""
    return {
        "id": "resp-tool-123",
        "object": "response",
        "created_at": 1677652288,
        "status": "completed",
        "model": "gpt-35-turbo",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc123",
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}'
            }
        ],
        "usage": {
            "input_tokens": 50,
            "output_tokens": 20,
            "total_tokens": 70
        }
    }


@pytest.fixture
def azure_model_with_tool_response(mock_env_vars, mock_azure_tool_call_response):
    """Create an Azure model with tool call response mocked."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_azure_tool_call_response

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch('httpx.Client', return_value=mock_client), \
         patch('httpx.AsyncClient', return_value=mock_async_client):
        model = AzureLanguageModel(model_name="test-deployment")

    return model


class TestToolConversion:
    """Tests for tool conversion to Responses API format."""

    def test_convert_single_tool(self, azure_model, sample_tools):
        """Test converting a single tool to Responses API format."""
        result = azure_model._convert_tools_to_responses([sample_tools[0]])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get the current weather for a location"
        assert result[0]["parameters"]["type"] == "object"
        assert "location" in result[0]["parameters"]["properties"]

    def test_convert_multiple_tools(self, azure_model, sample_tools):
        """Test converting multiple tools to Responses API format."""
        result = azure_model._convert_tools_to_responses(sample_tools)

        assert len(result) == 2
        assert result[0]["name"] == "get_weather"
        assert result[1]["name"] == "get_time"

    def test_convert_none_tools(self, azure_model):
        """Test converting None returns None."""
        result = azure_model._convert_tools_to_responses(None)
        assert result is None

    def test_convert_empty_tools(self, azure_model):
        """Test converting empty list returns None."""
        result = azure_model._convert_tools_to_responses([])
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, azure_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = azure_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, stream=False
        )

        # Check payload included tools
        call_args = azure_model_with_tool_response.client.post.call_args
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

    def test_chat_complete_with_tool_choice(self, azure_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        azure_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required", stream=False
        )

        call_args = azure_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, azure_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await azure_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = azure_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, azure_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        pytest.importorskip("jsonschema")

        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        # Should not raise
        response = azure_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, validate_tool_calls=True, stream=False
        )

        assert response.choices[0].message.tool_calls is not None
