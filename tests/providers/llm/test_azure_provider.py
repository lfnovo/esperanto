"""Tests for the Azure OpenAI language model provider."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Assuming openai.types.chat objects are used for mocking responses
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from openai.types.chat import ChatCompletionChunk as OpenAIChatCompletionChunk
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
)
from esperanto.providers.llm.azure import AzureLanguageModel

# --- Fixtures ---

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock necessary environment variables for Azure."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_azure_api_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-endpoint.openai.azure.com/")
    monkeypatch.setenv("OPENAI_API_VERSION", "2023-12-01-preview")
    # AZURE_OPENAI_DEPLOYMENT_NAME is passed as model_name

@pytest.fixture
def azure_model(mock_env_vars):
    """Return an AzureLanguageModel instance with mocked clients."""
    with patch("openai.AzureOpenAI") as MockAzureOpenAI, \
         patch("openai.AsyncAzureOpenAI") as MockAsyncAzureOpenAI:
        
        # Configure the mock clients
        mock_client = MockAzureOpenAI.return_value
        mock_async_client = MockAsyncAzureOpenAI.return_value

        # Mock the .chat.completions.create methods
        mock_client.chat.completions.create = MagicMock(
            return_value=OpenAIChatCompletion(
                id="chatcmpl-test123",
                choices=[
                    OpenAIChoice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            content="Hello from Azure!", role="assistant"
                        ),
                    )
                ],
                created=1677652288,
                model="gpt-35-turbo", # This is the underlying model, not deployment
                object="chat.completion",
                usage=CompletionUsage(completion_tokens=5, prompt_tokens=10, total_tokens=15),
            )
        )
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=OpenAIChatCompletion(
                id="chatcmpl-testasync123",
                choices=[
                    OpenAIChoice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            content="Hello from async Azure!", role="assistant"
                        ),
                    )
                ],
                created=1677652289,
                model="gpt-35-turbo",
                object="chat.completion",
                usage=CompletionUsage(completion_tokens=6, prompt_tokens=11, total_tokens=17),
            )
        )
        
        model = AzureLanguageModel(model_name="test-deployment") # model_name is deployment name
        model.client = mock_client
        model.async_client = mock_async_client
        return model

# --- Test Cases ---

def test_provider_name(azure_model):
    assert azure_model.provider == "azure"

def test_initialization_success(mock_env_vars):
    """Test successful initialization with environment variables."""
    model = AzureLanguageModel(model_name="test-deployment")
    assert model.api_key == "test_azure_api_key"
    assert model.azure_endpoint == "https://test-endpoint.openai.azure.com/"
    assert model.api_version == "2023-12-01-preview"
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
            "api_version": "2024-01-01",
        }
    )
    assert model.api_key == "direct_key"
    assert model.azure_endpoint == "https://direct-endpoint.com/"
    assert model.api_version == "2024-01-01"
    assert model.model_name == "direct-deployment"

@pytest.mark.parametrize(
    "missing_var, error_msg_part",
    [
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI API key not found"),
        ("AZURE_OPENAI_ENDPOINT", "Azure OpenAI endpoint not found"),
        ("OPENAI_API_VERSION", "Azure OpenAI API version not found"),
    ],
)
def test_initialization_missing_env_vars(monkeypatch, missing_var, error_msg_part):
    """Test initialization failure when an environment variable is missing."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-endpoint.com/")
    monkeypatch.setenv("OPENAI_API_VERSION", "2023-01-01")
    
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
    response = azure_model.chat_complete(messages)

    azure_model.client.chat.completions.create.assert_called_once()
    call_args = azure_model.client.chat.completions.create.call_args[1]
    
    assert call_args["model"] == "test-deployment" # Should be deployment name
    assert call_args["messages"] == messages
    assert not call_args.get("stream", False)

    assert isinstance(response, ChatCompletion)
    assert response.id == "chatcmpl-test123"
    assert response.choices[0].message.content == "Hello from Azure!"
    assert response.model == "gpt-35-turbo" # Underlying model from response
    assert response.provider == "azure"
    assert response.usage.total_tokens == 15

@pytest.mark.asyncio
async def test_achat_complete_non_streaming(azure_model):
    messages = [{"role": "user", "content": "Hello async Azure!"}]
    response = await azure_model.achat_complete(messages)

    azure_model.async_client.chat.completions.create.assert_called_once()
    call_args = azure_model.async_client.chat.completions.create.call_args[1]

    assert call_args["model"] == "test-deployment"
    assert call_args["messages"] == messages
    assert not call_args.get("stream", False)

    assert isinstance(response, ChatCompletion)
    assert response.id == "chatcmpl-testasync123"
    assert response.choices[0].message.content == "Hello from async Azure!"
    assert response.model == "gpt-35-turbo"
    assert response.provider == "azure"
    assert response.usage.total_tokens == 17


def test_chat_complete_streaming(azure_model):
    messages = [{"role": "user", "content": "Stream Azure hello!"}]
    
    # Mock streaming response
    mock_stream_chunk = OpenAIChatCompletionChunk(
        id="chatcmpl-stream-test",
        choices=[
            OpenAIChunkChoice(
                delta=ChoiceDelta(content="Hello ", role="assistant"), 
                index=0, 
                finish_reason=None
            )
        ],
        created=1677652290,
        model="gpt-35-turbo",
        object="chat.completion.chunk",
    )
    azure_model.client.chat.completions.create.return_value = [mock_stream_chunk]

    response_gen = azure_model.chat_complete(messages, stream=True)
    responses = list(response_gen)

    azure_model.client.chat.completions.create.assert_called_once()
    call_args = azure_model.client.chat.completions.create.call_args[1]
    assert call_args["stream"]

    assert len(responses) == 1
    chunk = responses[0]
    assert isinstance(chunk, ChatCompletionChunk)
    assert chunk.id == "chatcmpl-stream-test"
    assert chunk.choices[0].delta.content == "Hello "
    assert chunk.model == "gpt-35-turbo"

@pytest.mark.asyncio
async def test_achat_complete_streaming(azure_model):
    messages = [{"role": "user", "content": "Stream async Azure hello!"}]

    mock_stream_chunk = OpenAIChatCompletionChunk(
        id="chatcmpl-asyncstream-test",
        choices=[
            OpenAIChunkChoice(
                delta=ChoiceDelta(content="Async Hello ", role="assistant"), 
                index=0, 
                finish_reason=None
            )
        ],
        created=1677652291,
        model="gpt-35-turbo",
        object="chat.completion.chunk",
    )
    
    # Mock async generator for streaming
    async def mock_async_gen():
        yield mock_stream_chunk

    azure_model.async_client.chat.completions.create.return_value = mock_async_gen()

    response_gen = await azure_model.achat_complete(messages, stream=True)
    responses = [chunk async for chunk in response_gen]

    azure_model.async_client.chat.completions.create.assert_called_once()
    call_args = azure_model.async_client.chat.completions.create.call_args[1]
    assert call_args["stream"]

    assert len(responses) == 1
    chunk = responses[0]
    assert isinstance(chunk, ChatCompletionChunk)
    assert chunk.id == "chatcmpl-asyncstream-test"
    assert chunk.choices[0].delta.content == "Async Hello "
    assert chunk.model == "gpt-35-turbo"

def test_get_api_kwargs(azure_model):
    azure_model.temperature = 0.7
    azure_model.max_tokens = 100
    kwargs = azure_model._get_api_kwargs()
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100
    assert kwargs.get("model") == "test-deployment" # model_name is used

def test_get_api_kwargs_streaming(azure_model):
    azure_model.streaming = True
    kwargs = azure_model._get_api_kwargs()
    assert kwargs["stream"]

def test_get_api_kwargs_json_mode(azure_model):
    azure_model.structured = {"type": "json_object"}
    kwargs = azure_model._get_api_kwargs()
    assert kwargs["response_format"] == {"type": "json_object"}

    azure_model.structured = {"type": "json"} # Alias
    kwargs = azure_model._get_api_kwargs()
    assert kwargs["response_format"] == {"type": "json_object"}

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
    assert call_kwargs["api_version"] == "2023-12-01-preview"
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
    azure_model.api_version = "temp_version"

    with pytest.raises(ImportError, match="LangChain or langchain-openai not installed"):
        azure_model.to_langchain()
