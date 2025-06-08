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
    assert models[0].type == "language"
    assert models[1].type == "language"


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