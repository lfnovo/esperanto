"""Tests for Ollama LLM provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import ChatCompletion, ChatCompletionChunk
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


def test_ollama_default_num_ctx():
    """Test that default num_ctx (128000) is applied when not specified."""
    model = OllamaLanguageModel(model_name="gemma2")
    api_kwargs = model._get_api_kwargs()

    assert "options" in api_kwargs
    assert api_kwargs["options"]["num_ctx"] == 128000


def test_ollama_custom_num_ctx():
    """Test that custom num_ctx is passed correctly via config."""
    model = OllamaLanguageModel(model_name="gemma2", config={"num_ctx": 32768})
    api_kwargs = model._get_api_kwargs()

    assert "options" in api_kwargs
    assert api_kwargs["options"]["num_ctx"] == 32768


def test_ollama_to_langchain_default_num_ctx():
    """Test that to_langchain uses default num_ctx (128000) when not specified."""
    model = OllamaLanguageModel(model_name="gemma2")
    langchain_model = model.to_langchain()

    assert langchain_model.num_ctx == 128000


def test_ollama_to_langchain_custom_num_ctx():
    """Test that to_langchain passes custom num_ctx from config."""
    model = OllamaLanguageModel(model_name="gemma2", config={"num_ctx": 65536})
    langchain_model = model.to_langchain()

    assert langchain_model.num_ctx == 65536


def test_ollama_keep_alive_not_set_by_default():
    """Test that keep_alive is not set when not specified (no forced memory usage)."""
    model = OllamaLanguageModel(model_name="gemma2")
    api_kwargs = model._get_api_kwargs()

    assert "keep_alive" not in api_kwargs


def test_ollama_keep_alive_custom():
    """Test that custom keep_alive is passed correctly via config."""
    model = OllamaLanguageModel(model_name="gemma2", config={"keep_alive": "10m"})
    api_kwargs = model._get_api_kwargs()

    assert api_kwargs["keep_alive"] == "10m"


def test_ollama_keep_alive_zero():
    """Test that keep_alive can be set to 0 to unload immediately."""
    model = OllamaLanguageModel(model_name="gemma2", config={"keep_alive": "0"})
    api_kwargs = model._get_api_kwargs()

    assert api_kwargs["keep_alive"] == "0"


def test_ollama_to_langchain_no_keep_alive_by_default():
    """Test that to_langchain does not set keep_alive when not specified."""
    model = OllamaLanguageModel(model_name="gemma2")
    langchain_model = model.to_langchain()

    # LangChain's ChatOllama should not have keep_alive set
    assert langchain_model.keep_alive is None


def test_ollama_to_langchain_custom_keep_alive():
    """Test that to_langchain passes custom keep_alive from config."""
    model = OllamaLanguageModel(model_name="gemma2", config={"keep_alive": "30m"})
    langchain_model = model.to_langchain()

    assert langchain_model.keep_alive == "30m"
