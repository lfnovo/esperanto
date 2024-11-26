"""Tests for Ollama LLM provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.providers.llm.ollama import OllamaLanguageModel
from esperanto.types import ChatCompletion, ChatCompletionChunk


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


def test_ollama_chat_complete(ollama_model, mock_ollama_response):
    """Test chat completion."""
    with patch.object(ollama_model.client, "chat") as mock_chat:
        mock_chat.return_value = mock_ollama_response
        messages = [{"role": "user", "content": "Hello"}]
        completion = ollama_model.chat_complete(messages)

        assert isinstance(completion, ChatCompletion)
        assert completion.content == "Test response"
        assert completion.model == "gemma2"
        assert completion.provider == "ollama"


def test_ollama_chat_complete_streaming(ollama_model, mock_ollama_stream_response):
    """Test streaming chat completion."""
    with patch.object(ollama_model.client, "chat") as mock_chat:
        mock_chat.return_value = iter(mock_ollama_stream_response)

        messages = [{"role": "user", "content": "Hello"}]
        stream = ollama_model.chat_complete(messages, stream=True)

        chunks = list(stream)
        assert len(chunks) > 0
        assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
        assert chunks[0].choices[0].delta.content == "Test"


@pytest.mark.asyncio
async def test_ollama_achat_complete(ollama_model, mock_ollama_response):
    """Test async chat completion."""
    with patch.object(ollama_model.async_client, "chat") as mock_chat:
        mock_chat.return_value = mock_ollama_response
        messages = [{"role": "user", "content": "Hello"}]
        completion = await ollama_model.achat_complete(messages)

        assert isinstance(completion, ChatCompletion)
        assert completion.content == "Test response"
        assert completion.model == "gemma2"
        assert completion.provider == "ollama"


@pytest.mark.asyncio
async def test_ollama_achat_complete_streaming(
    ollama_model, mock_ollama_stream_response
):
    """Test async streaming chat completion."""

    async def mock_async_generator():
        for chunk in mock_ollama_stream_response:
            yield chunk

    with patch.object(ollama_model.async_client, "chat") as mock_chat:
        mock_chat.return_value = mock_async_generator()

        messages = [{"role": "user", "content": "Hello"}]
        stream = await ollama_model.achat_complete(messages, stream=True)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
        assert chunks[0].choices[0].delta.content == "Test"


def test_ollama_to_langchain(ollama_model):
    """Test conversion to LangChain."""
    langchain_model = ollama_model.to_langchain()
    assert langchain_model is not None
    assert hasattr(langchain_model, "invoke")
    assert langchain_model.base_url == ollama_model.base_url
    assert langchain_model.model == "gemma2"


def test_ollama_chat_complete_with_system_message(ollama_model, mock_ollama_response):
    """Test chat completion with system message."""
    with patch.object(ollama_model.client, "chat") as mock_chat:
        mock_chat.return_value = mock_ollama_response
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        completion = ollama_model.chat_complete(messages)
        assert isinstance(completion, ChatCompletion)
        assert completion.content == "Test response"


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
