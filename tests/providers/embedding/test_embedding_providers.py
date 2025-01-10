"""Tests for embedding providers."""

import os
from typing import List
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.embedding.google import GoogleEmbeddingModel
from esperanto.providers.embedding.ollama import OllamaEmbeddingModel
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
from esperanto.providers.embedding.vertex import VertexEmbeddingModel


# Mock responses
@pytest.fixture
def mock_openai_embedding_response():
    class EmbeddingData:
        def __init__(self):
            self.embedding = [0.1, 0.2, 0.3]
            self.index = 0

    class Response:
        def __init__(self):
            self.data = [EmbeddingData()]
            self.model = "text-embedding-3-small"
            self.usage = {"total_tokens": 4}

    return Response()


@pytest.fixture
def mock_ollama_embedding_response():
    class EmbeddingsResponse:
        def __init__(self):
            self.embedding = [0.1, 0.2, 0.3]

    return EmbeddingsResponse()


@pytest.fixture
def mock_google_embedding_response():
    return {"embedding": [0.1, 0.2, 0.3]}


@pytest.fixture
def mock_vertex_embedding_response():
    class EmbeddingValue:
        def __init__(self):
            self.values = [0.1, 0.2, 0.3]

    return [EmbeddingValue()]


# Mock clients
@pytest.fixture
def mock_openai_embedding_client(mock_openai_embedding_response):
    client = Mock()
    async_client = AsyncMock()

    # Mock synchronous embeddings
    client.embeddings.create.return_value = mock_openai_embedding_response

    # Mock async embeddings
    async_client.embeddings.create.return_value = mock_openai_embedding_response

    return client, async_client


@pytest.fixture
def mock_ollama_response(mock_ollama_embedding_response):
    mock_sync = Mock()
    mock_sync.embeddings.return_value = mock_ollama_embedding_response

    mock_async = AsyncMock()
    mock_async.embeddings.return_value = mock_ollama_embedding_response

    return mock_sync, mock_async


@pytest.fixture
def mock_vertex_model(mock_vertex_embedding_response):
    mock = Mock()
    mock.get_embeddings.return_value = mock_vertex_embedding_response
    return mock


# Provider fixtures
@pytest.fixture
def openai_embedding_model(mock_openai_embedding_client):
    model = OpenAIEmbeddingModel(
        api_key="test-key", model_name="text-embedding-3-small"
    )
    model.client, model.async_client = mock_openai_embedding_client
    return model


@pytest.fixture
def ollama_embedding_model(mock_ollama_response):
    model = OllamaEmbeddingModel(
        base_url="http://localhost:11434", model_name="mxbai-embed-large"
    )
    model.client = mock_ollama_response[0]
    model.async_client = mock_ollama_response[1]
    return model



@pytest.fixture
def vertex_embedding_model(mock_vertex_model):
    model = VertexEmbeddingModel(
        vertex_project="test-project", model_name="textembedding-gecko"
    )
    model._model = mock_vertex_model
    return model


# Test base embedding model configuration
class TestEmbeddingModel(EmbeddingModel):
    """Test implementation of EmbeddingModel."""

    @property
    def models(self):
        """List all available models for this provider."""
        return []

    @property
    def provider(self):
        """Get the provider name."""
        return "test"

    def _get_default_model(self):
        """Get the default model name."""
        return "test-default-model"

    def embed(self, texts: List[str], **kwargs):
        """Embed texts."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def aembed(self, texts: List[str], **kwargs):
        """Async embed texts."""
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_embedding_model_config():
    """Test embedding model configuration initialization."""
    config = {"model_name": "test-model", "api_key": "test-key", "base_url": "test-url"}
    model = TestEmbeddingModel(config=config)
    assert model.model_name == "test-model"
    assert model.api_key == "test-key"
    assert model.base_url == "test-url"


def test_embedding_model_get_model_name():
    """Test get_model_name with config and default."""
    # Test with model name in config
    model = TestEmbeddingModel(model_name="test-model")
    assert model.get_model_name() == "test-model"

    # Test fallback to default model
    model = TestEmbeddingModel()
    assert model.get_model_name() == "test-default-model"


def test_embedding_model_provider():
    """Test provider property."""
    model = TestEmbeddingModel()
    assert model.provider == "test"


# Tests for OpenAI Embedding Provider
def test_openai_provider_name(openai_embedding_model):
    assert openai_embedding_model.provider == "openai"


def test_openai_initialization_with_api_key():
    model = OpenAIEmbeddingModel(api_key="test-key")
    assert model.api_key == "test-key"


def test_openai_initialization_with_env_var():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
        model = OpenAIEmbeddingModel()
        assert model.api_key == "env-test-key"


def test_openai_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAIEmbeddingModel()


def test_openai_embed(openai_embedding_model):
    texts = ["Hello, world!", "Test text"]
    embeddings = openai_embedding_model.embed(texts)

    openai_embedding_model.client.embeddings.create.assert_called_once()
    call_kwargs = openai_embedding_model.client.embeddings.create.call_args[1]

    assert call_kwargs["input"] == texts
    assert call_kwargs["model"] == "text-embedding-3-small"
    assert embeddings == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_openai_aembed(openai_embedding_model):
    texts = ["Hello, world!", "Test text"]
    embeddings = await openai_embedding_model.aembed(texts)

    openai_embedding_model.async_client.embeddings.create.assert_called_once()
    call_kwargs = openai_embedding_model.async_client.embeddings.create.call_args[1]

    assert call_kwargs["input"] == texts
    assert call_kwargs["model"] == "text-embedding-3-small"
    assert embeddings == [[0.1, 0.2, 0.3]]


# Tests for Ollama Embedding Provider
def test_ollama_provider_name(ollama_embedding_model):
    """Test provider name."""
    assert ollama_embedding_model.provider == "ollama"


def test_ollama_initialization_with_base_url():
    """Test initialization with base URL."""
    model = OllamaEmbeddingModel(base_url="http://custom:11434")
    assert model.base_url == "http://custom:11434"


def test_ollama_initialization_with_env_var():
    """Test initialization with environment variable."""
    with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env:11434"}):
        model = OllamaEmbeddingModel()
        assert model.base_url == "http://env:11434"


def test_ollama_initialization_default():
    """Test initialization with default URL."""
    # Reset environment variables
    with patch.dict(os.environ, {}, clear=True):
        model = OllamaEmbeddingModel()
        assert model.base_url == "http://localhost:11434"


def test_ollama_get_api_kwargs():
    """Test _get_api_kwargs method."""
    model = OllamaEmbeddingModel(model_name="llama2", base_url="http://test:11434")
    kwargs = model._get_api_kwargs()
    assert "model_name" not in kwargs
    assert "base_url" not in kwargs


def test_ollama_embed_empty_text():
    """Test embed method with empty text."""
    model = OllamaEmbeddingModel()
    with pytest.raises(ValueError, match="Text cannot be empty"):
        model.embed([""])


def test_ollama_embed_none_text():
    """Test embed method with None text."""
    model = OllamaEmbeddingModel()
    with pytest.raises(ValueError, match="Text cannot be None"):
        model.embed([None])


def test_ollama_embed(ollama_embedding_model):
    """Test embed method."""
    with patch.object(ollama_embedding_model.client, "embeddings") as mock_embed:
        mock_response = Mock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_embed.return_value = mock_response
        texts = ["Hello, world!"]
        embeddings = ollama_embedding_model.embed(texts)
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with(model="mxbai-embed-large", prompt="Hello, world!")


@pytest.mark.asyncio
async def test_ollama_aembed(ollama_embedding_model):
    """Test async embed method."""
    with patch.object(ollama_embedding_model.async_client, "embeddings") as mock_embed:
        mock_response = Mock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_embed.return_value = mock_response
        texts = ["Hello, world!"]
        embeddings = await ollama_embedding_model.aembed(texts)
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with(model="mxbai-embed-large", prompt="Hello, world!")


def test_ollama_embed_multiple_texts(ollama_embedding_model):
    """Test embed method with multiple texts."""
    with patch.object(ollama_embedding_model.client, "embeddings") as mock_embed:
        mock_response1 = Mock()
        mock_response1.embedding = [0.1, 0.2, 0.3]
        mock_response2 = Mock()
        mock_response2.embedding = [0.4, 0.5, 0.6]
        mock_embed.side_effect = [mock_response1, mock_response2]
        texts = ["Hello, world!", "Another text"]
        embeddings = ollama_embedding_model.embed(texts)
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        assert mock_embed.call_count == 2
        mock_embed.assert_has_calls(
            [
                call(model="mxbai-embed-large", prompt="Hello, world!"),
                call(model="mxbai-embed-large", prompt="Another text"),
            ]
        )


# Tests for Google Embedding Provider
def test_google_provider_name():
    model = GoogleEmbeddingModel(api_key="test-key")
    assert model.provider == "google"


def test_google_initialization_with_api_key():
    model = GoogleEmbeddingModel(api_key="test-key")
    assert model.api_key == "test-key"


def test_google_initialization_with_env_var():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-test-key"}):
        model = GoogleEmbeddingModel()
        assert model.api_key == "env-test-key"


def test_google_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Google API key not found"):
            GoogleEmbeddingModel()


@pytest.fixture
def google_embedding_model():
    with patch("google.generativeai.embed_content") as mock_embed:
        mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
        yield GoogleEmbeddingModel(api_key="test_key")


def test_google_embed(google_embedding_model):
    texts = ["Hello, world!"]
    embeddings = google_embedding_model.embed(texts)
    assert embeddings == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_google_aembed(google_embedding_model):
    texts = ["Hello, world!"]
    embeddings = await google_embedding_model.aembed(texts)
    assert embeddings == [[0.1, 0.2, 0.3]]


# Tests for Vertex Embedding Provider
def test_vertex_provider_name(vertex_embedding_model):
    assert vertex_embedding_model.provider == "vertex"


def test_vertex_initialization_with_project():
    model = VertexEmbeddingModel(vertex_project="test-project")
    assert model.project_id == "test-project"


def test_vertex_initialization_with_env_var():
    with patch.dict(os.environ, {"VERTEX_PROJECT": "env-test-project"}):
        model = VertexEmbeddingModel()
        assert model.project_id == "env-test-project"


def test_vertex_initialization_without_project():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Google Cloud project ID not found"):
            VertexEmbeddingModel()


def test_vertex_embed(vertex_embedding_model):
    texts = ["Hello, world!"]
    embeddings = vertex_embedding_model.embed(texts)
    assert embeddings == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_vertex_aembed(vertex_embedding_model):
    texts = ["Hello, world!"]
    embeddings = await vertex_embedding_model.aembed(texts)
    assert embeddings == [[0.1, 0.2, 0.3]]
