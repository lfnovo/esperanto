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
    """Mock HTTP response for OpenAI embeddings API."""
    return {
        "data": [
            {
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": "text-embedding-3-small",
        "object": "list",
        "usage": {"total_tokens": 4}
    }


@pytest.fixture
def mock_openai_models_response():
    """Mock HTTP response for OpenAI models API."""
    return {
        "object": "list",
        "data": [
            {
                "id": "text-embedding-3-small",
                "object": "model",
                "owned_by": "openai"
            },
            {
                "id": "text-embedding-3-large",
                "object": "model",
                "owned_by": "openai"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "owned_by": "openai"
            }
        ]
    }


@pytest.fixture
def mock_ollama_embedding_response():
    return {
        "embedding": [0.1, 0.2, 0.3]
    }


@pytest.fixture
def mock_google_embedding_response():
    return {"embedding": [0.1, 0.2, 0.3]}


@pytest.fixture
def mock_vertex_embedding_response():
    return {
        "predictions": [
            {
                "embeddings": {
                    "values": [0.1, 0.2, 0.3]
                }
            }
        ]
    }


# Mock clients
@pytest.fixture
def mock_openai_embedding_client(mock_openai_embedding_response, mock_openai_models_response):
    """Mock httpx clients for OpenAI embeddings."""
    client = Mock()
    async_client = AsyncMock()

    # Mock HTTP response objects
    def make_response(status_code, data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data
        return response

    def make_async_response(status_code, data):
        response = AsyncMock()
        response.status_code = status_code
        response.json = Mock(return_value=data)
        return response

    # Configure responses based on URL
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/embeddings"):
            return make_response(200, mock_openai_embedding_response)
        return make_response(404, {"error": "Not found"})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_response(200, mock_openai_models_response)
        return make_response(404, {"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/embeddings"):
            return make_async_response(200, mock_openai_embedding_response)
        return make_async_response(404, {"error": "Not found"})

    async def mock_async_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_async_response(200, mock_openai_models_response)
        return make_async_response(404, {"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    async_client.get.side_effect = mock_async_get_side_effect

    return client, async_client


@pytest.fixture
def mock_ollama_response(mock_ollama_embedding_response):
    """Mock httpx clients for Ollama embeddings."""
    client = Mock()
    async_client = AsyncMock()

    # Mock HTTP response objects
    def make_response(status_code, data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data
        return response

    def make_async_response(status_code, data):
        response = AsyncMock()
        response.status_code = status_code
        response.json = Mock(return_value=data)
        return response

    # Configure responses based on URL
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/api/embeddings"):
            return make_response(200, mock_ollama_embedding_response)
        elif url.endswith("/api/tags"):
            return make_response(200, {"models": [{"name": "mxbai-embed-large"}]})
        return make_response(404, {"error": "Not found"})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/api/tags"):
            return make_response(200, {"models": [{"name": "mxbai-embed-large"}]})
        return make_response(404, {"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/api/embeddings"):
            return make_async_response(200, mock_ollama_embedding_response)
        elif url.endswith("/api/tags"):
            return make_async_response(200, {"models": [{"name": "mxbai-embed-large"}]})
        return make_async_response(404, {"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect

    return client, async_client


@pytest.fixture
def mock_vertex_client(mock_vertex_embedding_response):
    """Mock httpx clients for Vertex AI API."""
    client = Mock()
    async_client = AsyncMock()
    
    # Mock HTTP response objects
    def make_response(status_code, data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data
        return response
    
    def make_async_response(status_code, data):
        response = AsyncMock()
        response.status_code = status_code
        response.json = Mock(return_value=data)
        return response
    
    # Configure responses
    def mock_post_side_effect(url, **kwargs):
        if "predict" in url:
            return make_response(200, mock_vertex_embedding_response)
        return make_response(404, {"error": "Not found"})
    
    async def mock_async_post_side_effect(url, **kwargs):
        if "predict" in url:
            return make_async_response(200, mock_vertex_embedding_response)
        return make_async_response(404, {"error": "Not found"})
    
    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    
    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    
    return client, async_client


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
    model.client, model.async_client = mock_ollama_response
    return model


@pytest.fixture
def vertex_embedding_model(mock_vertex_client):
    with patch('subprocess.run') as mock_subprocess:
        mock_subprocess.return_value.stdout = "mock_access_token"
        mock_subprocess.return_value.returncode = 0
        
        model = VertexEmbeddingModel(
            vertex_project="test-project", model_name="textembedding-gecko"
        )
        model.client, model.async_client = mock_vertex_client
        
        # Mock the instance method directly to prevent token calls during tests
        model._get_access_token = Mock(return_value="mock_access_token")
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

    # Verify HTTP POST was called
    openai_embedding_model.client.post.assert_called_once()
    call_args = openai_embedding_model.client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/embeddings"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["input"] == texts
    assert json_payload["model"] == "text-embedding-3-small"
    
    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"
    
    assert embeddings == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_openai_aembed(openai_embedding_model):
    texts = ["Hello, world!", "Test text"]
    embeddings = await openai_embedding_model.aembed(texts)

    # Verify async HTTP POST was called
    openai_embedding_model.async_client.post.assert_called_once()
    call_args = openai_embedding_model.async_client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/embeddings"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["input"] == texts
    assert json_payload["model"] == "text-embedding-3-small"
    
    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"
    
    assert embeddings == [[0.1, 0.2, 0.3]]


def test_openai_models(openai_embedding_model):
    """Test that the models property works with HTTP."""
    models = openai_embedding_model.models
    
    # Verify HTTP GET was called
    openai_embedding_model.client.get.assert_called_with(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }
    )
    
    # Check that only embedding models are returned
    assert len(models) == 2
    assert models[0].id == "text-embedding-3-small"
    assert models[1].id == "text-embedding-3-large"
    assert models[0].type == "embedding"
    assert models[1].type == "embedding"


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
    texts = ["Hello, world!"]
    embeddings = ollama_embedding_model.embed(texts)
    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
    
    # Verify HTTP POST was called
    ollama_embedding_model.client.post.assert_called_once()
    call_args = ollama_embedding_model.client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "http://localhost:11434/api/embeddings"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["prompt"] == "Hello, world!"
    assert json_payload["model"] == "mxbai-embed-large"


@pytest.mark.asyncio
async def test_ollama_aembed(ollama_embedding_model):
    """Test async embed method."""
    texts = ["Hello, world!"]
    embeddings = await ollama_embedding_model.aembed(texts)
    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
    
    # Verify async HTTP POST was called
    ollama_embedding_model.async_client.post.assert_called_once()
    call_args = ollama_embedding_model.async_client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "http://localhost:11434/api/embeddings"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["prompt"] == "Hello, world!"
    assert json_payload["model"] == "mxbai-embed-large"


def test_ollama_embed_multiple_texts(ollama_embedding_model):
    """Test embed method with multiple texts."""
    # Mock different responses for each call
    def mock_post_side_effect(url, **kwargs):
        response = Mock()
        response.status_code = 200
        # Return different embeddings based on the prompt
        prompt = kwargs.get("json", {}).get("prompt", "")
        if prompt == "Hello, world!":
            response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        elif prompt == "Another text":
            response.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
        else:
            response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        return response
    
    ollama_embedding_model.client.post.side_effect = mock_post_side_effect
    
    texts = ["Hello, world!", "Another text"]
    embeddings = ollama_embedding_model.embed(texts)
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    assert ollama_embedding_model.client.post.call_count == 2


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
    """Create GoogleEmbeddingModel with mocked HTTP client."""
    model = GoogleEmbeddingModel(api_key="test-key", model_name="text-embedding-004")
    
    # Mock the HTTP clients
    mock_client = Mock()
    mock_async_client = AsyncMock()
    
    def mock_post_side_effect(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        
        if "embedContent" in url:
            mock_response.json.return_value = {
                "embedding": {
                    "values": [0.1, 0.2, 0.3]
                }
            }
        elif "models" in url:
            mock_response.json.return_value = {
                "models": [
                    {
                        "name": "models/text-embedding-004", 
                        "inputTokenLimit": 2048,
                        "supportedGenerationMethods": ["embedContent"]
                    }
                ]
            }
        else:
            mock_response.json.return_value = {}
        return mock_response
    
    async def mock_async_post_side_effect(url, **kwargs):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        
        if "embedContent" in url:
            mock_response.json = Mock(return_value={
                "embedding": {
                    "values": [0.1, 0.2, 0.3]
                }
            })
        elif "models" in url:
            mock_response.json = Mock(return_value={
                "models": [
                    {
                        "name": "models/text-embedding-004", 
                        "inputTokenLimit": 2048,
                        "supportedGenerationMethods": ["embedContent"]
                    }
                ]
            })
        else:
            mock_response.json = Mock(return_value={})
        return mock_response

    def mock_get_side_effect(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "models/text-embedding-004", 
                    "inputTokenLimit": 2048,
                    "supportedGenerationMethods": ["embedContent"]
                }
            ]
        }
        return mock_response
    
    mock_client.post.side_effect = mock_post_side_effect
    mock_client.get.side_effect = mock_get_side_effect
    mock_async_client.post.side_effect = mock_async_post_side_effect
    
    model.client = mock_client
    model.async_client = mock_async_client
    
    yield model


def test_google_embed(google_embedding_model):
    texts = ["Hello, world!"]
    embeddings = google_embedding_model.embed(texts)
    assert embeddings == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_google_aembed(google_embedding_model):
    texts = ["Hello, world!"]
    embeddings = await google_embedding_model.aembed(texts)
    assert embeddings == [[0.1, 0.2, 0.3]]


def test_google_embed_with_task_type():
    """Test Google embeddings with task type configuration."""
    from esperanto.common_types.task_type import EmbeddingTaskType
    
    with patch('httpx.Client.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
        
        model = GoogleEmbeddingModel(
            api_key="test-key",
            model_name="text-embedding-004",
            config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
        )
        
        embeddings = model.embed(["Hello, world!"])
        assert embeddings == [[0.1, 0.2, 0.3]]
        
        # Verify task_type was included in API call
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["task_type"] == "RETRIEVAL_QUERY"


@pytest.mark.asyncio
async def test_google_aembed_with_task_type():
    """Test Google async embeddings with task type configuration."""
    from esperanto.common_types.task_type import EmbeddingTaskType
    
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
        mock_post.return_value = mock_response
        
        model = GoogleEmbeddingModel(
            api_key="test-key", 
            model_name="text-embedding-004",
            config={"task_type": EmbeddingTaskType.CLASSIFICATION}
        )
        
        embeddings = await model.aembed(["Hello, world!"])
        assert embeddings == [[0.1, 0.2, 0.3]]
        
        # Verify task_type was included in async API call
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["task_type"] == "CLASSIFICATION"


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
