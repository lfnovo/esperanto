from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.providers.embedding.mistral import MistralEmbeddingModel


@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for Mistral API."""
    def create_response(texts):
        if not texts:
            return {
                "data": [],
                "object": "list",
                "usage": {"prompt_tokens": 0, "total_tokens": 0}
            }
        return {
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": [0.1, 0.2, 0.3]
                } for i, _ in enumerate(texts)
            ],
            "object": "list",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
    return create_response

@pytest.fixture
def mistral_embedding_model(mock_httpx_response):
    """Create MistralEmbeddingModel with mocked HTTP client."""
    model = MistralEmbeddingModel(api_key="test-key", model_name="mistral-embed")
    
    # Mock the HTTP clients
    mock_client = Mock()
    mock_async_client = AsyncMock()
    
    def mock_post(url, **kwargs):
        json_data = kwargs.get('json', {})
        texts = json_data.get('input', [])
        response_data = mock_httpx_response(texts)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        return mock_response
    
    async def mock_async_post(url, **kwargs):
        json_data = kwargs.get('json', {})
        texts = json_data.get('input', [])
        response_data = mock_httpx_response(texts)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        return mock_response
    
    mock_client.post = mock_post
    mock_async_client.post = mock_async_post
    
    model.client = mock_client
    model.async_client = mock_async_client
    return model

def test_provider_name(mistral_embedding_model):
    assert mistral_embedding_model.provider == "mistral"

def test_initialization_with_api_key():
    model = MistralEmbeddingModel(api_key="test-key")
    assert model.api_key == "test-key"

def test_initialization_with_env_var(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "env-test-key")
    model = MistralEmbeddingModel()
    assert model.api_key == "env-test-key"

def test_initialization_without_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Mistral API key not found"):
        MistralEmbeddingModel()

def test_embed(mistral_embedding_model):
    texts = ["Hello world"]
    result = mistral_embedding_model.embed(texts)
    assert result == [[0.1, 0.2, 0.3]]

async def test_aembed(mistral_embedding_model):
    texts = ["Hello world"]
    result = await mistral_embedding_model.aembed(texts)
    assert result == [[0.1, 0.2, 0.3]]

def test_embed_empty(mistral_embedding_model):
    texts = []
    result = mistral_embedding_model.embed(texts)
    assert result == []

def test_get_default_model():
    model = MistralEmbeddingModel(api_key="test-key")
    assert model._get_default_model() == "mistral-embed"


def test_embed_null_embedding_raises():
    """embed() raises RuntimeError with index when response has null embedding."""
    model = MistralEmbeddingModel(api_key="test-key")
    null_response = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": None, "index": 1},
        ],
        "object": "list",
    }
    mock_client = Mock()

    def mock_post(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = null_response
        return mock_response

    mock_client.post = mock_post
    model.client = mock_client

    with pytest.raises(RuntimeError) as exc_info:
        model.embed(["text one", "text two"])
    assert "1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_aembed_null_embedding_raises():
    """aembed() raises RuntimeError with index when response has null embedding."""
    model = MistralEmbeddingModel(api_key="test-key")
    null_response = {
        "data": [
            {"embedding": None, "index": 0},
        ],
        "object": "list",
    }
    mock_async_client = AsyncMock()

    async def mock_async_post(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = null_response
        return mock_response

    mock_async_client.post = mock_async_post
    model.async_client = mock_async_client

    with pytest.raises(RuntimeError) as exc_info:
        await model.aembed(["text one"])
    assert "0" in str(exc_info.value)