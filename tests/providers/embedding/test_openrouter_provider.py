"""Tests for OpenRouter embedding provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.providers.embedding.openrouter import OpenRouterEmbeddingModel


@pytest.fixture
def mock_openrouter_response():
    """Mock httpx response for OpenRouter API."""
    return {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ],
        "model": "openai/text-embedding-3-small",
        "usage": {
            "total_tokens": 10
        }
    }


@pytest.fixture
def openrouter_model(mock_openrouter_response):
    """Create OpenRouterEmbeddingModel with mocked HTTP client."""
    model = OpenRouterEmbeddingModel(api_key="test-key", model_name="openai/text-embedding-3-small")

    # Mock the HTTP clients
    mock_client = Mock()
    mock_async_client = AsyncMock()

    def mock_post_side_effect(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openrouter_response
        return mock_response

    async def mock_async_post_side_effect(url, **kwargs):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=mock_openrouter_response)
        return mock_response

    mock_client.post.side_effect = mock_post_side_effect
    mock_async_client.post.side_effect = mock_async_post_side_effect

    model.client = mock_client
    model.async_client = mock_async_client

    return model


def test_init_with_api_key():
    """Test initialization with API key."""
    model = OpenRouterEmbeddingModel(api_key="test-key")
    assert model.api_key == "test-key"
    assert model.base_url == "https://openrouter.ai/api/v1"


def test_init_with_env_api_key():
    """Test initialization with API key from environment."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-test-key"}):
        model = OpenRouterEmbeddingModel()
        assert model.api_key == "env-test-key"


def test_init_with_custom_base_url():
    """Test initialization with custom base URL."""
    with patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test-key",
        "OPENROUTER_BASE_URL": "https://custom.openrouter.ai/v1"
    }):
        model = OpenRouterEmbeddingModel()
        assert model.base_url == "https://custom.openrouter.ai/v1"


def test_init_without_api_key():
    """Test initialization without API key raises error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenRouter API key not found"):
            OpenRouterEmbeddingModel()


def test_get_default_model():
    """Test getting default model name."""
    model = OpenRouterEmbeddingModel(api_key="test-key")
    assert model.get_model_name() == "openai/text-embedding-3-small"


def test_provider_name():
    """Test getting provider name."""
    model = OpenRouterEmbeddingModel(api_key="test-key")
    assert model.provider == "openrouter"


def test_custom_model_name():
    """Test initialization with custom model name."""
    model = OpenRouterEmbeddingModel(api_key="test-key", model_name="cohere/embed-english-v3.0")
    assert model.get_model_name() == "cohere/embed-english-v3.0"


def test_get_headers():
    """Test that OpenRouter-specific headers are included."""
    model = OpenRouterEmbeddingModel(api_key="test-key")
    headers = model._get_headers()

    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"
    assert "HTTP-Referer" in headers
    assert headers["HTTP-Referer"] == "https://github.com/lfnovo/esperanto"
    assert "X-Title" in headers
    assert headers["X-Title"] == "Esperanto"
    assert "Content-Type" in headers


def test_embed(openrouter_model):
    """Test embedding creation."""
    texts = ["Hello", "World"]
    embeddings = openrouter_model.embed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    # Verify HTTP POST was called
    openrouter_model.client.post.assert_called_once()
    call_args = openrouter_model.client.post.call_args

    # Check URL
    assert call_args[0][0] == "https://openrouter.ai/api/v1/embeddings"

    # Check that data parameter was used (not json)
    assert "data" in call_args[1]
    assert "json" not in call_args[1]

    # Check headers include OpenRouter-specific headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert "HTTP-Referer" in headers
    assert headers["HTTP-Referer"] == "https://github.com/lfnovo/esperanto"
    assert "X-Title" in headers
    assert headers["X-Title"] == "Esperanto"


@pytest.mark.asyncio
async def test_aembed(openrouter_model):
    """Test async embedding creation."""
    texts = ["Hello"]
    embeddings = await openrouter_model.aembed(texts)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]

    # Verify async HTTP POST was called
    openrouter_model.async_client.post.assert_called_once()
    call_args = openrouter_model.async_client.post.call_args

    # Check URL
    assert call_args[0][0] == "https://openrouter.ai/api/v1/embeddings"

    # Check that data parameter was used (not json)
    assert "data" in call_args[1]
    assert "json" not in call_args[1]

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers


def test_embed_error_handling(openrouter_model):
    """Test error handling in embedding creation."""
    def mock_error_post(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.text = "Unauthorized"
        return mock_response

    openrouter_model.client.post.side_effect = mock_error_post

    with pytest.raises(RuntimeError, match="OpenRouter API error: Invalid API key"):
        openrouter_model.embed(["test"])


def test_embed_error_handling_rate_limit(openrouter_model):
    """Test error handling for rate limit errors."""
    def mock_error_post(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_response.text = "Too Many Requests"
        return mock_response

    openrouter_model.client.post.side_effect = mock_error_post

    with pytest.raises(RuntimeError, match="OpenRouter API error: Rate limit exceeded"):
        openrouter_model.embed(["test"])


def test_embed_error_handling_server_error(openrouter_model):
    """Test error handling for server errors."""
    def mock_error_post(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_response.text = "Internal Server Error"
        return mock_response

    openrouter_model.client.post.side_effect = mock_error_post

    with pytest.raises(RuntimeError, match="OpenRouter API error: HTTP 500"):
        openrouter_model.embed(["test"])


def test_text_cleaning(openrouter_model):
    """Test that newlines in texts are replaced with spaces."""
    texts = ["Hello\nWorld", "Test\nText"]
    openrouter_model.embed(texts)

    # Check that the input was cleaned - need to parse the JSON from data parameter
    call_args = openrouter_model.client.post.call_args
    import json
    data_str = call_args[1]["data"]
    payload = json.loads(data_str)
    assert payload["input"] == ["Hello World", "Test Text"]


def test_models_list():
    """Test listing available models uses dedicated embeddings endpoint."""
    model = OpenRouterEmbeddingModel(api_key="test-key")

    # Mock the HTTP client
    mock_client = Mock()

    # Mock response from /embeddings/models endpoint (only contains embedding models)
    embeddings_models_response = {
        "data": [
            {
                "id": "openai/text-embedding-3-small",
                "context_length": 8191
            },
            {
                "id": "openai/text-embedding-3-large",
                "context_length": 8191
            },
            {
                "id": "cohere/embed-english-v3.0",
                "context_length": 512
            },
            {
                "id": "voyage/voyage-3-lite",
                "context_length": 32000
            }
        ]
    }

    def mock_get_side_effect(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = embeddings_models_response
        return mock_response

    mock_client.get.side_effect = mock_get_side_effect
    model.client = mock_client

    models = model._get_models()

    # Verify the correct endpoint was called
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args
    assert "/embeddings/models" in call_args[0][0]

    # Should include all models from the embeddings endpoint
    assert len(models) == 4
    model_ids = [m.id for m in models]

    assert "openai/text-embedding-3-small" in model_ids
    assert "openai/text-embedding-3-large" in model_ids
    assert "cohere/embed-english-v3.0" in model_ids
    assert "voyage/voyage-3-lite" in model_ids


def test_models_list_owned_by():
    """Test that model owned_by is correctly extracted from model ID."""
    model = OpenRouterEmbeddingModel(api_key="test-key")

    mock_client = Mock()

    def mock_get_side_effect(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "openai/text-embedding-3-small", "context_length": 8191},
                {"id": "cohere/embed-english-v3.0", "context_length": 512}
            ]
        }
        return mock_response

    mock_client.get.side_effect = mock_get_side_effect
    model.client = mock_client

    models = model._get_models()

    # Check owned_by is extracted from provider prefix
    assert models[0].owned_by == "openai"
    assert models[1].owned_by == "cohere"


def test_embed_with_config():
    """Test initialization with config dictionary."""
    config = {
        "api_key": "config-key",
        "base_url": "https://custom.api.com/v1"
    }
    model = OpenRouterEmbeddingModel(config=config)

    assert model.api_key == "config-key"
    assert model.base_url == "https://custom.api.com/v1"
