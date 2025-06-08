"""Tests for Voyage AI embedding provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.providers.embedding.voyage import VoyageEmbeddingModel


@pytest.fixture
def mock_voyage_response():
    """Mock httpx response for Voyage API."""
    return {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ],
        "model": "voyage-3-large",
        "usage": {
            "total_tokens": 10
        }
    }


@pytest.fixture
def voyage_model(mock_voyage_response):
    """Create VoyageEmbeddingModel with mocked HTTP client."""
    model = VoyageEmbeddingModel(api_key="test-key", model_name="voyage-3-large")
    
    # Mock the HTTP clients
    mock_client = Mock()
    mock_async_client = AsyncMock()
    
    def mock_post_side_effect(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_voyage_response
        return mock_response
    
    async def mock_async_post_side_effect(url, **kwargs):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=mock_voyage_response)
        return mock_response
    
    mock_client.post.side_effect = mock_post_side_effect
    mock_async_client.post.side_effect = mock_async_post_side_effect
    
    model.client = mock_client
    model.async_client = mock_async_client
    
    return model


def test_init_with_api_key():
    """Test initialization with API key."""
    model = VoyageEmbeddingModel(api_key="test-key")
    assert model.api_key == "test-key"


def test_init_with_env_api_key():
    """Test initialization with API key from environment."""
    with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
        model = VoyageEmbeddingModel()
        assert model.api_key == "test-key"


def test_init_without_api_key():
    """Test initialization without API key raises error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Voyage API key not found"):
            VoyageEmbeddingModel()


def test_get_default_model():
    """Test getting default model name."""
    model = VoyageEmbeddingModel(api_key="test-key")
    assert model.get_model_name() == "voyage-3-large"


def test_provider_name():
    """Test getting provider name."""
    model = VoyageEmbeddingModel(api_key="test-key")
    assert model.provider == "voyage"


def test_models_list():
    """Test listing available models."""
    model = VoyageEmbeddingModel(api_key="test-key")
    models = model.models
    assert len(models) == 7  # Updated count based on new models
    model_ids = [m.id for m in models]
    assert "voyage-3-large" in model_ids
    assert "voyage-code-3" in model_ids


def test_embed(voyage_model):
    """Test embedding creation."""
    texts = ["Hello", "World"]
    embeddings = voyage_model.embed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    
    # Verify HTTP POST was called
    voyage_model.client.post.assert_called_once()
    call_args = voyage_model.client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.voyageai.com/v1/embeddings"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["input"] == ["Hello", "World"]
    assert json_payload["model"] == "voyage-3-large"
    
    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_aembed(voyage_model):
    """Test async embedding creation."""
    texts = ["Hello"]
    embeddings = await voyage_model.aembed(texts)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    
    # Verify async HTTP POST was called
    voyage_model.async_client.post.assert_called_once()
    call_args = voyage_model.async_client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.voyageai.com/v1/embeddings"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["input"] == ["Hello"]
    assert json_payload["model"] == "voyage-3-large"


def test_embed_error_handling(voyage_model):
    """Test error handling in embedding creation."""
    def mock_error_post(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.text = "Unauthorized"
        return mock_response
    
    voyage_model.client.post.side_effect = mock_error_post
    
    with pytest.raises(RuntimeError, match="Voyage API error: Invalid API key"):
        voyage_model.embed(["test"])


def test_text_cleaning(voyage_model):
    """Test that newlines in texts are replaced with spaces."""
    texts = ["Hello\nWorld", "Test\nText"]
    voyage_model.embed(texts)
    
    # Check that the input was cleaned
    call_args = voyage_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["input"] == ["Hello World", "Test Text"]