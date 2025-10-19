"""Tests for the OpenAI-compatible Embedding provider."""
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from esperanto.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingModel


@pytest.fixture
def mock_embedding_response():
    """Mock embedding response data."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            {
                "object": "embedding",
                "index": 1,
                "embedding": [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        ],
        "model": "nomic-embed-text",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }


@pytest.fixture
def mock_openai_compatible_models_response():
    """Mock HTTP response for OpenAI-compatible models API."""
    return {
        "object": "list",
        "data": [
            {
                "id": "nomic-embed-text",
                "object": "model",
                "owned_by": "custom"
            },
            {
                "id": "text-embedding-3-small",
                "object": "model",
                "owned_by": "custom"
            }
        ]
    }


@pytest.fixture
def mock_httpx_clients(mock_embedding_response, mock_openai_compatible_models_response):
    """Mock httpx clients for OpenAI-compatible embeddings."""
    client = Mock()
    async_client = AsyncMock()

    # Mock HTTP response objects
    def make_response(status_code, json_data=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        return response

    def make_async_response(status_code, json_data=None):
        response = AsyncMock()
        response.status_code = status_code
        if json_data is not None:
            response.json = Mock(return_value=json_data)
        return response

    # Configure responses based on URL
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/embeddings"):
            return make_response(200, json_data=mock_embedding_response)
        return make_response(404, json_data={"error": "Not found"})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_response(200, json_data=mock_openai_compatible_models_response)
        return make_response(404, json_data={"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/embeddings"):
            return make_async_response(200, json_data=mock_embedding_response)
        return make_async_response(404, json_data={"error": "Not found"})

    async def mock_async_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_async_response(200, json_data=mock_openai_compatible_models_response)
        return make_async_response(404, json_data={"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    async_client.get.side_effect = mock_async_get_side_effect

    return client, async_client


@pytest.fixture
def embedding_model(mock_httpx_clients):
    """Create an embedding model instance with mocked HTTP clients."""
    model = OpenAICompatibleEmbeddingModel(
        model_name="nomic-embed-text",
        api_key="test-key",
        base_url="http://localhost:1234/v1"
    )
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init_with_config():
    """Test model initialization with config."""
    model = OpenAICompatibleEmbeddingModel(
        model_name="nomic-embed-text",
        api_key="test-key",
        base_url="http://localhost:1234/v1"
    )
    assert model.model_name == "nomic-embed-text"
    assert model.provider == "openai-compatible"
    assert model.base_url == "http://localhost:1234/v1"
    assert model.api_key == "test-key"


def test_init_with_env_vars(monkeypatch):
    """Test model initialization with environment variables."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1235/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "env-key")

    model = OpenAICompatibleEmbeddingModel(model_name="test-model")
    assert model.base_url == "http://localhost:1235/v1"
    assert model.api_key == "env-key"


def test_init_missing_base_url(monkeypatch):
    """Test that initialization fails without base URL."""
    # Clear all environment variables
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OpenAI-compatible base URL is required"):
        OpenAICompatibleEmbeddingModel(model_name="test-model")


def test_base_url_trailing_slash():
    """Test that trailing slash is stripped from base URL."""
    model = OpenAICompatibleEmbeddingModel(
        model_name="test-model",
        base_url="http://localhost:1234/v1/"
    )
    assert model.base_url == "http://localhost:1234/v1"


def test_timeout_configuration():
    """Test timeout configuration."""
    model = OpenAICompatibleEmbeddingModel(
        model_name="test-model",
        base_url="http://localhost:1234/v1",
        config={"timeout": 300}
    )
    assert model.timeout == 300


def test_embed(embedding_model):
    """Test synchronous embedding generation."""
    texts = ["Hello world", "How are you?"]
    embeddings = embedding_model.embed(texts)

    # Verify HTTP POST was called
    embedding_model.client.post.assert_called_once()
    call_args = embedding_model.client.post.call_args

    # Check URL
    assert call_args[0][0] == "http://localhost:1234/v1/embeddings"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"

    # Check JSON payload
    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "nomic-embed-text"
    assert json_payload["input"] == texts

    # Check response
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert embeddings[1] == [0.6, 0.7, 0.8, 0.9, 1.0]


@pytest.mark.asyncio
async def test_aembed(embedding_model):
    """Test asynchronous embedding generation."""
    texts = ["Hello async world", "How are you doing?"]
    embeddings = await embedding_model.aembed(texts)

    # Verify async HTTP POST was called
    embedding_model.async_client.post.assert_called_once()
    call_args = embedding_model.async_client.post.call_args

    # Check URL
    assert call_args[0][0] == "http://localhost:1234/v1/embeddings"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"

    # Check JSON payload
    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "nomic-embed-text"
    assert json_payload["input"] == texts

    # Check response
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert embeddings[1] == [0.6, 0.7, 0.8, 0.9, 1.0]


def test_models(embedding_model):
    """Test that the models property works with HTTP."""
    models = embedding_model.models

    # Verify HTTP GET was called
    embedding_model.client.get.assert_called_with(
        "http://localhost:1234/v1/models",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }
    )

    # Check that models are returned
    assert len(models) == 2
    assert models[0].id == "nomic-embed-text"
    assert models[1].id == "text-embedding-3-small"
    assert models[0].type == "embedding"
    assert models[1].type == "embedding"
    assert models[0].owned_by == "custom"
    assert models[1].owned_by == "custom"


def test_models_fallback():
    """Test that models property falls back gracefully when endpoint doesn't support it."""
    model = OpenAICompatibleEmbeddingModel(
        api_key="test-key",
        model_name="test-model",
        base_url="http://localhost:1234/v1"
    )

    # Mock client that throws exception for models endpoint
    client = Mock()
    client.get.side_effect = Exception("Connection error")
    model.client = client

    models = model.models

    # Should return empty list
    assert models == []


def test_error_handling():
    """Test error handling for HTTP errors."""
    model = OpenAICompatibleEmbeddingModel(
        api_key="test-key",
        model_name="test-model",
        base_url="http://localhost:1234/v1"
    )

    # Mock client that returns error response
    client = Mock()
    response = Mock()
    response.status_code = 500
    response.text = "Internal server error"
    response.json.side_effect = Exception("Not JSON")
    client.post.return_value = response
    model.client = client

    with pytest.raises(RuntimeError, match="OpenAI-compatible embedding endpoint error"):
        model.embed(["test"])


def test_embed_with_additional_kwargs(embedding_model):
    """Test embedding generation with additional parameters."""
    texts = ["Hello world"]
    embeddings = embedding_model.embed(texts, user="test-user", encoding_format="float")

    # Check that additional kwargs are passed through
    call_args = embedding_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["user"] == "test-user"
    assert json_payload["encoding_format"] == "float"


def test_text_cleaning(embedding_model):
    """Test that text cleaning is applied."""
    texts = ["Hello    world  !", "This has\n\nnewlines   and    spacing"]
    embedding_model.embed(texts)

    # Check that texts were cleaned
    call_args = embedding_model.client.post.call_args
    json_payload = call_args[1]["json"]
    cleaned_texts = json_payload["input"]

    # Should normalize spacing and clean text
    assert "    " not in cleaned_texts[0]  # Multiple spaces should be normalized
    assert "\n\n" not in cleaned_texts[1]  # Newlines should be replaced


def test_get_default_model():
    """Test default model name."""
    model = OpenAICompatibleEmbeddingModel(
        base_url="http://localhost:1234/v1"
    )
    assert model._get_default_model() == "text-embedding-3-small"


def test_provider_name(embedding_model):
    """Test provider name."""
    assert embedding_model.provider == "openai-compatible"


def test_get_model_name(embedding_model):
    """Test getting model name."""
    assert embedding_model.get_model_name() == "nomic-embed-text"


def test_get_headers(embedding_model):
    """Test header generation."""
    headers = embedding_model._get_headers()
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"


def test_provider_specific_env_var_precedence(monkeypatch):
    """Test that provider-specific env vars take precedence over generic ones."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING", "http://embedding-specific:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_EMBEDDING", "embedding-specific-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-key")

    model = OpenAICompatibleEmbeddingModel(model_name="test-model")
    assert model.base_url == "http://embedding-specific:1234"
    assert model.api_key == "embedding-specific-key"


def test_fallback_to_generic_env_var(monkeypatch):
    """Test fallback to generic env vars when provider-specific ones are not set."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-key")

    model = OpenAICompatibleEmbeddingModel(model_name="test-model")
    assert model.base_url == "http://generic:5678"
    assert model.api_key == "generic-key"


def test_config_overrides_provider_specific_env_vars(monkeypatch):
    """Test that config parameters override provider-specific env vars."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING", "http://embedding-env:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_EMBEDDING", "embedding-env-key")

    model = OpenAICompatibleEmbeddingModel(
        model_name="test-model",
        base_url="http://config:9090",
        api_key="config-key"
    )
    assert model.base_url == "http://config:9090"
    assert model.api_key == "config-key"


def test_direct_params_override_all_env_vars(monkeypatch):
    """Test that direct parameters override all environment variables."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING", "http://embedding-env:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic-env:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_EMBEDDING", "embedding-env-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-env-key")

    model = OpenAICompatibleEmbeddingModel(
        model_name="test-model",
        base_url="http://direct:3000",
        api_key="direct-key"
    )
    assert model.base_url == "http://direct:3000"
    assert model.api_key == "direct-key"


def test_error_message_mentions_both_env_vars(monkeypatch):
    """Test that error message mentions both provider-specific and generic env vars."""
    # Clear all environment variables
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)

    with pytest.raises(ValueError) as exc_info:
        OpenAICompatibleEmbeddingModel(model_name="test-model", api_key="test-key")

    error_message = str(exc_info.value)
    assert "OPENAI_COMPATIBLE_BASE_URL_EMBEDDING" in error_message
    assert "OPENAI_COMPATIBLE_BASE_URL" in error_message