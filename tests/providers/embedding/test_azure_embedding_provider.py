"""Tests for Azure OpenAI embedding provider."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from esperanto.providers.embedding.azure import AzureEmbeddingModel

# Fixtures

@pytest.fixture
def azure_openai_model():
    """Create AzureEmbeddingModel with mocked HTTP client."""
    model = AzureEmbeddingModel(api_key="AZURE_OPENAI_API_KEY",
                                base_url="AZURE_OPENAI_ENDPOINT",
                                api_version="AZURE_OPENAI_API_VERSION",
                                model_name="text-embedding-3-large")

    # Mock response
    mock_response = MagicMock()
    mock_embedding_instance1 = MagicMock()
    mock_embedding_instance1.embedding = [0.1, 0.2, 0.3]
    mock_embedding_instance2 = MagicMock()
    mock_embedding_instance2.embedding = [0.4, 0.5, 0.6]
    mock_response.data = [mock_embedding_instance1, mock_embedding_instance2]

    # Mock the clients
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    mock_async_client = MagicMock()
    mock_async_client.embeddings.create = AsyncMock(return_value=mock_response)

    model.client = mock_client
    model.async_client = mock_async_client

    return model

# Tests


def test_init_with_parameters():
    """Test initialization with API key."""
    model = AzureEmbeddingModel(api_key="test-key", base_url="https://endpoint",
                                api_version="2023-05-15", model_name="my-deployment")
    assert model.api_key == "test-key"
    assert model.azure_endpoint == "https://endpoint"
    assert model.api_version == "2023-05-15"
    assert model.model_name == "my-deployment"


def test_init_with_env():
    """Test initialization with API key from environment."""
    env_vars = {
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_ENDPOINT": "https://env-endpoint",
        "AZURE_OPENAI_API_VERSION": "2023-01-01",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        model = AzureEmbeddingModel(model_name="env-deployment")
        assert model.api_key == "test-key"
        assert model.azure_endpoint == "https://env-endpoint"
        assert model.api_version == "2023-01-01"
        assert model.model_name == "env-deployment"


def test_init_without_api_key():
    """Test initialization without API key raises error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Azure OpenAI API key not found"):
            AzureEmbeddingModel(
                api_version="2023-01-01", base_url="https://endpoint", model_name="deployment")


def test_init_without_endpoint():
    """Test initialization without Azure endpoint raises error."""
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "key"}, clear=True):
        with pytest.raises(ValueError, match="Azure OpenAI endpoint not found"):
            AzureEmbeddingModel(api_version="2023-01-01",
                                model_name="deployment")


def test_init_without_api_version():
    """Test initialization without API version raises error."""
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "key", "AZURE_OPENAI_ENDPOINT": "https://endpoint"}, clear=True):
        with pytest.raises(ValueError, match="Azure OpenAI API version not found"):
            AzureEmbeddingModel(base_url="https://endpoint",
                                model_name="deployment")


def test_get_default_model():
    """Test getting default model name."""
    model = AzureEmbeddingModel(api_key="test-key", base_url="https://endpoint",
                                api_version="2023-05-15")
    assert model.get_model_name() == "text-embedding-3-small"


def test_provider_name():
    """Test getting provider name."""
    model = AzureEmbeddingModel(api_key="test-key", base_url="https://endpoint",
                                api_version="2023-05-15")
    assert model.provider == "azure"


def test_models_list():
    """Test listing available models."""
    model = AzureEmbeddingModel(api_key="test-key", base_url="https://endpoint",
                                api_version="2023-05-15")
    models = model.models
    assert len(models) == 0  # No ability to pull this info


def test_embed(azure_openai_model):
    """Test embedding creation."""
    texts = ["Hello", "World"]
    embeddings = azure_openai_model.embed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    # Verify client was called
    azure_openai_model.client.embeddings.create.assert_called_once()
    call_args = azure_openai_model.client.embeddings.create.call_args

    _, kwargs = call_args

    # Check request
    assert kwargs['input'] == texts
    assert kwargs['model'] == "text-embedding-3-large"


@pytest.mark.asyncio
async def test_aembed(azure_openai_model):
    """Test async embedding creation."""
    texts = ["Hello"]
    embeddings = await azure_openai_model.aembed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    # Verify client was called
    azure_openai_model.async_client.embeddings.create.assert_called_once()
    call_args = azure_openai_model.async_client.embeddings.create.call_args

    _, kwargs = call_args

    # Check request
    assert kwargs['input'] == texts
    assert kwargs['model'] == "text-embedding-3-large"


def test_text_cleaning(azure_openai_model):
    """Test that newlines in texts are replaced with spaces."""
    texts = [
    "Hello, world!",                    # Normal case, no change
    "Hello    world!",                  # Multiple spaces between words
    "Hello , world.",                   # Space before punctuation
    "Wait.. what?",                    # Repeated punctuation
    "This is a test.\nNew line here.", # Newlines in text
    "Multiple\n\nnew\nlines...",       # Multiple newlines and ellipsis
    "Spaces before . commas , and dots . ",  # Spaces before punctuation
    " Leading and trailing spaces ",   # Leading/trailing spaces
]
    azure_openai_model.embed(texts)

    # Check that the input was cleaned
    call_args = azure_openai_model.client.embeddings.create.call_args
    _, kwargs = call_args
    assert kwargs['input'] == [
                                    "Hello, world!",
                                    "Hello world!",
                                    "Hello, world.",
                                    "Wait. what?",
                                    "This is a test. New line here.",
                                    "Multiple new lines.",
                                    "Spaces before. commas, and dots.",
                                    "Leading and trailing spaces",
                                ]
