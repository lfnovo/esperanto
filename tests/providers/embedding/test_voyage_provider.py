"""Tests for Voyage AI embedding provider."""

import os
from unittest.mock import patch

import pytest
import responses

from esperanto.providers.embedding.voyage import VoyageEmbeddingModel


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
    assert model.get_model_name() == "voyage-large-2"


def test_provider_name():
    """Test getting provider name."""
    model = VoyageEmbeddingModel(api_key="test-key")
    assert model.provider == "voyage"


def test_models_list():
    """Test listing available models."""
    model = VoyageEmbeddingModel(api_key="test-key")
    models = model.models
    assert len(models) == 2
    assert models[0].id == "voyage-large-2"
    assert models[1].id == "voyage-code-2"


@responses.activate
def test_embed():
    """Test embedding creation."""
    # Mock response
    responses.add(
        responses.POST,
        "https://api.voyageai.com/v1/embeddings",
        json={
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        },
        status=200,
    )

    model = VoyageEmbeddingModel(api_key="test-key")
    texts = ["Hello", "World"]
    embeddings = model.embed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]


@responses.activate
def test_embed_error():
    """Test error handling in embedding creation."""
    # Mock error response
    responses.add(
        responses.POST,
        "https://api.voyageai.com/v1/embeddings",
        json={"error": {"message": "Invalid API key"}},
        status=401,
    )

    model = VoyageEmbeddingModel(api_key="invalid-key")
    with pytest.raises(RuntimeError, match="Voyage API error: Invalid API key"):
        model.embed(["test"])


@pytest.mark.asyncio
async def test_aembed():
    """Test async embedding creation."""
    with patch.object(VoyageEmbeddingModel, "embed") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        model = VoyageEmbeddingModel(api_key="test-key")
        embeddings = await model.aembed(["test"])

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with(texts=["test"])
