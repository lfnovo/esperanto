import os
from unittest.mock import AsyncMock, Mock, patch
import pytest
from esperanto.providers.embedding.mistral import MistralEmbeddingModel

@pytest.fixture
def mock_mistral_embedding_client():
    client = Mock()
    async_client = AsyncMock()
    # Mock embeddings.create to return [] if input is empty, else [[0.1, 0.2, 0.3]]
    def create_side_effect(model, inputs, **kwargs):
        if not inputs:
            return Mock(data=[])
        embedding_data = Mock()
        embedding_data.embedding = [0.1, 0.2, 0.3]
        return Mock(data=[embedding_data])
    async def create_async_side_effect(model, inputs, **kwargs):
        if not inputs:
            return Mock(data=[])
        embedding_data = Mock()
        embedding_data.embedding = [0.1, 0.2, 0.3]
        return Mock(data=[embedding_data])
    client.embeddings.create.side_effect = create_side_effect
    async_client.embeddings.create_async.side_effect = create_async_side_effect
    return client, async_client

@pytest.fixture
def mistral_embedding_model(monkeypatch, mock_mistral_embedding_client):
    client, async_client = mock_mistral_embedding_client
    with patch("esperanto.providers.embedding.mistral.Mistral", return_value=client):
        model = MistralEmbeddingModel(api_key="test-key", model_name="mistral-embed")
        model.client = client
        model.async_client = async_client
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

def test_aembed(mistral_embedding_model):
    import asyncio
    texts = ["Hello world"]
    async def run():
        result = await mistral_embedding_model.aembed(texts)
        assert result == [[0.1, 0.2, 0.3]]
    asyncio.run(run())

def test_embed_empty(mistral_embedding_model):
    texts = []
    result = mistral_embedding_model.embed(texts)
    assert result == []

def test_get_default_model():
    model = MistralEmbeddingModel(api_key="test-key")
    assert model._get_default_model() == "mistral-embed"
