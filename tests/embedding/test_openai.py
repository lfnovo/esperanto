"""Tests for OpenAI embedding model implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.embedding import Embedding

from esperanto.providers.embedding.openai import OpenAIEmbeddingModel


@pytest.fixture
def model():
    """Create a test model instance."""
    return OpenAIEmbeddingModel(model_name="text-embedding-3-small")


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = Mock(spec=AsyncOpenAI)
    mock_client.embeddings = Mock()
    mock_client.embeddings.create = AsyncMock()
    return mock_client


class TestOpenAIEmbeddingModel:
    """Test suite for OpenAI embedding model."""

    async def test_initialization(self):
        """Test model initialization with valid config."""
        model = OpenAIEmbeddingModel(model_name="text-embedding-3-small")
        assert model.model_name == "text-embedding-3-small"
        assert model.config == {}
        assert model._client is None

    async def test_initialization_with_config(self):
        """Test model initialization with config values."""
        config = {"api_key": "test-key"}
        model = OpenAIEmbeddingModel(model_name="text-embedding-3-small", config=config)
        assert model.model_name == "text-embedding-3-small"
        assert model.config == config
        assert model._client is None

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(
            ValueError, match="model_name must be specified for OpenAI embedding model"
        ):
            OpenAIEmbeddingModel(model_name="")

    async def test_client_lazy_initialization(self):
        """Test lazy initialization of OpenAI client."""
        mock_client = Mock(spec=AsyncOpenAI)
        with patch(
            "esperanto.providers.embedding.openai.AsyncOpenAI", return_value=mock_client
        ) as mock_openai:
            model = OpenAIEmbeddingModel(model_name="text-embedding-3-small")

            # First access should create client
            assert model.client is not None
            mock_openai.assert_called_once()

            # Second access should use cached client
            cached_client = model.client
            assert model.client is cached_client
            mock_openai.assert_called_once()

    async def test_client_setter(self):
        """Test setting the OpenAI client."""
        model = OpenAIEmbeddingModel(model_name="text-embedding-3-small")
        mock_client = Mock(spec=AsyncOpenAI)
        model.client = mock_client
        assert model.client is mock_client

    async def test_client_deleter(self):
        """Test deleting the OpenAI client."""
        model = OpenAIEmbeddingModel(model_name="text-embedding-3-small")
        mock_client = Mock(spec=AsyncOpenAI)
        model.client = mock_client
        del model.client
        assert model._client is None

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "openai"

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exceptions

    async def test_validate_config_failure_no_model_name(self):
        """Test config validation with no model name."""
        with pytest.raises(
            ValueError, match="model_name must be specified for OpenAI embedding model"
        ):
            OpenAIEmbeddingModel(model_name="")

    async def test_embed(self, model, mock_openai_client):
        """Test text embedding generation."""
        # Mock response data
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_response = CreateEmbeddingResponse(
            data=[
                Embedding(embedding=embedding, index=i, object="embedding")
                for i, embedding in enumerate(mock_embeddings)
            ],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )

        # Set up mock client
        mock_openai_client.embeddings.create.return_value = mock_response
        model.client = mock_openai_client

        # Test multiple texts with newlines
        texts = ["First\ntext", "Second\ntext"]
        result = await model.embed(texts)

        # Verify results
        assert len(result) == 2
        assert result == mock_embeddings

        # Verify client was called with correct parameters
        mock_openai_client.embeddings.create.assert_called_once_with(
            input=["First text", "Second text"], model="text-embedding-3-small"
        )
