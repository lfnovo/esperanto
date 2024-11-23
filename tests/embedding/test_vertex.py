"""Tests for Vertex AI embedding model implementation."""

from unittest.mock import Mock, patch

import pytest
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from esperanto.providers.embedding.vertex import VertexEmbeddingModel


@pytest.fixture
def mock_embedding_values():
    """Create mock embedding values."""
    return [0.1, 0.2, 0.3]


@pytest.fixture
def mock_embedding(mock_embedding_values):
    """Create mock embedding response."""
    mock_emb = Mock()
    mock_emb.values = mock_embedding_values
    return mock_emb


@pytest.fixture
def mock_model(mock_embedding):
    """Create mock Vertex AI model."""
    mock_mdl = Mock(spec=TextEmbeddingModel)
    mock_mdl.get_embeddings.return_value = [mock_embedding]
    return mock_mdl


@pytest.fixture
def model():
    """Create a test model instance."""
    return VertexEmbeddingModel(model_name="text-embedding-model")


class TestVertexEmbeddingModel:
    """Test suite for Vertex AI embedding model."""

    async def test_initialization(self):
        """Test model initialization with valid config."""
        model = VertexEmbeddingModel(model_name="text-embedding-model")
        assert model.model_name == "text-embedding-model"
        assert model.config == {}
        assert model._model is None

    async def test_initialization_with_config(self):
        """Test model initialization with config values."""
        config = {"project": "test-project"}
        model = VertexEmbeddingModel(
            model_name="text-embedding-model",
            config=config
        )
        assert model.model_name == "text-embedding-model"
        assert model.config == config
        assert model._model is None

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified for Vertex AI embedding model"):
            VertexEmbeddingModel(model_name="")

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "vertex"

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exceptions

    async def test_validate_config_failure_no_model_name(self):
        """Test config validation with no model name."""
        with pytest.raises(ValueError, match="model_name must be specified for Vertex AI embedding model"):
            VertexEmbeddingModel(model_name="")

    async def test_model_lazy_initialization(self, model):
        """Test lazy initialization of Vertex AI model."""
        mock_model = Mock(spec=TextEmbeddingModel)
        
        with patch.object(TextEmbeddingModel, "from_pretrained", return_value=mock_model) as mock_from_pretrained:
            # First access should initialize the model
            assert model.model == mock_model
            mock_from_pretrained.assert_called_once_with("text-embedding-model")

            # Second access should use cached model
            assert model.model == mock_model
            mock_from_pretrained.assert_called_once()

    async def test_embed_single_text(self, model, mock_model, mock_embedding, mock_embedding_values):
        """Test embedding generation for single text."""
        with patch.object(TextEmbeddingModel, "from_pretrained", return_value=mock_model):
            texts = ["Test text"]
            result = await model.embed(texts)

            # Verify results
            assert len(result) == 1
            assert result[0] == mock_embedding_values

            # Verify model calls
            mock_model.get_embeddings.assert_called_once()
            call_args = mock_model.get_embeddings.call_args[0][0]
            assert len(call_args) == 1
            assert isinstance(call_args[0], TextEmbeddingInput)
            assert call_args[0].text == "Test text"

    async def test_embed_multiple_texts(self, model):
        """Test embedding generation for multiple texts."""
        # Create mock embeddings for multiple texts
        mock_embeddings = [
            Mock(values=[0.1, 0.2]),
            Mock(values=[0.3, 0.4]),
            Mock(values=[0.5, 0.6])
        ]
        mock_model = Mock(spec=TextEmbeddingModel)
        mock_model.get_embeddings.return_value = mock_embeddings

        with patch.object(TextEmbeddingModel, "from_pretrained", return_value=mock_model):
            texts = ["First text", "Second text", "Third text"]
            result = await model.embed(texts)

            # Verify results
            assert len(result) == 3
            assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

            # Verify model calls
            mock_model.get_embeddings.assert_called_once()
            call_args = mock_model.get_embeddings.call_args[0][0]
            assert len(call_args) == 3
            assert all(isinstance(arg, TextEmbeddingInput) for arg in call_args)
            assert [arg.text for arg in call_args] == texts

    async def test_embed_with_kwargs(self, model, mock_model):
        """Test embedding generation with additional kwargs."""
        with patch.object(TextEmbeddingModel, "from_pretrained", return_value=mock_model):
            texts = ["Test text"]
            await model.embed(texts, task_type="retrieval", title="Test")

            # Verify model calls with kwargs
            mock_model.get_embeddings.assert_called_once()
            call_args = mock_model.get_embeddings.call_args[0][0]
            assert len(call_args) == 1
            assert isinstance(call_args[0], TextEmbeddingInput)
            assert call_args[0].text == "Test text"
