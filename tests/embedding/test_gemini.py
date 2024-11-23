"""Tests for Gemini embedding model implementation."""

from unittest.mock import Mock, patch

import pytest

from esperanto.providers.embedding.gemini import GeminiEmbeddingModel


@pytest.fixture
def model():
    """Create a test model instance."""
    return GeminiEmbeddingModel(model_name="embedding-001")


class TestGeminiEmbeddingModel:
    """Test suite for Gemini embedding model."""

    async def test_initialization(self):
        """Test model initialization with valid config."""
        model = GeminiEmbeddingModel(model_name="embedding-001")
        assert model.model_name == "embedding-001"
        assert model.config == {}

    async def test_initialization_with_config(self):
        """Test model initialization with config values."""
        config = {"api_key": "test-key"}
        model = GeminiEmbeddingModel(
            model_name="embedding-001",
            config=config
        )
        assert model.model_name == "embedding-001"
        assert model.config == config

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified for Gemini embedding model"):
            GeminiEmbeddingModel(model_name="")

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "gemini"

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exceptions

    async def test_validate_config_failure_no_model_name(self):
        """Test config validation with no model name."""
        with pytest.raises(ValueError, match="model_name must be specified for Gemini embedding model"):
            GeminiEmbeddingModel(model_name="")

    async def test_embed_with_model_prefix(self, model):
        """Test embedding generation with model prefix."""
        # Mock response data
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_responses = [
            {"embedding": embedding}
            for embedding in mock_embeddings
        ]

        # Mock genai.embed_content
        with patch("google.generativeai.embed_content") as mock_embed:
            mock_embed.side_effect = mock_responses

            # Test multiple texts
            texts = ["First text", "Second text"]
            result = await model.embed(texts)

            # Verify results
            assert len(result) == 2
            assert result == mock_embeddings

            # Verify embed_content was called with correct parameters
            assert mock_embed.call_count == 2
            mock_embed.assert_any_call(
                model="models/embedding-001",
                content="First text"
            )
            mock_embed.assert_any_call(
                model="models/embedding-001",
                content="Second text"
            )

    async def test_embed_without_model_prefix(self):
        """Test embedding generation without model prefix."""
        model = GeminiEmbeddingModel(model_name="models/embedding-001")
        mock_embedding = [0.1, 0.2, 0.3]
        mock_response = {"embedding": mock_embedding}

        # Mock genai.embed_content
        with patch("google.generativeai.embed_content") as mock_embed:
            mock_embed.return_value = mock_response

            # Test single text
            texts = ["Test text"]
            result = await model.embed(texts)

            # Verify results
            assert len(result) == 1
            assert result[0] == mock_embedding

            # Verify embed_content was called with correct parameters
            mock_embed.assert_called_once_with(
                model="models/embedding-001",
                content="Test text"
            )

    async def test_embed_with_kwargs(self, model):
        """Test embedding generation with additional kwargs."""
        # Mock response data
        mock_embedding = [0.1, 0.2, 0.3]
        mock_response = {"embedding": mock_embedding}

        # Mock genai.embed_content
        with patch("google.generativeai.embed_content") as mock_embed:
            mock_embed.return_value = mock_response

            # Test with additional kwargs
            texts = ["Test text"]
            result = await model.embed(texts, task_type="retrieval", title="Test")

            # Verify results
            assert len(result) == 1
            assert result[0] == mock_embedding

            # Verify embed_content was called with correct parameters
            mock_embed.assert_called_once_with(
                model="models/embedding-001",
                content="Test text",
                task_type="retrieval",
                title="Test"
            )
