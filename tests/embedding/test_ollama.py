"""Tests for Ollama embedding model implementation."""

import os
from unittest.mock import patch

import pytest
import requests

from esperanto.providers.embedding.ollama import OllamaEmbeddingModel


@pytest.fixture
def model():
    """Create a test model instance."""
    with patch.dict(os.environ, {"OLLAMA_API_BASE": "http://test-api:11434"}):
        return OllamaEmbeddingModel(
            model_name="llama2",
            config={}
        )


class TestOllamaEmbeddingModel:
    """Test suite for Ollama embedding model."""

    async def test_initialization(self, model):
        """Test model initialization with valid config."""
        assert model.model_name == "llama2"
        assert model.config == {}
        assert model.base_url == "http://test-api:11434"

    async def test_initialization_default_values(self):
        """Test model initialization with default values."""
        with patch.dict(os.environ, {"OLLAMA_API_BASE": "http://localhost:11434"}, clear=True):
            model = OllamaEmbeddingModel(model_name="llama2")
            assert model.model_name == "llama2"
            assert model.config == {}
            assert model.base_url == "http://localhost:11434"

    async def test_initialization_with_config(self):
        """Test model initialization with config values."""
        model = OllamaEmbeddingModel(
            model_name="llama2",
            config={"base_url": "http://custom-api:11434"}
        )
        assert model.model_name == "llama2"
        assert model.config == {"base_url": "http://custom-api:11434"}
        assert model.base_url == "http://custom-api:11434"

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified for Ollama embedding model"):
            OllamaEmbeddingModel(model_name="")

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exceptions

    async def test_validate_config_failure_no_model_name(self):
        """Test config validation with no model name."""
        with pytest.raises(ValueError, match="model_name must be specified for Ollama embedding model"):
            OllamaEmbeddingModel(model_name="")

    async def test_validate_config_failure_no_base_url(self):
        """Test config validation with no base URL."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="base_url must be specified in config or OLLAMA_API_BASE environment variable"):
                OllamaEmbeddingModel(model_name="llama2", config={})

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "ollama"

    async def test_embed(self, model, mocker):
        """Test text embedding generation."""
        # Mock response data
        mock_embedding = [0.1, 0.2, 0.3]
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"embeddings": [mock_embedding]}

        # Mock requests.post
        mock_post = mocker.patch.object(
            requests, "post", return_value=mock_response
        )

        # Test single text
        texts = ["This is a test"]
        result = await model.embed(texts)

        # Verify results
        assert len(result) == 1
        assert result[0] == mock_embedding

        # Verify post was called with correct parameters
        mock_post.assert_called_once_with(
            "http://test-api:11434/api/embed",
            json={"model": "llama2", "input": ["This is a test"]}
        )

    async def test_embed_multiple_texts(self, model, mocker):
        """Test embedding generation for multiple texts."""
        # Mock response data
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_responses = []
        for embedding in mock_embeddings:
            mock_response = mocker.Mock()
            mock_response.json.return_value = {"embeddings": [embedding]}
            mock_responses.append(mock_response)

        # Mock requests.post
        mock_post = mocker.patch.object(
            requests, "post", side_effect=mock_responses
        )

        # Test multiple texts with newlines
        texts = [
            "First\ntext",
            "Second\ntext",
            "Third\ntext"
        ]
        result = await model.embed(texts)

        # Verify results
        assert len(result) == 3
        assert result == mock_embeddings

        # Verify post was called with correct parameters
        assert mock_post.call_count == 3
        mock_post.assert_has_calls([
            mocker.call(
                "http://test-api:11434/api/embed",
                json={"model": "llama2", "input": ["First text"]}
            ),
            mocker.call(
                "http://test-api:11434/api/embed",
                json={"model": "llama2", "input": ["Second text"]}
            ),
            mocker.call(
                "http://test-api:11434/api/embed",
                json={"model": "llama2", "input": ["Third text"]}
            )
        ])
