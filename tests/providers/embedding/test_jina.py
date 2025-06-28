"""Tests for Jina embedding provider."""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from esperanto.common_types.task_type import EmbeddingTaskType
from esperanto.providers.embedding.jina import JinaEmbeddingModel


class TestJinaEmbeddingModel:
    """Test suite for JinaEmbeddingModel."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        model = JinaEmbeddingModel(api_key="test-key")
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.jina.ai/v1/embeddings"
        assert model.provider == "jina"

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"JINA_API_KEY": "env-key"}):
            model = JinaEmbeddingModel()
            assert model.api_key == "env-key"

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Jina API key not found"):
                JinaEmbeddingModel()

    def test_default_model(self):
        """Test default model selection."""
        model = JinaEmbeddingModel(api_key="test-key")
        assert model.get_model_name() == "jina-embeddings-v3"

    def test_custom_model(self):
        """Test custom model selection."""
        model = JinaEmbeddingModel(
            api_key="test-key",
            model_name="jina-embeddings-v4"
        )
        assert model.get_model_name() == "jina-embeddings-v4"

    def test_task_type_mapping(self):
        """Test task type enum to API value mapping."""
        model = JinaEmbeddingModel(
            api_key="test-key",
            config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
        )
        assert model.task_type == EmbeddingTaskType.RETRIEVAL_QUERY
        assert model._map_task_type() == "retrieval.query"

    def test_task_type_string_conversion(self):
        """Test task type string is converted to enum."""
        model = JinaEmbeddingModel(
            api_key="test-key",
            config={"task_type": "retrieval.query"}
        )
        assert model.task_type == EmbeddingTaskType.RETRIEVAL_QUERY

    def test_build_request_payload_basic(self):
        """Test building basic request payload."""
        model = JinaEmbeddingModel(api_key="test-key")
        payload = model._build_request_payload(["Hello world"])
        
        assert payload == {
            "model": "jina-embeddings-v3",
            "input": [{"text": "Hello world"}],
            "truncate": True
        }

    def test_build_request_payload_with_task_type(self):
        """Test building request payload with task type."""
        model = JinaEmbeddingModel(
            api_key="test-key",
            config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
        )
        payload = model._build_request_payload(["Query text"])
        
        assert payload["task"] == "retrieval.query"

    def test_build_request_payload_with_all_features(self):
        """Test building request payload with all features."""
        model = JinaEmbeddingModel(
            api_key="test-key",
            model_name="jina-embeddings-v4",
            config={
                "task_type": EmbeddingTaskType.CLASSIFICATION,
                "late_chunking": True,
                "output_dimensions": 512,
                "truncate_at_max_length": False
            }
        )
        payload = model._build_request_payload(["Classify this"])
        
        assert payload == {
            "model": "jina-embeddings-v4",
            "input": [{"text": "Classify this"}],
            "task": "classification",
            "late_chunking": True,
            "dimensions": 512
        }

    def test_embed_success(self):
        """Test successful embedding generation."""
        model = JinaEmbeddingModel(api_key="test-key")
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }
        
        with patch.object(model.client, "post", return_value=mock_response):
            embeddings = model.embed(["text1", "text2"])
            
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_embed_empty_texts(self):
        """Test embedding with empty text list."""
        model = JinaEmbeddingModel(api_key="test-key")
        embeddings = model.embed([])
        assert embeddings == []

    def test_embed_error_handling(self):
        """Test error handling in embed method."""
        model = JinaEmbeddingModel(api_key="test-key")
        
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "type": "invalid_request",
                "message": "Invalid input"
            }
        }
        
        with patch.object(model.client, "post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Jina API error \\(invalid_request\\): Invalid input"):
                model.embed(["test"])

    def test_embed_timeout(self):
        """Test timeout handling."""
        model = JinaEmbeddingModel(api_key="test-key")
        
        with patch.object(model.client, "post", side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(RuntimeError, match="Request to Jina API timed out"):
                model.embed(["test"])

    def test_embed_network_error(self):
        """Test network error handling."""
        model = JinaEmbeddingModel(api_key="test-key")
        
        with patch.object(model.client, "post", side_effect=httpx.RequestError("Network error")):
            with pytest.raises(RuntimeError, match="Network error calling Jina API"):
                model.embed(["test"])

    @pytest.mark.asyncio
    async def test_aembed_success(self):
        """Test successful async embedding generation."""
        model = JinaEmbeddingModel(api_key="test-key")
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]}
            ]
        }
        
        with patch.object(model.async_client, "post", return_value=mock_response):
            embeddings = await model.aembed(["test"])
            
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

    def test_models_property(self):
        """Test models property returns correct list."""
        model = JinaEmbeddingModel(api_key="test-key")
        models = model.models
        
        assert len(models) > 0
        assert any(m.id == "jina-embeddings-v4" for m in models)
        assert any(m.id == "jina-embeddings-v3" for m in models)
        assert all(m.type == "embedding" for m in models)

    def test_text_cleaning(self):
        """Test text cleaning is applied."""
        model = JinaEmbeddingModel(api_key="test-key")
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1]}]}
        
        with patch.object(model.client, "post", return_value=mock_response) as mock_post:
            model.embed(["Hello  world\n\nTest"])
            
            # Check that text was cleaned
            call_args = mock_post.call_args[1]["json"]
            assert call_args["input"][0]["text"] == "Hello world Test"

    def test_native_task_optimization(self):
        """Test that Jina doesn't apply task prefixes (handles natively)."""
        model = JinaEmbeddingModel(
            api_key="test-key",
            config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
        )
        
        # The base class method should return unchanged texts
        texts = ["search query"]
        optimized = model._apply_task_optimization(texts)
        assert optimized == texts  # No prefix added

    def test_native_late_chunking(self):
        """Test that Jina doesn't apply chunking (handles natively)."""
        model = JinaEmbeddingModel(
            api_key="test-key",
            config={"late_chunking": True}
        )
        
        # The method should return unchanged texts
        texts = ["very long text " * 100]
        chunked = model._apply_late_chunking(texts)
        assert chunked == texts  # No chunking applied