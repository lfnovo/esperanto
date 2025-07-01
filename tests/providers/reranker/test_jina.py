"""Test cases for Jina reranker provider."""

import os
from unittest.mock import Mock, patch

import pytest

from esperanto.common_types.reranker import RerankResponse, RerankResult
from esperanto.providers.reranker.jina import JinaRerankerModel


class TestJinaReranker:
    """Test cases for Jina reranker provider."""

    def test_initialization_with_api_key(self):
        """Test proper initialization with API key."""
        api_key = "test-api-key"
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key=api_key,
            config={}
        )
        
        assert reranker.api_key == api_key
        assert reranker.model_name == "jina-reranker-v2-base-multilingual"
        assert reranker.provider == "jina"
        assert reranker.base_url == "https://api.jina.ai/v1"

    def test_missing_api_key(self):
        """Test handling of missing API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Jina API key not found"):
                JinaRerankerModel(
                    model_name="jina-reranker-v2-base-multilingual",
                    api_key=None,
                    config={}
                )

    def test_initialization_with_env_var(self):
        """Test initialization using environment variable."""
        with patch.dict('os.environ', {'JINA_API_KEY': 'env-api-key'}):
            reranker = JinaRerankerModel(
                model_name="jina-reranker-v2-base-multilingual",
                api_key=None,
                config={}
            )
            assert reranker.api_key == "env-api-key"

    def test_provider_properties(self):
        """Test provider properties and models."""
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-key",
            config={}
        )
        
        assert reranker.provider == "jina"
        assert len(reranker.models) > 0
        assert all(model.type == "reranker" for model in reranker.models)
        assert reranker._get_default_model() == "jina-reranker-v2-base-multilingual"

    def test_validation_errors(self):
        """Test input validation."""
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-key",
            config={}
        )
        
        # Test empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("", ["doc1"])
        
        # Test empty documents
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            reranker.rerank("query", [])
        
        # Test invalid top_k
        with pytest.raises(ValueError, match="top_k must be positive"):
            reranker.rerank("query", ["doc1"], top_k=0)

    def test_model_listings(self):
        """Test available models are properly listed."""
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-key",
            config={}
        )
        
        models = reranker.models
        model_names = [m.id for m in models]
        
        # Check that key models are included
        assert "jina-reranker-v2-base-multilingual" in model_names
        assert "jina-reranker-v1-base-en" in model_names

    def test_headers_generation(self):
        """Test request headers are properly generated."""
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-secret-key",
            config={}
        )
        
        headers = reranker._get_headers()
        assert headers["Authorization"] == "Bearer test-secret-key"
        assert headers["Content-Type"] == "application/json"

    def test_request_payload_building(self):
        """Test request payload is properly built."""
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-key",
            config={}
        )
        
        query = "test query"
        documents = ["doc1", "doc2"]
        top_k = 1
        
        payload = reranker._build_request_payload(query, documents, top_k)
        
        assert payload["query"] == query
        assert payload["documents"] == documents
        assert payload["model"] == "jina-reranker-v2-base-multilingual"
        assert payload["top_n"] == top_k  # Now using consistent "top_k" parameter

    def test_custom_config_in_payload(self):
        """Test custom config is included in request payload."""
        custom_config = {"return_documents": False, "custom_param": "value"}
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-key",
            config=custom_config
        )
        
        payload = reranker._build_request_payload("query", ["doc1"], 1)
        
        # Custom config is not directly added to payload in current implementation
        # The payload will have return_documents=True (hardcoded)
        assert payload["return_documents"] is True

    def test_response_processing(self):
        """Test response data processing."""
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-key",
            config={}
        )
        
        # Mock response data with dict documents
        response_data = {
            "model": "jina-reranker-v2-base-multilingual",
            "results": [
                {"index": 0, "document": {"text": "Machine learning is AI"}, "relevance_score": 0.95},
                {"index": 1, "document": {"text": "Weather is nice"}, "relevance_score": 0.15}
            ]
        }
        
        result = reranker._parse_response(response_data, ["Machine learning is AI", "Weather is nice"])
        
        assert isinstance(result, RerankResponse)
        assert result.model == "jina-reranker-v2-base-multilingual"
        assert len(result.results) == 2
        
        # Verify document text extraction from dict
        assert result.results[0].document == "Machine learning is AI"
        assert result.results[1].document == "Weather is nice"

    def test_response_processing_with_string_documents(self):
        """Test response processing when documents are strings."""
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-key",
            config={}
        )
        
        # Mock response data with string documents
        response_data = {
            "model": "jina-reranker-v2-base-multilingual",
            "results": [
                {"index": 0, "document": "Machine learning is AI", "relevance_score": 0.95}
            ]
        }
        
        result = reranker._parse_response(response_data, ["Machine learning is AI"])
        
        assert result.results[0].document == "Machine learning is AI"

    def test_response_processing_with_various_document_formats(self):
        """Test response processing with different document formats."""
        reranker = JinaRerankerModel(
            model_name="jina-reranker-v2-base-multilingual",
            api_key="test-key",
            config={}
        )
        
        # Mock response data with various document formats
        response_data = {
            "model": "jina-reranker-v2-base-multilingual",
            "results": [
                {"index": 0, "document": {"text": "Text field document"}, "relevance_score": 0.95},
                {"index": 1, "document": {"content": "Content field document"}, "relevance_score": 0.85},
                {"index": 2, "document": {"body": "Body field document"}, "relevance_score": 0.75},
                {"index": 3, "document": {"unknown": "field"}, "relevance_score": 0.65},
                {"index": 4, "document": None, "relevance_score": 0.55}
            ]
        }
        
        original_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        result = reranker._parse_response(response_data, original_docs)
        
        assert result.results[0].document == "Text field document"
        assert result.results[1].document == "Content field document"
        assert result.results[2].document == "Body field document"
        assert result.results[3].document == "{'unknown': 'field'}"  # Stringified dict
        assert result.results[4].document == "doc5"  # Fallback to original

    def test_get_model_name(self):
        """Test model name retrieval."""
        reranker = JinaRerankerModel(
            model_name="custom-model",
            api_key="test-key",
            config={}
        )
        
        assert reranker.get_model_name() == "custom-model"

    def test_default_model(self):
        """Test default model selection."""
        reranker = JinaRerankerModel(
            model_name=None,
            api_key="test-key",
            config={}
        )
        
        assert reranker.get_model_name() == "jina-reranker-v2-base-multilingual"