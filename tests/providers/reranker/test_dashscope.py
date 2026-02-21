"""Test cases for DashScope reranker provider."""

import os
from unittest.mock import patch

import pytest

from esperanto.common_types.reranker import RerankResponse
from esperanto.providers.reranker.dashscope import DashScopeRerankerModel


class TestDashScopeReranker:
    """Test cases for DashScope reranker provider."""

    def test_initialization_with_api_key(self):
        """Test proper intialization with API key."""
        api_key = "test-api-key"
        reranker = DashScopeRerankerModel(
            model_name="gte-rerank-v2",
            api_key=api_key,
            config={}
        )

        assert reranker.api_key == api_key
        assert reranker.model_name == "gte-rerank-v2"
        assert reranker.provider == "dashscope"
        assert reranker.base_url == "https://dashscope.aliyuncs.com/api/v1"

    def test_missing_api_key(self):
        """Test handling of missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DashScope API key not found"):
                DashScopeRerankerModel(
                    model_name="qwen3-rerank",
                    api_key=None,
                    config={}
                )

    def test_initialization_with_env_var(self):
        """Test initialization using environment variable."""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": 'env-api-key'}):
            reranker = DashScopeRerankerModel(
                model_name="qwen3-rerank",
                api_key=None,
                config={}
            )
            assert reranker.api_key == "env-api-key"

    def test_provider_properties(self):
        """Test provider properties and models."""
        reranker = DashScopeRerankerModel(
            model_name="qwen3-rerank",
            api_key="test-key",
            config={}
        )

        assert reranker.provider == "dashscope"
        assert len(reranker.models) > 0
        # Model type is None when not explicitly provided by the API
        assert all(model.type is None for model in reranker.models)
        assert reranker._get_default_model() == "qwen3-rerank"

    def test_validation_errors(self):
        """Test input validation.
        DashScopeRerankerModel includes vl models, thus containing external
        validation logic.
        """
        reranker = DashScopeRerankerModel(
            model_name="qwen3-rerank",
            api_key="test-api-key",
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

        # Test input format for non-vl models
        with pytest.raises(
            ValueError, match="inputs must be formatted as list of strings"
        ):
            reranker.rerank("test-query", [{"text": "test-doc"}])

        reranker_vl = DashScopeRerankerModel(
            model_name="qwen3-vl-rerank",
            api_key="test-api-key",
            config={}
        )
        # Test input format for vl models
        with pytest.raises(
            ValueError, match="inputs must be formatted as list of dict"
        ):
            reranker_vl.rerank("test-query", ["doc1", "doc2"])

        # Check image input format for vl models
        # Invalid URL
        with pytest.raises(
            ValueError,
            match="only accepts image as either public URL or base64 data"
        ):
            reranker_vl.rerank(
                "test-query",
                [
                    {"image": "ftp://test/image.png"}
                ]
            )

        # Invalid base64 encoding
        with pytest.raises(
            ValueError,
            match="only accepts image as either public URL or base64 data"
        ):
            reranker_vl.rerank(
                "test-query",
                [
                    {"image": "data:video/JPEG;base64,122987974"}
                ]
            )

        # Correct base64 format but invalid image format
        with pytest.raises(
            ValueError,
            match="Unsupported image format: gif"
        ):
            reranker_vl.rerank(
                "test-query",
                [
                    {"image": "data:image/gif;base64,correct/base64/format==="}
                ]
            )

        # Check video input format for vl models
        with pytest.raises(
            ValueError,
            match="only accepts video as public URL"
        ):
            reranker_vl.rerank(
                "test-query",
                [
                    {"video": "ftp://test/video.mp4"}
                ]
            )

    def test_model_listings(self):
        """Test available models are properly listed."""
        reranker = DashScopeRerankerModel(
            model_name="qwen3-rerank",
            api_key="test-api-key",
            config={}
        )

        models = reranker.models
        model_names = [mod.id for mod in models]

        assert "qwen3-rerank" in model_names
        assert "gte-rerank-v2" in model_names
        assert "qwen3-vl-rerank" in model_names

    def test_headers_generation(self):
        """Test request headers are properly generated."""
        reranker = DashScopeRerankerModel(
            model_name="gte-rerank-v2",
            api_key="test-secret-key",
            config={}
        )

        headers = reranker._get_headers()
        assert headers["Authorization"] == "Bearer test-secret-key"
        assert headers["Content-Type"] == "application/json"

    def test_non_vl_request_payload_building(self):
        query = "test query"
        documents = ["doc1", "doc2"]
        top_k = 1
        instruct = "test instruct"

        reranker = DashScopeRerankerModel(
            model_name="gte-rerank-v2",
            api_key="test-key",
            config={}
        )

        payload = reranker._build_request_payload(
            query, documents, top_k, instruct
        )

        # Check nested structure
        assert "input" in payload
        assert "parameters" in payload

        assert payload["input"]["query"] == query
        assert payload["input"]["documents"] == documents
        assert payload["model"] == "gte-rerank-v2"
        assert payload["parameters"]["top_n"] == top_k

        # Check ignored instruct for gte
        assert "instruct" not in payload["parameters"]

        reranker = DashScopeRerankerModel(
            model_name="qwen3-rerank",
            api_key="test-key",
            config={}
        )

        payload = reranker._build_request_payload(
            query, documents, top_k, instruct
        )

        # Check nested structure
        assert "input" in payload
        assert "parameters" in payload

        assert payload["input"]["query"] == query
        assert payload["input"]["documents"] == documents
        assert payload["model"] == "qwen3-rerank"
        assert payload["parameters"]["top_n"] == top_k

        # Check impl instruct for qwen rerankers
        assert payload["parameters"]["instruct"] == instruct

    def test_vl_request_payload_building(self):
        query = "test query"
        documents = [
            {"text": "doc"},
            {"image": "https://test/image1.png"},
            {"video": "https://test/video1.mp4"}
        ]
        top_k = 2
        instruct = 'test instruct'

        reranker = DashScopeRerankerModel(
            model_name="qwen3-vl-rerank",
            api_key="test-key",
            config={}
        )

        vl_payload = reranker._build_request_payload(
            query, documents, top_k, instruct
        )

        # Check nested structure
        assert "input" in vl_payload
        assert "parameters" in vl_payload

        assert vl_payload["input"]["query"] == query
        assert vl_payload["input"]["documents"] == documents
        assert vl_payload["model"] == "qwen3-vl-rerank"
        assert vl_payload["parameters"]["top_n"] == top_k

        # Check impl instruct for qwen rerankers
        assert vl_payload["parameters"]["instruct"] == instruct

    def test_custom_config_in_payload(self):
        """Test custom config is included in request payload."""
        custom_config = {"return_documents": True, "custom_param": "value"}
        reranker = DashScopeRerankerModel(
            model_name="gte-rerank-v2",
            api_key="test-key",
            config=custom_config
        )

        payload = reranker._build_request_payload("query", ["doc1"], 1)
        # Hardcoded because some dashscope models don't support 'return_documents'
        assert payload["parameters"]["return_documents"] is False
        # In current implementation custom params not directly added.

    def test_response_processing_with_no_documents_returned(self):
        """Test response data processing (with no documents returned)."""
        reranker = DashScopeRerankerModel(
            model_name="qwen3-rerank",
            api_key="test-key",
            config={}
        )

        documents = ["Machine learning is AI", "Weather is nice", "Weather is bad"]
        raw_results = [
            {"index": 0, "relevance_score": 0.50},
            {"index": 2, "relevance_score": 0.15},
            {"index": 1, "relevance_score": 0.00}
        ]

        raw_response = {
            "output": {
                "results": raw_results
            },
            "usage": {
                "total_tokens": 100
            },
            "request_id": "test-request-id"
        }

        result = reranker._parse_response(raw_response, documents)
        assert isinstance(result, RerankResponse)
        assert result.model == "qwen3-rerank"
        assert len(result.results) == 3
        assert result.results[0].document == "Machine learning is AI"
        assert result.results[1].document == "Weather is bad"
        assert result.results[2].document == "Weather is nice"
        assert result.results[0].relevance_score == 1.00
        assert result.results[1].relevance_score == 0.30
        assert result.results[2].relevance_score == 0.00

    def test_get_model_name(self):
        """Test model name retrieval."""
        reranker = DashScopeRerankerModel(
            model_name="custom-model",
            api_key="test-key",
            config={}
        )

        assert reranker.get_model_name() == "custom-model"

    def test_default_model(self):
        """Test default model selection."""
        reranker = DashScopeRerankerModel(
            model_name=None,
            api_key="test-key",
            config={}
        )

        assert reranker.get_model_name() == "qwen3-rerank"
