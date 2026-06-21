"""Test cases for Cohere reranker provider."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types.reranker import RerankResponse
from esperanto.providers.reranker.cohere import CohereRerankerModel


def _mock_rerank_response():
    return {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 2, "relevance_score": 0.42},
            {"index": 1, "relevance_score": 0.10},
        ],
        "meta": {"billed_units": {"search_units": 1}},
    }


def _make_model(config=None):
    # Avoid creating real httpx clients we immediately discard for mocks.
    with patch.object(CohereRerankerModel, "_create_http_clients", lambda self: None):
        model = CohereRerankerModel(
            model_name="rerank-v4.0-pro", api_key="test-key", config=config or {}
        )
    mock_client = Mock()
    mock_async_client = AsyncMock()

    def post(url, **kwargs):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = _mock_rerank_response()
        return resp

    async def apost(url, **kwargs):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = _mock_rerank_response()
        return resp

    mock_client.post.side_effect = post
    mock_async_client.post.side_effect = apost
    model.client = mock_client
    model.async_client = mock_async_client
    return model


class TestCohereReranker:
    def test_initialization_with_api_key(self):
        reranker = CohereRerankerModel(
            model_name="rerank-v4.0-pro", api_key="test-api-key", config={}
        )
        assert reranker.api_key == "test-api-key"
        assert reranker.provider == "cohere"
        assert reranker.base_url == "https://api.cohere.com"

    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Cohere API key not found"):
                CohereRerankerModel(model_name="rerank-v4.0-pro", api_key=None, config={})

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {"COHERE_API_KEY": "env-api-key"}):
            reranker = CohereRerankerModel(model_name="rerank-v4.0-pro", api_key=None, config={})
            assert reranker.api_key == "env-api-key"

    def test_default_model(self):
        reranker = CohereRerankerModel(model_name=None, api_key="test-key", config={})
        assert reranker.get_model_name() == "rerank-v4.0-pro"

    def test_headers(self):
        reranker = CohereRerankerModel(model_name="rerank-v4.0-pro", api_key="secret", config={})
        headers = reranker._get_headers()
        assert headers["Authorization"] == "Bearer secret"
        assert headers["Content-Type"] == "application/json"

    def test_payload_uses_top_n(self):
        reranker = CohereRerankerModel(model_name="rerank-v4.0-pro", api_key="test-key", config={})
        payload = reranker._build_request_payload("q", ["a", "b"], 1)
        assert payload["query"] == "q"
        assert payload["documents"] == ["a", "b"]
        assert payload["model"] == "rerank-v4.0-pro"
        assert payload["top_n"] == 1
        assert "top_k" not in payload

    def test_validation_errors(self):
        reranker = CohereRerankerModel(model_name="rerank-v4.0-pro", api_key="test-key", config={})
        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("", ["doc1"])
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            reranker.rerank("query", [])
        with pytest.raises(ValueError, match="top_k must be positive"):
            reranker.rerank("query", ["doc1"], top_k=0)

    def test_rerank(self):
        reranker = _make_model()
        documents = ["best match", "worst match", "middle match"]
        result = reranker.rerank("query", documents)

        assert isinstance(result, RerankResponse)
        assert len(result.results) == 3
        # Documents are attached from the original list by index.
        assert result.results[0].document == "best match"
        assert result.results[1].document == "middle match"
        assert result.results[2].document == "worst match"
        # Verify the request hit the v2 rerank endpoint.
        url = reranker.client.post.call_args[0][0]
        assert url == "https://api.cohere.com/v2/rerank"

    def test_response_parsing(self):
        reranker = CohereRerankerModel(model_name="rerank-v4.0-pro", api_key="test-key", config={})
        result = reranker._parse_response(_mock_rerank_response(), ["d0", "d1", "d2"])
        assert isinstance(result, RerankResponse)
        assert result.results[0].index == 0
        assert result.results[1].index == 2

    @pytest.mark.asyncio
    async def test_arerank(self):
        reranker = _make_model()
        result = await reranker.arerank("query", ["best match", "worst match", "middle match"])
        assert isinstance(result, RerankResponse)
        assert len(result.results) == 3
        # Async parsing must map documents from the original list by index.
        assert result.results[0].document == "best match"
        assert result.results[1].document == "middle match"
        assert result.results[2].document == "worst match"
        reranker.async_client.post.assert_awaited_once()
