"""Test cases for Cohere embedding provider."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.providers.embedding.cohere import CohereEmbeddingModel


def _embed_response(vectors):
    return {
        "embeddings": {"float": vectors},
        "meta": {"billed_units": {"input_tokens": 10}},
    }


def _make_model(config=None, batches=None):
    """Create a model whose POST returns one batch response per call.

    ``batches`` is a list of vector-lists; each successive POST returns the next
    one. Defaults to a single 2-vector response.
    """
    if batches is None:
        batches = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]
    # Avoid creating real httpx clients we immediately discard for mocks.
    with patch.object(CohereEmbeddingModel, "_create_http_clients", lambda self: None):
        model = CohereEmbeddingModel(
            model_name="embed-v4.0", api_key="test-key", config=config or {}
        )

    sync_calls = {"i": 0}
    async_calls = {"i": 0}

    def post(url, **kwargs):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = _embed_response(batches[sync_calls["i"]])
        sync_calls["i"] += 1
        return resp

    async def apost(url, **kwargs):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = _embed_response(batches[async_calls["i"]])
        async_calls["i"] += 1
        return resp

    mock_client = Mock()
    mock_async_client = AsyncMock()
    mock_client.post.side_effect = post
    mock_async_client.post.side_effect = apost
    model.client = mock_client
    model.async_client = mock_async_client
    return model


class TestCohereEmbedding:
    def test_initialization_with_api_key(self):
        model = CohereEmbeddingModel(model_name="embed-v4.0", api_key="test-key", config={})
        assert model.api_key == "test-key"
        assert model.provider == "cohere"
        assert model.base_url == "https://api.cohere.com"
        assert model.input_type == "search_document"

    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Cohere API key not found"):
                CohereEmbeddingModel(model_name="embed-v4.0", api_key=None, config={})

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {"COHERE_API_KEY": "env-key"}):
            model = CohereEmbeddingModel(model_name="embed-v4.0", api_key=None, config={})
            assert model.api_key == "env-key"

    def test_default_model(self):
        model = CohereEmbeddingModel(api_key="test-key", config={})
        assert model.get_model_name() == "embed-v4.0"

    def test_input_type_from_config(self):
        model = CohereEmbeddingModel(
            model_name="embed-v4.0", api_key="test-key", config={"input_type": "search_query"}
        )
        assert model.input_type == "search_query"

    def test_payload_shape(self):
        model = CohereEmbeddingModel(model_name="embed-v4.0", api_key="test-key", config={})
        payload = model._build_payload(["hello"])
        assert payload["model"] == "embed-v4.0"
        assert payload["texts"] == ["hello"]
        assert payload["input_type"] == "search_document"
        assert payload["embedding_types"] == ["float"]

    def test_embed(self):
        model = _make_model()
        result = model.embed(["Hello", "World"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        url = model.client.post.call_args[0][0]
        assert url == "https://api.cohere.com/v2/embed"
        payload = model.client.post.call_args[1]["json"]
        assert payload["input_type"] == "search_document"

    def test_embed_per_call_input_type(self):
        model = _make_model()
        model.embed(["Hello"], input_type="search_query")
        payload = model.client.post.call_args[1]["json"]
        assert payload["input_type"] == "search_query"

    def test_batching_over_96(self):
        # 200 texts -> 96 + 96 + 8 == 3 batches/POSTs.
        batches = [
            [[float(i)] for i in range(96)],
            [[float(i)] for i in range(96)],
            [[float(i)] for i in range(8)],
        ]
        model = _make_model(batches=batches)
        texts = [f"t{i}" for i in range(200)]
        result = model.embed(texts)
        assert model.client.post.call_count == 3
        assert len(result) == 200

    @pytest.mark.asyncio
    async def test_aembed(self):
        model = _make_model()
        result = await model.aembed(["Hello", "World"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        model.async_client.post.assert_awaited_once()
