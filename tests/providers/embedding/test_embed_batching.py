"""Tests for auto-batching of embedding requests across providers.

These verify the batching envelope added on top of each provider's single-request
body: an input longer than the effective batch size is split into
``ceil(len / batch)`` ordered slices, each sent as its own HTTP request, with the
decoded embeddings concatenated back in input order. Empty input makes zero
requests. All tests are hermetic — no real HTTP clients are created.
"""

import math
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.providers.embedding.azure import AzureEmbeddingModel
from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.embedding.cohere import CohereEmbeddingModel
from esperanto.providers.embedding.google import GoogleEmbeddingModel
from esperanto.providers.embedding.mistral import MistralEmbeddingModel
from esperanto.providers.embedding.ollama import OllamaEmbeddingModel
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
from esperanto.providers.embedding.openai_compatible import (
    OpenAICompatibleEmbeddingModel,
)
from esperanto.providers.embedding.openrouter import OpenRouterEmbeddingModel
from esperanto.providers.embedding.vertex import VertexEmbeddingModel
from esperanto.providers.embedding.voyage import VoyageEmbeddingModel

# --- response builders -------------------------------------------------------
# Each embedding encodes the numeric suffix of its text ("t7" -> [7.0]) so we can
# assert both concatenation and ordering of the final result.


def _text_value(text):
    return float(int(text[1:]))


def _data_response(batch):
    """OpenAI/Voyage/Mistral-style response (``data`` -> list of embeddings)."""
    return {"data": [{"embedding": [_text_value(t)]} for t in batch]}


def _cohere_response(batch):
    """Cohere v2-style response (``embeddings.float``)."""
    return {"embeddings": {"float": [[_text_value(t)] for t in batch]}}


# Provider -> (class, input payload key, response builder, MAX_BATCH_SIZE)
PROVIDERS = {
    "openai": (OpenAIEmbeddingModel, "input", _data_response, 2048),
    "voyage": (VoyageEmbeddingModel, "input", _data_response, 1000),
    "mistral": (MistralEmbeddingModel, "input", _data_response, 64),
    "cohere": (CohereEmbeddingModel, "texts", _cohere_response, 96),
    "openrouter": (OpenRouterEmbeddingModel, "input", _data_response, 96),
}


def _make_model(cls, response_builder, config=None, **extra):
    """Build a provider instance with mocked sync/async HTTP clients.

    Each POST inspects the request payload and returns a response sized to the
    batch it was given, so batching behaviour is exercised end to end. ``extra``
    forwards provider-specific constructor kwargs (e.g. Azure's endpoint).
    """
    ctor = {"model_name": "test-model", "api_key": "test-key", "config": config or {}}
    ctor.update(extra)
    with patch.object(cls, "_create_http_clients", lambda self: None):
        model = cls(**ctor)

    def _batch_from(kwargs):
        payload = kwargs["json"]
        return payload.get("input", payload.get("texts"))

    def post(url, **kwargs):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = response_builder(_batch_from(kwargs))
        return resp

    async def apost(url, **kwargs):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = response_builder(_batch_from(kwargs))
        return resp

    mock_client = Mock()
    mock_async_client = AsyncMock()
    mock_client.post.side_effect = post
    mock_async_client.post.side_effect = apost
    model.client = mock_client
    model.async_client = mock_async_client
    return model


def _sent_batches(mock_client):
    """Reconstruct the list-of-batches actually sent, in call order."""
    batches = []
    for call in mock_client.post.call_args_list:
        payload = call.kwargs["json"]
        batches.append(payload.get("input", payload.get("texts")))
    return batches


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_multi_batch_sync(provider):
    cls, _key, builder, _max = PROVIDERS[provider]
    model = _make_model(cls, builder, config={"embed_batch_size": 2})

    texts = [f"t{i}" for i in range(5)]
    result = model.embed(texts)

    # 5 texts, batch 2 -> ceil(5/2) == 3 requests.
    assert model.client.post.call_count == math.ceil(5 / 2)
    # Correct ordered slices.
    assert _sent_batches(model.client) == [["t0", "t1"], ["t2", "t3"], ["t4"]]
    # Concatenated in input order.
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_multi_batch_async(provider):
    cls, _key, builder, _max = PROVIDERS[provider]
    model = _make_model(cls, builder, config={"embed_batch_size": 2})

    texts = [f"t{i}" for i in range(5)]
    result = await model.aembed(texts)

    assert model.async_client.post.await_count == math.ceil(5 / 2)
    assert _sent_batches(model.async_client) == [["t0", "t1"], ["t2", "t3"], ["t4"]]
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_empty_input_makes_zero_requests(provider):
    cls, _key, builder, _max = PROVIDERS[provider]
    model = _make_model(cls, builder)

    assert model.embed([]) == []
    assert model.client.post.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_empty_input_makes_zero_requests_async(provider):
    cls, _key, builder, _max = PROVIDERS[provider]
    model = _make_model(cls, builder)

    assert await model.aembed([]) == []
    assert model.async_client.post.await_count == 0


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_input_within_batch_is_single_request(provider):
    cls, _key, builder, _max = PROVIDERS[provider]
    model = _make_model(cls, builder)  # no override -> MAX_BATCH_SIZE (large)

    texts = [f"t{i}" for i in range(3)]
    result = model.embed(texts)

    assert model.client.post.call_count == 1
    assert _sent_batches(model.client) == [["t0", "t1", "t2"]]
    assert result == [[0.0], [1.0], [2.0]]


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_override_above_max_clamps_to_max(provider):
    cls, _key, _builder, expected_max = PROVIDERS[provider]
    model = _make_model(cls, _builder, config={"embed_batch_size": expected_max + 10_000})

    # Effective batch size is clamped to the provider ceiling.
    assert model._get_embed_batch_size() == expected_max


def test_clamp_is_functional_for_mistral():
    # Mistral MAX == 64. Override 1000 clamps to 64, so 65 texts -> 2 requests.
    cls, _key, builder, _max = PROVIDERS["mistral"]
    model = _make_model(cls, builder, config={"embed_batch_size": 1000})

    texts = [f"t{i}" for i in range(65)]
    result = model.embed(texts)

    assert model.client.post.call_count == 2
    batches = _sent_batches(model.client)
    assert [len(b) for b in batches] == [64, 1]
    assert result == [[float(i)] for i in range(65)]


@pytest.mark.parametrize("provider", list(PROVIDERS))
@pytest.mark.parametrize("bad", [0, -1, -50])
def test_non_positive_batch_size_raises(provider, bad):
    cls, _key, builder, _max = PROVIDERS[provider]
    model = _make_model(cls, builder, config={"embed_batch_size": bad})

    with pytest.raises(ValueError, match="embed_batch_size must be a positive integer"):
        model.embed(["t0"])
    assert model.client.post.call_count == 0


# --- base resolver / iterator unit tests -------------------------------------


class _DummyEmbed(EmbeddingModel):
    """Minimal concrete provider (MAX_BATCH_SIZE=10) for base-logic tests."""

    MAX_BATCH_SIZE = 10

    def embed(self, texts, **kwargs):  # pragma: no cover - not exercised
        return []

    async def aembed(self, texts, **kwargs):  # pragma: no cover - not exercised
        return []

    def _get_models(self):
        return []

    def _get_default_model(self):
        return "dummy"

    @property
    def provider(self):
        return "dummy"


class _NoCapEmbed(_DummyEmbed):
    """Provider with no batch cap (MAX_BATCH_SIZE=0)."""

    MAX_BATCH_SIZE = 0


def test_resolver_default_is_max():
    model = _DummyEmbed()
    assert model._get_embed_batch_size() == 10


def test_resolver_override_below_max():
    model = _DummyEmbed(config={"embed_batch_size": 4})
    assert model._get_embed_batch_size() == 4


def test_resolver_override_above_max_clamps():
    model = _DummyEmbed(config={"embed_batch_size": 999})
    assert model._get_embed_batch_size() == 10


def test_resolver_no_cap_returns_none():
    model = _NoCapEmbed()
    assert model._get_embed_batch_size() is None


def test_resolver_no_cap_override_applies():
    model = _NoCapEmbed(config={"embed_batch_size": 3})
    assert model._get_embed_batch_size() == 3


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "5"])
def test_resolver_invalid_override_raises(bad):
    model = _DummyEmbed(config={"embed_batch_size": bad})
    with pytest.raises(ValueError, match="embed_batch_size must be a positive integer"):
        model._get_embed_batch_size()


def test_iter_batches_slices_in_order():
    model = _DummyEmbed(config={"embed_batch_size": 10})
    texts = [f"t{i}" for i in range(25)]
    batches = list(model._iter_embed_batches(texts))
    assert [len(b) for b in batches] == [10, 10, 5]
    # Slices reassemble to the original input, in order.
    assert [t for b in batches for t in b] == texts


def test_iter_batches_empty_yields_nothing():
    model = _DummyEmbed()
    assert list(model._iter_embed_batches([])) == []


def test_iter_batches_no_cap_single_yield():
    model = _NoCapEmbed()
    texts = [f"t{i}" for i in range(50)]
    batches = list(model._iter_embed_batches(texts))
    assert batches == [texts]


# --- regression: embed_batch_size must stay client-side ----------------------


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_embed_batch_size_not_leaked_into_payload(provider):
    """The override is a client-side control; it must never reach the API body."""
    cls, _key, builder, _max = PROVIDERS[provider]
    model = _make_model(cls, builder, config={"embed_batch_size": 2})

    model.embed([f"t{i}" for i in range(3)])

    for call in model.client.post.call_args_list:
        assert "embed_batch_size" not in call.kwargs["json"]


# --- regression: batch offset preserves the original input index -------------


def test_error_index_is_global_across_batches():
    """An invalid embedding in a later batch reports its original input index."""
    model = _make_model(OpenAIEmbeddingModel, _data_response, config={"embed_batch_size": 2})

    seen = []

    def spy(idx, raw):
        seen.append(idx)
        return [float(idx)]

    with patch("esperanto.providers.embedding.openai.validate_and_decode_embedding", spy):
        model.embed([f"t{i}" for i in range(5)])

    # Global indices, not per-batch (which would be [0, 1, 0, 1, 0]).
    assert seen == [0, 1, 2, 3, 4]


# --- Azure: distinct payload/construction path -------------------------------


def _make_azure(config=None):
    return _make_model(
        AzureEmbeddingModel,
        _data_response,
        config=config,
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-01",
    )


def test_azure_multi_batch_sync():
    model = _make_azure(config={"embed_batch_size": 2})
    result = model.embed([f"t{i}" for i in range(5)])
    assert model.client.post.call_count == 3
    assert _sent_batches(model.client) == [["t0", "t1"], ["t2", "t3"], ["t4"]]
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


@pytest.mark.asyncio
async def test_azure_multi_batch_async():
    model = _make_azure(config={"embed_batch_size": 2})
    result = await model.aembed([f"t{i}" for i in range(5)])
    assert model.async_client.post.await_count == 3
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_azure_empty_input_makes_zero_requests():
    model = _make_azure()
    assert model.embed([]) == []
    assert model.client.post.call_count == 0


# --- Ollama: empty input returns [] (batching contract), no request ----------


def _ollama_response(batch):
    return {"embeddings": [[_text_value(t)] for t in batch]}


def _make_ollama(config=None):
    return _make_model(
        OllamaEmbeddingModel,
        _ollama_response,
        config=config,
        base_url="http://localhost:11434",
    )


def test_ollama_empty_input_returns_empty():
    model = _make_ollama()
    assert model.embed([]) == []
    assert model.client.post.call_count == 0


@pytest.mark.asyncio
async def test_ollama_empty_input_returns_empty_async():
    model = _make_ollama()
    assert await model.aembed([]) == []
    assert model.async_client.post.await_count == 0


# --- Google: native :batchEmbedContents (issue #203) -------------------------


def _make_google(config=None):
    with patch.object(GoogleEmbeddingModel, "_create_http_clients", lambda self: None):
        model = GoogleEmbeddingModel(
            model_name="text-embedding-004", api_key="test-key", config=config or {}
        )

    def _texts_of(kwargs):
        return [r["content"]["parts"][0]["text"] for r in kwargs["json"]["requests"]]

    def post(url, **kwargs):
        assert "batchEmbedContents" in url
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "embeddings": [{"values": [_text_value(t)]} for t in _texts_of(kwargs)]
        }
        return resp

    async def apost(url, **kwargs):
        return post(url, **kwargs)

    model.client = Mock()
    model.async_client = AsyncMock()
    model.client.post.side_effect = post
    model.async_client.post.side_effect = apost
    return model


def test_google_multi_batch_uses_batch_endpoint():
    model = _make_google(config={"embed_batch_size": 2})
    result = model.embed([f"t{i}" for i in range(5)])

    # 5 texts, batch 2 -> 3 batchEmbedContents calls.
    assert model.client.post.call_count == 3
    # Each request bundles its slice under "requests".
    sizes = [len(c.kwargs["json"]["requests"]) for c in model.client.post.call_args_list]
    assert sizes == [2, 2, 1]
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


@pytest.mark.asyncio
async def test_google_multi_batch_async():
    model = _make_google(config={"embed_batch_size": 2})
    result = await model.aembed([f"t{i}" for i in range(5)])
    assert model.async_client.post.await_count == 3
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_google_empty_input_makes_zero_requests():
    model = _make_google()
    assert model.embed([]) == []
    assert model.client.post.call_count == 0


def test_google_default_batch_size_is_250():
    assert GoogleEmbeddingModel.MAX_BATCH_SIZE == 250
    model = _make_google()
    assert model._get_embed_batch_size() == 250


# --- Vertex: multiple instances per :predict request -------------------------


def _make_vertex(config=None):
    with patch("subprocess.run") as mock_subprocess, patch.object(
        VertexEmbeddingModel, "_create_http_clients", lambda self: None
    ):
        mock_subprocess.return_value.stdout = "mock_access_token"
        mock_subprocess.return_value.returncode = 0
        model = VertexEmbeddingModel(
            vertex_project="test-project", model_name="textembedding-gecko", config=config or {}
        )
    model._get_access_token = Mock(return_value="mock_access_token")

    def _texts_of(kwargs):
        return [inst["content"] for inst in kwargs["json"]["instances"]]

    def post(url, **kwargs):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "predictions": [
                {"embeddings": {"values": [_text_value(t)]}} for t in _texts_of(kwargs)
            ]
        }
        return resp

    async def apost(url, **kwargs):
        return post(url, **kwargs)

    model.client = Mock()
    model.async_client = AsyncMock()
    model.client.post.side_effect = post
    model.async_client.post.side_effect = apost
    return model


def test_vertex_multi_batch_bundles_instances():
    model = _make_vertex(config={"embed_batch_size": 2})
    result = model.embed([f"t{i}" for i in range(5)])

    assert model.client.post.call_count == 3
    sizes = [len(c.kwargs["json"]["instances"]) for c in model.client.post.call_args_list]
    assert sizes == [2, 2, 1]
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_vertex_empty_input_makes_zero_requests():
    model = _make_vertex()
    assert model.embed([]) == []
    assert model.client.post.call_count == 0


# --- OpenAI-compatible: batches like OpenAI (2048), needs a base_url ----------


def _make_openai_compatible(config=None):
    return _make_model(
        OpenAICompatibleEmbeddingModel,
        _data_response,
        config=config,
        base_url="http://localhost:1234/v1",
    )


def test_openai_compatible_multi_batch_sync():
    model = _make_openai_compatible(config={"embed_batch_size": 2})
    result = model.embed([f"t{i}" for i in range(5)])
    assert model.client.post.call_count == 3
    assert _sent_batches(model.client) == [["t0", "t1"], ["t2", "t3"], ["t4"]]
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


@pytest.mark.asyncio
async def test_openai_compatible_multi_batch_async():
    model = _make_openai_compatible(config={"embed_batch_size": 2})
    result = await model.aembed([f"t{i}" for i in range(5)])
    assert model.async_client.post.await_count == 3
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_openai_compatible_empty_input_makes_zero_requests():
    model = _make_openai_compatible()
    assert model.embed([]) == []
    assert model.client.post.call_count == 0


def test_openai_compatible_default_batch_size_is_2048():
    assert OpenAICompatibleEmbeddingModel.MAX_BATCH_SIZE == 2048
