"""Real integration tests for embedding providers — these call actual APIs.

These tests verify that embedding models work correctly with real API calls.
They require API keys to be configured in the environment.

Run with: uv run pytest tests/integration/test_embedding_real.py -v -s -m release
"""

import asyncio
import os

import pytest

from esperanto import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# =============================================================================
# Module-level availability flags
# =============================================================================

try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# =============================================================================
# Helpers
# =============================================================================

TEXTS_SINGLE = ["Hello world"]
TEXTS_BATCH = ["Hello world", "How are you?", "Testing embeddings"]


def _assert_valid_embedding(result: list, expected_len: int) -> None:
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == expected_len, f"Expected {expected_len} embeddings, got {len(result)}"
    for emb in result:
        assert isinstance(emb, list), f"Each embedding must be a list, got {type(emb)}"
        assert len(emb) > 0, "Embedding vector must be non-empty"
        assert all(isinstance(v, float) for v in emb), "Embedding values must be floats"


# --- Auto-batching (split + concatenate) -----------------------------------
#
# The plain test_batch_embed cases above send 3 texts, which is under every
# provider ceiling — they never exercise the split. These helpers force a small
# batch size so the request actually fans out against the real API, then check
# the results come back in input order across the batch boundaries.

# 10 texts at batch size 4 => 3 requests (4 + 4 + 2). The last text repeats the
# first, so the two land in *different* requests: if the concatenation ever
# scrambled order, the repeat would stop lining up with the original.
BATCH_SPLIT_SIZE = 4
TEXTS_OVER_CEILING = [f"Batch item number {i}" for i in range(9)] + ["Batch item number 0"]


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def _assert_batches_in_input_order(result: list) -> None:
    """The repeated text must line up with its original across a batch boundary.

    Compared by cosine similarity rather than equality: OpenAI, Voyage and
    Mistral return float-level differences for the same text across separate
    requests, so `==` fails on providers whose ordering is perfectly correct.
    Similarity is also threshold-free here — we assert the repeat's *nearest*
    vector is the original, which a scrambled concatenation could not satisfy.
    """
    _assert_valid_embedding(result, len(TEXTS_OVER_CEILING))
    originals = result[:-1]
    nearest = max(range(len(originals)), key=lambda i: _cosine(result[-1], originals[i]))
    assert nearest == 0, (
        f"The repeated text is nearest to input {nearest}, not input 0 — "
        "results were not concatenated in input order across batches"
    )


def _assert_splits_across_requests(provider: str, model: str, **config) -> None:
    """Embed more texts than the forced batch size and verify order survives."""
    embed_model = AIFactory.create_embedding(
        provider, model, config={"embed_batch_size": BATCH_SPLIT_SIZE, **config}
    )
    _assert_batches_in_input_order(embed_model.embed(TEXTS_OVER_CEILING))


# =============================================================================
# OpenAI Tests
# =============================================================================


def _ollama_available(required_model: str = "") -> bool:
    """Probe for a reachable Ollama instance, optionally requiring a model.

    Ollama defaults to ``http://localhost:11434`` per its own provider source,
    so the test should run whenever Ollama is reachable — locally OR via the
    optional ``OLLAMA_BASE_URL`` / ``OLLAMA_API_BASE`` env override. Avoids
    skipping tests when the user has Ollama running locally without setting
    an env var.

    When ``required_model`` is supplied, also confirm that model is in the
    server's ``/api/tags`` listing — protects tests that need a specific
    model (e.g. tool-calling needs ``qwen3:32b``) from running and failing
    against a server that doesn't have it pulled.
    """
    import httpx
    base_url = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_API_BASE")
        or "http://localhost:11434"
    )
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        if response.status_code != 200:
            return False
        if required_model:
            tags = [m.get("name", "") for m in response.json().get("models", [])]
            return any(required_model in tag for tag in tags)
        return True
    except Exception:
        return False


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not configured",
)
class TestOpenAIEmbedding:
    """Real integration tests for OpenAI embeddings."""

    def test_sync_embed(self):
        model = AIFactory.create_embedding("openai", "text-embedding-3-small")
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = AIFactory.create_embedding("openai", "text-embedding-3-small")
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = AIFactory.create_embedding("openai", "text-embedding-3-small")
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Input above the batch size fans out and comes back in order."""
        _assert_splits_across_requests("openai", "text-embedding-3-small")


# =============================================================================
# Google Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
class TestGoogleEmbedding:
    """Real integration tests for Google embeddings."""

    def test_sync_embed(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_embedding("google", "gemini-embedding-001", config={"api_key": api_key})
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_embedding("google", "gemini-embedding-001", config={"api_key": api_key})
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_embedding("google", "gemini-embedding-001", config={"api_key": api_key})
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Google fans out over the native :batchEmbedContents endpoint, in order."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        _assert_splits_across_requests(
            "google", "gemini-embedding-001", api_key=api_key
        )

    def test_task_type_embed(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_embedding(
            "google",
            "gemini-embedding-001",
            config={"api_key": api_key, "task_type": EmbeddingTaskType.RETRIEVAL_QUERY},
        )
        result = model.embed(["query text"])
        _assert_valid_embedding(result, 1)


# =============================================================================
# Vertex AI Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")),
    reason="VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT not configured",
)
class TestVertexEmbedding:
    """Real integration tests for Vertex AI embeddings."""

    def test_sync_embed(self):
        model = AIFactory.create_embedding("vertex", "text-embedding-005")
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = AIFactory.create_embedding("vertex", "text-embedding-005")
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = AIFactory.create_embedding("vertex", "text-embedding-005")
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Vertex fans out over the multi-instance :predict endpoint, in order."""
        _assert_splits_across_requests("vertex", "text-embedding-005")

    def test_batch_embed_over_native_ceiling(self):
        """Vertex's real default ceiling is 25 — cross it without forcing a size."""
        model = AIFactory.create_embedding("vertex", "text-embedding-005")
        texts = [f"Native ceiling item {i}" for i in range(29)] + ["Native ceiling item 0"]
        result = model.embed(texts)
        _assert_valid_embedding(result, 30)
        nearest = max(range(29), key=lambda i: _cosine(result[-1], result[i]))
        assert nearest == 0, "Vertex batches were not concatenated in input order"


# =============================================================================
# Azure Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (
        (os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING") or os.getenv("AZURE_OPENAI_API_KEY"))
        and (os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING") or os.getenv("AZURE_OPENAI_ENDPOINT"))
    ),
    reason="Azure embedding requires both an API key and an endpoint (AZURE_OPENAI_API_KEY[_EMBEDDING] + AZURE_OPENAI_ENDPOINT[_EMBEDDING])",
)
class TestAzureEmbedding:
    """Real integration tests for Azure OpenAI embeddings."""

    def _make_model(self, **extra_config):
        return AIFactory.create_embedding(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING", "text-embedding-3-small"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING") or os.getenv("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": (
                    os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING") or os.getenv("AZURE_OPENAI_ENDPOINT")
                ),
                "api_version": (
                    os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING")
                    or os.getenv("OPENAI_API_VERSION")
                    or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
                ),
                **extra_config,
            },
        )

    def test_sync_embed(self):
        model = self._make_model()
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = self._make_model()
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = self._make_model()
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Input above the batch size fans out and comes back in order."""
        model = self._make_model(embed_batch_size=BATCH_SPLIT_SIZE)
        _assert_batches_in_input_order(model.embed(TEXTS_OVER_CEILING))


# =============================================================================
# Jina Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("JINA_API_KEY"),
    reason="JINA_API_KEY not configured",
)
class TestJinaEmbedding:
    """Real integration tests for Jina AI embeddings."""

    def test_sync_embed(self):
        model = AIFactory.create_embedding("jina", "jina-embeddings-v3")
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = AIFactory.create_embedding("jina", "jina-embeddings-v3")
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = AIFactory.create_embedding("jina", "jina-embeddings-v3")
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Input above the batch size fans out and comes back in order."""
        _assert_splits_across_requests("jina", "jina-embeddings-v3")

    def test_task_type_embed(self):
        model = AIFactory.create_embedding(
            "jina",
            "jina-embeddings-v3",
            config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY},
        )
        result = model.embed(["query text"])
        _assert_valid_embedding(result, 1)


# =============================================================================
# Voyage Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not configured",
)
class TestVoyageEmbedding:
    """Real integration tests for Voyage AI embeddings.

    Note: Voyage does not declare SUPPORTED_FEATURES and uses the base class
    prefix-based task optimization instead of a native task_type API parameter.
    Accordingly, test_task_type_embed is omitted — the task_type feature is
    handled transparently by the base class and does not require a separate
    native-API verification test.
    """

    def test_sync_embed(self):
        model = AIFactory.create_embedding("voyage", "voyage-3-large")
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = AIFactory.create_embedding("voyage", "voyage-3-large")
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = AIFactory.create_embedding("voyage", "voyage-3-large")
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Input above the batch size fans out and comes back in order."""
        _assert_splits_across_requests("voyage", "voyage-3-large")


# =============================================================================
# Mistral Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not configured",
)
class TestMistralEmbedding:
    """Real integration tests for Mistral embeddings."""

    def test_sync_embed(self):
        model = AIFactory.create_embedding("mistral", "mistral-embed")
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = AIFactory.create_embedding("mistral", "mistral-embed")
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = AIFactory.create_embedding("mistral", "mistral-embed")
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Input above the batch size fans out and comes back in order."""
        _assert_splits_across_requests("mistral", "mistral-embed")

    def test_batch_embed_over_native_ceiling(self):
        """Mistral's real default ceiling is 64 — cross it without forcing a size."""
        model = AIFactory.create_embedding("mistral", "mistral-embed")
        texts = [f"Native ceiling item {i}" for i in range(69)] + ["Native ceiling item 0"]
        result = model.embed(texts)
        _assert_valid_embedding(result, 70)
        nearest = max(range(69), key=lambda i: _cosine(result[-1], result[i]))
        assert nearest == 0, "Mistral batches were not concatenated in input order"


# =============================================================================
# Transformers Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence_transformers not installed",
)
class TestTransformersEmbedding:
    """Real integration tests for HuggingFace Transformers embeddings."""

    def test_sync_embed(self):
        model = AIFactory.create_embedding("transformers", "sentence-transformers/all-MiniLM-L6-v2")
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = AIFactory.create_embedding("transformers", "sentence-transformers/all-MiniLM-L6-v2")
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = AIFactory.create_embedding("transformers", "sentence-transformers/all-MiniLM-L6-v2")
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """transformers keeps its own internal batching — order must still hold."""
        _assert_splits_across_requests(
            "transformers", "sentence-transformers/all-MiniLM-L6-v2"
        )


# =============================================================================
# Ollama Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not reachable at configured base URL or localhost:11434",
)
class TestOllamaEmbedding:
    """Real integration tests for Ollama embeddings."""

    def _make_model(self, **extra_config):
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")
        return AIFactory.create_embedding(
            "ollama", "nomic-embed-text", config={"base_url": base_url, **extra_config}
        )

    def test_sync_embed(self):
        model = self._make_model()
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = self._make_model()
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = self._make_model()
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Ollama is unbatched by default; an explicit size must still split cleanly."""
        model = self._make_model(embed_batch_size=BATCH_SPLIT_SIZE)
        _assert_batches_in_input_order(model.embed(TEXTS_OVER_CEILING))


# =============================================================================
# OpenRouter Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not configured",
)
class TestOpenRouterEmbedding:
    """Real integration tests for OpenRouter embeddings."""

    def test_sync_embed(self):
        model = AIFactory.create_embedding("openrouter", "openai/text-embedding-3-small")
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = AIFactory.create_embedding("openrouter", "openai/text-embedding-3-small")
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = AIFactory.create_embedding("openrouter", "openai/text-embedding-3-small")
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Input above the batch size fans out and comes back in order."""
        _assert_splits_across_requests("openrouter", "openai/text-embedding-3-small")


# =============================================================================
# OpenAI-Compatible Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (
        (os.getenv("OPENAI_COMPATIBLE_API_KEY_EMBEDDING") or os.getenv("OPENAI_COMPATIBLE_API_KEY"))
        and (os.getenv("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING") or os.getenv("OPENAI_COMPATIBLE_BASE_URL"))
    ),
    reason="OpenAI-compatible embedding requires both API key and base URL (OPENAI_COMPATIBLE_API_KEY[_EMBEDDING] + OPENAI_COMPATIBLE_BASE_URL[_EMBEDDING])",
)
class TestOpenAICompatibleEmbedding:
    """Real integration tests for OpenAI-compatible embeddings."""

    def _make_model(self, **extra_config):
        api_key = (
            os.getenv("OPENAI_COMPATIBLE_API_KEY_EMBEDDING") or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        )
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL_EMBEDDING") or os.getenv(
            "OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1"
        )
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL_EMBEDDING", "text-embedding-3-small")
        return AIFactory.create_embedding(
            "openai-compatible",
            model_name,
            config={"api_key": api_key, "base_url": base_url, **extra_config},
        )

    def test_sync_embed(self):
        model = self._make_model()
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        model = self._make_model()
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        model = self._make_model()
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_batch_embed_splits_across_requests(self):
        """Input above the batch size fans out and comes back in order."""
        model = self._make_model(embed_batch_size=BATCH_SPLIT_SIZE)
        _assert_batches_in_input_order(model.embed(TEXTS_OVER_CEILING))
