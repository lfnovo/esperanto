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


# =============================================================================
# OpenAI Tests
# =============================================================================


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


# =============================================================================
# Google Tests
# =============================================================================


@pytest.mark.xfail(
    reason="Google default model text-embedding-004 deprecated on v1beta — see #177",
    strict=False,
)
@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
class TestGoogleEmbedding:
    """Real integration tests for Google embeddings."""

    def test_sync_embed(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_embedding("google", "text-embedding-004", config={"api_key": api_key})
        result = model.embed(TEXTS_SINGLE)
        _assert_valid_embedding(result, 1)

    def test_async_embed(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_embedding("google", "text-embedding-004", config={"api_key": api_key})
        result = asyncio.run(model.aembed(TEXTS_SINGLE))
        _assert_valid_embedding(result, 1)

    def test_batch_embed(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_embedding("google", "text-embedding-004", config={"api_key": api_key})
        result = model.embed(TEXTS_BATCH)
        _assert_valid_embedding(result, 3)

    def test_task_type_embed(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_embedding(
            "google",
            "text-embedding-004",
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

    def _make_model(self):
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


# =============================================================================
# Ollama Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")),
    reason="OLLAMA_BASE_URL or OLLAMA_API_BASE not configured",
)
class TestOllamaEmbedding:
    """Real integration tests for Ollama embeddings."""

    def _make_model(self):
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")
        return AIFactory.create_embedding("ollama", "nomic-embed-text", config={"base_url": base_url})

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

    def _make_model(self):
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
            config={"api_key": api_key, "base_url": base_url},
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
