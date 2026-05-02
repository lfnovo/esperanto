"""Real integration tests for reranking - these call actual APIs.

These tests verify that reranking works correctly with real API calls.
They require API keys to be configured in the environment.

Run with: uv run pytest tests/integration/test_reranker_real.py -v -s -m release
"""

import asyncio
import os

import pytest

from esperanto import AIFactory

# =============================================================================
# Test Configuration
# =============================================================================

QUERY = "What is the capital of France?"
CANDIDATES = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
]


def _transformers_available() -> bool:
    """Check if sentence_transformers package is available."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# Jina Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("JINA_API_KEY"),
    reason="JINA_API_KEY not configured",
)
class TestJinaReranker:
    """Real integration tests for Jina reranker."""

    def test_sync_rerank(self):
        """Test sync reranking."""
        model = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")
        response = model.rerank(QUERY, CANDIDATES)
        assert response.results[0].index == 0
        assert "paris" in response.results[0].document.lower()

    def test_async_arerank(self):
        """Test async reranking."""
        model = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

        async def _run() -> object:
            return await model.arerank(QUERY, CANDIDATES)

        response = asyncio.run(_run())
        assert response.results[0].index == 0
        assert "paris" in response.results[0].document.lower()


# =============================================================================
# Voyage Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not configured",
)
class TestVoyageReranker:
    """Real integration tests for Voyage reranker."""

    def test_sync_rerank(self):
        """Test sync reranking."""
        model = AIFactory.create_reranker("voyage", "rerank-2")
        response = model.rerank(QUERY, CANDIDATES)
        assert response.results[0].index == 0
        assert "paris" in response.results[0].document.lower()

    def test_async_arerank(self):
        """Test async reranking."""
        model = AIFactory.create_reranker("voyage", "rerank-2")

        async def _run() -> object:
            return await model.arerank(QUERY, CANDIDATES)

        response = asyncio.run(_run())
        assert response.results[0].index == 0
        assert "paris" in response.results[0].document.lower()


# =============================================================================
# Transformers Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not _transformers_available(),
    reason="sentence_transformers package not installed",
)
class TestTransformersReranker:
    """Real integration tests for Transformers reranker (local, no API key needed)."""

    def test_sync_rerank(self):
        """Test sync reranking."""
        model = AIFactory.create_reranker(
            "transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        response = model.rerank(QUERY, CANDIDATES)
        assert response.results[0].index == 0
        assert "paris" in response.results[0].document.lower()

    def test_async_arerank(self):
        """Test async reranking (runs sync in thread pool)."""
        model = AIFactory.create_reranker(
            "transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        async def _run() -> object:
            return await model.arerank(QUERY, CANDIDATES)

        response = asyncio.run(_run())
        assert response.results[0].index == 0
        assert "paris" in response.results[0].document.lower()
