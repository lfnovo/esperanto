"""Tests for HttpConnectionMixin functionality."""

import gc
import pytest

from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.reranker.base import RerankerModel
from esperanto.providers.stt.base import SpeechToTextModel
from esperanto.providers.tts.base import TextToSpeechModel


# Test implementations for each provider type
class MockEmbeddingModel(EmbeddingModel):
    """Test embedding model implementation."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "test-embedding")
        self.api_key = kwargs.get("api_key", "test-key")
        self.base_url = kwargs.get("base_url", "https://api.test.com")
        super().__post_init__()
        self._create_http_clients()

    def embed(self, texts, **kwargs):
        pass

    async def aembed(self, texts, **kwargs):
        pass

    def _get_default_model(self):
        return "test-embedding"

    @property
    def provider(self):
        return "test"

    def _get_models(self):
        return []


class MockLanguageModel(LanguageModel):
    """Test language model implementation."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "test-llm")
        self.api_key = kwargs.get("api_key", "test-key")
        self.base_url = kwargs.get("base_url", "https://api.test.com")
        super().__post_init__()
        self._create_http_clients()

    def chat_complete(self, messages, **kwargs):
        pass

    async def achat_complete(self, messages, **kwargs):
        pass

    def _get_default_model(self):
        return "test-llm"

    @property
    def provider(self):
        return "test"

    def _get_models(self):
        return []

    def to_langchain(self):
        pass


class MockRerankerModel(RerankerModel):
    """Test reranker model implementation."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "test-reranker")
        self.api_key = kwargs.get("api_key", "test-key")
        self.base_url = kwargs.get("base_url", "https://api.test.com")
        super().__post_init__()
        self._create_http_clients()

    def rerank(self, query, documents, **kwargs):
        pass

    def _get_default_model(self):
        return "test-reranker"

    @property
    def provider(self):
        return "test"

    def _get_models(self):
        return []

    async def arerank(self, query, documents, **kwargs):
        pass

    def to_langchain(self):
        pass


class MockSpeechToTextModel(SpeechToTextModel):
    """Test speech-to-text model implementation."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "test-stt")
        self.api_key = kwargs.get("api_key", "test-key")
        self.base_url = kwargs.get("base_url", "https://api.test.com")
        super().__post_init__()
        self._create_http_clients()

    def transcribe(self, audio, **kwargs):
        pass

    async def atranscribe(self, audio, **kwargs):
        pass

    def _get_default_model(self):
        return "test-stt"

    @property
    def provider(self):
        return "test"

    def _get_models(self):
        return []


class MockTextToSpeechModel(TextToSpeechModel):
    """Test text-to-speech model implementation."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "test-tts")
        self.api_key = kwargs.get("api_key", "test-key")
        self.base_url = kwargs.get("base_url", "https://api.test.com")
        super().__post_init__()
        self._create_http_clients()

    def synthesize(self, text, **kwargs):
        pass

    async def asynthesize(self, text, **kwargs):
        pass

    def _get_default_model(self):
        return "test-tts"

    @property
    def provider(self):
        return "test"

    def _get_models(self):
        return []

    async def agenerate_speech(self, text, voice, **kwargs):
        pass

    @property
    def available_voices(self):
        return {}

    def generate_speech(self, text, voice, **kwargs):
        pass


class TestHTTPClientCreation:
    """Test HTTP client creation functionality."""

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_http_clients_created(self, model_class):
        """Test that HTTP clients are created after initialization."""
        model = model_class()
        assert hasattr(model, "client")
        assert hasattr(model, "async_client")
        assert model.client is not None
        assert model.async_client is not None
        assert not model.client.is_closed
        assert not model.async_client.is_closed

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_http_clients_have_correct_type(self, model_class):
        """Test that HTTP clients are of correct httpx types."""
        import httpx
        model = model_class()
        assert isinstance(model.client, httpx.Client)
        assert isinstance(model.async_client, httpx.AsyncClient)


class TestSyncContextManager:
    """Test synchronous context manager functionality."""

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_sync_context_manager_enters(self, model_class):
        """Test that sync context manager returns self on entry."""
        model = model_class()
        with model as ctx_model:
            assert ctx_model is model
            assert not model.client.is_closed
            assert not model.async_client.is_closed

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_sync_context_manager_closes_client(self, model_class):
        """Test that sync context manager closes client on exit."""
        model = model_class()
        with model:
            pass
        # Client should be closed after context manager exit
        assert model.client.is_closed
        # Async client should still be open (only sync client is closed)
        assert not model.async_client.is_closed

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_sync_context_manager_handles_exceptions(self, model_class):
        """Test that sync context manager handles exceptions correctly."""
        model = model_class()
        try:
            with model:
                raise ValueError("Test exception")
        except ValueError:
            pass
        # Client should still be closed even if exception occurred
        assert model.client.is_closed


class TestAsyncContextManager:
    """Test asynchronous context manager functionality."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_async_context_manager_enters(self, model_class):
        """Test that async context manager returns self on entry."""
        model = model_class()
        async with model as ctx_model:
            assert ctx_model is model
            assert not model.client.is_closed
            assert not model.async_client.is_closed

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_async_context_manager_closes_async_client(self, model_class):
        """Test that async context manager closes async client on exit."""
        model = model_class()
        async with model:
            pass
        # Async client should be closed after context manager exit
        assert model.async_client.is_closed
        # Sync client should still be open (only async client is closed)
        assert not model.client.is_closed

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_async_context_manager_handles_exceptions(self, model_class):
        """Test that async context manager handles exceptions correctly."""
        model = model_class()
        try:
            async with model:
                raise ValueError("Test exception")
        except ValueError:
            pass
        # Async client should still be closed even if exception occurred
        assert model.async_client.is_closed


class TestManualClose:
    """Test manual close functionality."""

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_manual_close_sync_client(self, model_class):
        """Test that manual close() closes sync client."""
        model = model_class()
        assert not model.client.is_closed
        model.close()
        assert model.client.is_closed
        # Async client should still be open
        assert not model.async_client.is_closed

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_manual_close_idempotent(self, model_class):
        """Test that calling close() multiple times is safe."""
        model = model_class()
        model.close()
        assert model.client.is_closed
        # Calling close again should not raise an error
        model.close()
        assert model.client.is_closed

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_manual_close_with_none_client(self, model_class):
        """Test that close() handles None client gracefully."""
        model = model_class()
        model.client = None
        # Should not raise an error
        model.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_manual_aclose_async_client(self, model_class):
        """Test that manual aclose() closes async client."""
        model = model_class()
        assert not model.async_client.is_closed
        await model.aclose()
        assert model.async_client.is_closed
        # Sync client should still be open
        assert not model.client.is_closed

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_manual_aclose_idempotent(self, model_class):
        """Test that calling aclose() multiple times is safe."""
        model = model_class()
        await model.aclose()
        assert model.async_client.is_closed
        # Calling aclose again should not raise an error
        await model.aclose()
        assert model.async_client.is_closed

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_manual_aclose_with_none_client(self, model_class):
        """Test that aclose() handles None async_client gracefully."""
        model = model_class()
        model.async_client = None
        # Should not raise an error
        await model.aclose()


class TestCombinedClose:
    """Test combined close functionality."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_close_both_clients(self, model_class):
        """Test that both clients can be closed independently."""
        model = model_class()
        assert not model.client.is_closed
        assert not model.async_client.is_closed

        model.close()
        assert model.client.is_closed
        assert not model.async_client.is_closed

        await model.aclose()
        assert model.client.is_closed
        assert model.async_client.is_closed


class TestDestructor:
    """Test destructor cleanup functionality."""

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_destructor_closes_client(self, model_class):
        """Test that destructor closes sync client."""
        model = model_class()
        client = model.client
        assert not client.is_closed

        # Delete the model to trigger destructor
        del model
        gc.collect()

        # Client should be closed after destructor
        assert client.is_closed

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_destructor_with_none_client(self, model_class):
        """Test that destructor handles None client gracefully."""
        model = model_class()
        model.client = None
        # Should not raise an error when destructor is called
        del model
        gc.collect()

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_destructor_with_already_closed_client(self, model_class):
        """Test that destructor handles already closed client gracefully."""
        model = model_class()
        model.client.close()
        assert model.client.is_closed
        # Should not raise an error when destructor is called
        del model
        gc.collect()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_close_without_client_attribute(self, model_class):
        """Test that close() handles missing client attribute gracefully."""
        model = model_class()
        # Manually remove client attribute to simulate edge case
        if hasattr(model, "client"):
            delattr(model, "client")
        # Should not raise an error
        model.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_aclose_without_async_client_attribute(self, model_class):
        """Test that aclose() handles missing async_client attribute gracefully."""
        model = model_class()
        # Manually remove async_client attribute to simulate edge case
        if hasattr(model, "async_client"):
            delattr(model, "async_client")
        # Should not raise an error
        await model.aclose()

    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    def test_context_manager_nested_usage(self, model_class):
        """Test that context manager can be used multiple times (recreating clients)."""
        model = model_class()

        # First context manager usage
        with model:
            assert model.client is not None
            assert not model.client.is_closed

        # Client should be closed after first context
        assert model.client.is_closed

        # Recreate clients for second usage
        model._create_http_clients()
        new_client = model.client

        # Second context manager usage
        with model:
            assert model.client is not None
            assert not model.client.is_closed
            assert model.client is new_client  # Should be a new client

        # Client should be closed after second context
        assert model.client.is_closed

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_class", [
        MockEmbeddingModel,
        MockLanguageModel,
        MockRerankerModel,
        MockSpeechToTextModel,
        MockTextToSpeechModel,
    ])
    async def test_async_context_manager_nested_usage(self, model_class):
        """Test that async context manager can be used multiple times."""
        model = model_class()

        # First async context manager usage
        async with model:
            assert model.async_client is not None
            assert not model.async_client.is_closed

        # Async client should be closed after first context
        assert model.async_client.is_closed

        # Recreate clients for second usage
        model._create_http_clients()
        new_async_client = model.async_client

        # Second async context manager usage
        async with model:
            assert model.async_client is not None
            assert not model.async_client.is_closed
            assert model.async_client is new_async_client  # Should be a new client

        # Async client should be closed after second context
        assert model.async_client.is_closed
