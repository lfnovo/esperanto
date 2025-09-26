#!/usr/bin/env python3
"""Integration tests for timeout configuration across AIFactory and providers."""

import os
from unittest.mock import MagicMock, patch
import pytest

from esperanto import AIFactory
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.reranker.base import RerankerModel
from esperanto.providers.stt.base import SpeechToTextModel
from esperanto.providers.tts.base import TextToSpeechModel


class TestAIFactoryTimeoutIntegration:
    """Test AIFactory timeout configuration integration."""

    def test_language_model_timeout_config(self):
        """Test that timeout config is passed through to language models."""
        with patch("esperanto.providers.llm.openai.OpenAILanguageModel.__post_init__") as mock_init:
            # Mock the provider to avoid API key requirements
            mock_instance = MagicMock()
            mock_instance._get_timeout.return_value = 120.0

            with patch("esperanto.providers.llm.openai.OpenAILanguageModel") as mock_class:
                mock_class.return_value = mock_instance

                # Create model with timeout config
                model = AIFactory.create_language(
                    "openai",
                    "gpt-3.5-turbo",
                    config={"timeout": 120.0}
                )

                # Verify the model was created with config containing timeout
                mock_class.assert_called_once()
                call_args = mock_class.call_args
                assert "config" in call_args.kwargs
                assert call_args.kwargs["config"]["timeout"] == 120.0

    def test_embedding_model_timeout_config(self):
        """Test that timeout config is passed through to embedding models."""
        with patch("esperanto.providers.embedding.openai.OpenAIEmbeddingModel.__post_init__") as mock_init:
            mock_instance = MagicMock()
            mock_instance._get_timeout.return_value = 90.0

            with patch("esperanto.providers.embedding.openai.OpenAIEmbeddingModel") as mock_class:
                mock_class.return_value = mock_instance

                model = AIFactory.create_embedding(
                    "openai",
                    "text-embedding-3-small",
                    config={"timeout": 90.0}
                )

                mock_class.assert_called_once()
                call_args = mock_class.call_args
                assert "config" in call_args.kwargs
                assert call_args.kwargs["config"]["timeout"] == 90.0

    def test_stt_model_timeout_config(self):
        """Test that timeout config is passed through to STT models."""
        with patch("esperanto.providers.stt.openai.OpenAISpeechToTextModel.__post_init__") as mock_init:
            mock_instance = MagicMock()
            mock_instance._get_timeout.return_value = 600.0

            with patch("esperanto.providers.stt.openai.OpenAISpeechToTextModel") as mock_class:
                mock_class.return_value = mock_instance

                model = AIFactory.create_speech_to_text(
                    "openai",
                    config={"timeout": 600.0}
                )

                mock_class.assert_called_once()
                call_args = mock_class.call_args
                # STT uses kwargs pattern, so timeout should be in kwargs
                assert "timeout" in call_args.kwargs
                assert call_args.kwargs["timeout"] == 600.0

    def test_tts_model_timeout_config(self):
        """Test that timeout config is passed through to TTS models."""
        with patch("esperanto.providers.tts.elevenlabs.ElevenLabsTextToSpeechModel.__post_init__") as mock_init:
            mock_instance = MagicMock()
            mock_instance._get_timeout.return_value = 300.0

            with patch("esperanto.providers.tts.elevenlabs.ElevenLabsTextToSpeechModel") as mock_class:
                mock_class.return_value = mock_instance

                try:
                    model = AIFactory.create_text_to_speech(
                        "elevenlabs",
                        timeout=300.0
                    )

                    mock_class.assert_called_once()
                    call_args = mock_class.call_args
                    # TTS uses direct parameter pattern
                    assert "timeout" in call_args.kwargs
                    assert call_args.kwargs["timeout"] == 300.0

                except ValueError as e:
                    if "API key" in str(e):
                        pytest.skip("ElevenLabs API key not available")
                    raise

    def test_reranker_model_timeout_config(self):
        """Test that timeout config is passed through to reranker models."""
        with patch("esperanto.providers.reranker.voyage.VoyageRerankerModel.__post_init__") as mock_init:
            mock_instance = MagicMock()
            mock_instance._get_timeout.return_value = 75.0

            with patch("esperanto.providers.reranker.voyage.VoyageRerankerModel") as mock_class:
                mock_class.return_value = mock_instance

                try:
                    model = AIFactory.create_reranker(
                        "voyage",
                        "rerank-2",
                        config={"timeout": 75.0}
                    )

                    mock_class.assert_called_once()
                    call_args = mock_class.call_args
                    assert "config" in call_args.kwargs
                    assert call_args.kwargs["config"]["timeout"] == 75.0

                except ValueError as e:
                    if "API key" in str(e):
                        pytest.skip("Voyage API key not available")
                    raise


class TestBaseClassTimeoutIntegration:
    """Test base class timeout integration."""

    def test_language_model_provider_type(self):
        """Test that LanguageModel returns correct provider type."""

        class TestLanguageModel(LanguageModel):
            def chat_complete(self, messages, **kwargs):
                pass

            async def achat_complete(self, messages, **kwargs):
                pass

            def _get_default_model(self):
                return "test-model"

            @property
            def provider(self):
                return "test"

            @property
            def models(self):
                return []

            def to_langchain(self):
                pass

        model = TestLanguageModel(model_name="test")
        assert model._get_provider_type() == "language"

    def test_embedding_model_provider_type(self):
        """Test that EmbeddingModel returns correct provider type."""

        class TestEmbeddingModel(EmbeddingModel):
            def embed(self, texts, **kwargs):
                pass

            async def aembed(self, texts, **kwargs):
                pass

            def _get_default_model(self):
                return "test-model"

            @property
            def provider(self):
                return "test"

            @property
            def models(self):
                return []

        model = TestEmbeddingModel(model_name="test")
        assert model._get_provider_type() == "embedding"

    def test_reranker_model_provider_type(self):
        """Test that RerankerModel returns correct provider type."""

        class TestRerankerModel(RerankerModel):
            def rerank(self, query, documents, **kwargs):
                pass

            async def arerank(self, query, documents, **kwargs):
                pass

            def _get_default_model(self):
                return "test-model"

            @property
            def provider(self):
                return "test"

            @property
            def models(self):
                return []

            def to_langchain(self):
                pass

        model = TestRerankerModel(model_name="test")
        assert model._get_provider_type() == "reranker"

    def test_stt_model_provider_type(self):
        """Test that SpeechToTextModel returns correct provider type."""

        class TestSTTModel(SpeechToTextModel):
            def transcribe(self, audio_file, **kwargs):
                pass

            async def atranscribe(self, audio_file, **kwargs):
                pass

            def _get_default_model(self):
                return "test-model"

            @property
            def provider(self):
                return "test"

            @property
            def models(self):
                return []

        model = TestSTTModel(model_name="test")
        assert model._get_provider_type() == "speech_to_text"

    def test_tts_model_provider_type(self):
        """Test that TextToSpeechModel returns correct provider type."""

        class TestTTSModel(TextToSpeechModel):
            def generate_speech(self, text, voice, **kwargs):
                pass

            async def agenerate_speech(self, text, voice, **kwargs):
                pass

            def _get_default_model(self):
                return "test-model"

            @property
            def provider(self):
                return "test"

            @property
            def models(self):
                return []

            @property
            def available_voices(self):
                return {}

        model = TestTTSModel(model_name="test")
        assert model._get_provider_type() == "text_to_speech"


class TestHTTPClientTimeoutIntegration:
    """Test HTTP client timeout configuration."""

    def test_http_client_timeout_configuration(self):
        """Test that HTTP clients are created with correct timeout values."""

        class TestLanguageModel(LanguageModel):
            def __init__(self, **kwargs):
                self.model_name = kwargs.get("model_name", "test")
                self.api_key = "test-key"
                self.base_url = "https://api.test.com"
                self.config = {"timeout": 120.0}  # Set config before calling super
                super().__post_init__()
                self._create_http_clients()

            def chat_complete(self, messages, **kwargs):
                pass

            async def achat_complete(self, messages, **kwargs):
                pass

            def _get_default_model(self):
                return "test-model"

            @property
            def provider(self):
                return "test"

            @property
            def models(self):
                return []

            def to_langchain(self):
                pass

        model = TestLanguageModel()

        # Check that HTTP clients were created
        assert hasattr(model, 'client')
        assert hasattr(model, 'async_client')

        # Check that clients have correct timeout
        assert model.client.timeout.read == 120.0
        assert model.async_client.timeout.read == 120.0

    def test_default_timeout_http_clients(self):
        """Test that HTTP clients use default timeouts when none specified."""

        class TestEmbeddingModel(EmbeddingModel):
            def __init__(self, **kwargs):
                self.model_name = kwargs.get("model_name", "test")
                self.api_key = "test-key"
                self.base_url = "https://api.test.com"
                self._config = {}
                super().__post_init__()
                self._create_http_clients()

            def embed(self, texts, **kwargs):
                pass

            async def aembed(self, texts, **kwargs):
                pass

            def _get_default_model(self):
                return "test-model"

            @property
            def provider(self):
                return "test"

            @property
            def models(self):
                return []

        model = TestEmbeddingModel()

        # Check that HTTP clients use embedding default (60 seconds)
        assert model.client.timeout.read == 60.0
        assert model.async_client.timeout.read == 60.0

    def test_environment_variable_timeout_http_clients(self):
        """Test that HTTP clients use environment variable timeouts."""
        # Set environment variable
        os.environ["ESPERANTO_LLM_TIMEOUT"] = "90.0"

        try:
            class TestLanguageModel(LanguageModel):
                def __init__(self, **kwargs):
                    self.model_name = kwargs.get("model_name", "test")
                    self.api_key = "test-key"
                    self.base_url = "https://api.test.com"
                    self._config = {}
                    super().__post_init__()
                    self._create_http_clients()

                def chat_complete(self, messages, **kwargs):
                    pass

                async def achat_complete(self, messages, **kwargs):
                    pass

                def _get_default_model(self):
                    return "test-model"

                @property
                def provider(self):
                    return "test"

                @property
                def models(self):
                    return []

                def to_langchain(self):
                    pass

            model = TestLanguageModel()

            # Check that HTTP clients use environment variable timeout
            assert model.client.timeout.read == 90.0
            assert model.async_client.timeout.read == 90.0

        finally:
            # Clean up environment variable
            os.environ.pop("ESPERANTO_LLM_TIMEOUT", None)


class TestRealProviderTimeoutIntegration:
    """Test timeout integration with real providers (mocked to avoid API keys)."""

    @patch("httpx.Client")
    @patch("httpx.AsyncClient")
    def test_openai_language_model_timeout_integration(self, mock_async_client, mock_client):
        """Test OpenAI language model uses timeout configuration."""
        try:
            from esperanto.providers.llm.openai import OpenAILanguageModel

            # Mock environment variable for API key
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                model = OpenAILanguageModel(
                    model_name="gpt-3.5-turbo",
                    config={"timeout": 150.0}
                )

                # Verify httpx clients were called with correct timeout
                mock_client.assert_called_once_with(timeout=150.0)
                mock_async_client.assert_called_once_with(timeout=150.0)

        except ImportError:
            pytest.skip("OpenAI provider not available")

    @patch("httpx.Client")
    @patch("httpx.AsyncClient")
    def test_openai_embedding_model_timeout_integration(self, mock_async_client, mock_client):
        """Test OpenAI embedding model uses timeout configuration."""
        try:
            from esperanto.providers.embedding.openai import OpenAIEmbeddingModel

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                model = OpenAIEmbeddingModel(
                    model_name="text-embedding-3-small",
                    config={"timeout": 180.0}
                )

                mock_client.assert_called_once_with(timeout=180.0)
                mock_async_client.assert_called_once_with(timeout=180.0)

        except ImportError:
            pytest.skip("OpenAI provider not available")

    @patch("httpx.Client")
    @patch("httpx.AsyncClient")
    def test_stt_model_timeout_integration(self, mock_async_client, mock_client):
        """Test STT model uses timeout configuration."""
        try:
            from esperanto.providers.stt.openai import OpenAISpeechToTextModel

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                model = OpenAISpeechToTextModel(
                    model_name="whisper-1",
                    timeout=450.0  # STT uses direct parameter
                )

                mock_client.assert_called_once_with(timeout=450.0)
                mock_async_client.assert_called_once_with(timeout=450.0)

        except ImportError:
            pytest.skip("OpenAI STT provider not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])