#!/usr/bin/env python3
"""Tests for deprecation warnings on .models property."""

import warnings
from unittest.mock import MagicMock, patch
import pytest

from esperanto.common_types import Model
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.reranker.base import RerankerModel
from esperanto.providers.stt.base import SpeechToTextModel
from esperanto.providers.tts.base import TextToSpeechModel


class TestLanguageModelDeprecation:
    """Test deprecation warnings for LanguageModel.models property."""

    def test_models_property_deprecated(self):
        """Test that .models property emits deprecation warning."""

        # Create a test implementation
        class TestLLM(LanguageModel):
            def _get_models(self):
                return [Model(id="test-model", owned_by="test")]

            @property
            def provider(self):
                return "test"

            def _get_default_model(self):
                return "test-model"

            def chat_complete(self, messages, **kwargs):
                pass

            async def achat_complete(self, messages, **kwargs):
                pass

            def to_langchain(self):
                pass

        model = TestLLM(model_name="test")

        # Check that accessing .models raises a deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = model.models

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "version 3.0" in str(w[0].message)
            assert "AIFactory.get_provider_models" in str(w[0].message)

    def test_models_property_still_works(self):
        """Test that .models property still returns correct data."""

        class TestLLM(LanguageModel):
            def _get_models(self):
                return [
                    Model(id="model1", owned_by="test"),
                    Model(id="model2", owned_by="test"),
                ]

            @property
            def provider(self):
                return "test"

            def _get_default_model(self):
                return "test-model"

            def chat_complete(self, messages, **kwargs):
                pass

            async def achat_complete(self, messages, **kwargs):
                pass

            def to_langchain(self):
                pass

        model = TestLLM(model_name="test")

        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = model.models

            assert len(models) == 2
            assert models[0].id == "model1"
            assert models[1].id == "model2"


class TestEmbeddingModelDeprecation:
    """Test deprecation warnings for EmbeddingModel.models property."""

    def test_models_property_deprecated(self):
        """Test that .models property emits deprecation warning."""

        class TestEmbedding(EmbeddingModel):
            def _get_models(self):
                return [Model(id="test-embedding", owned_by="test")]

            @property
            def provider(self):
                return "test"

            def _get_default_model(self):
                return "test-model"

            def embed(self, texts, **kwargs):
                pass

            async def aembed(self, texts, **kwargs):
                pass

        model = TestEmbedding(model_name="test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = model.models

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


class TestRerankerModelDeprecation:
    """Test deprecation warnings for RerankerModel.models property."""

    def test_models_property_deprecated(self):
        """Test that .models property emits deprecation warning."""

        class TestReranker(RerankerModel):
            def _get_models(self):
                return [Model(id="test-reranker", owned_by="test")]

            @property
            def provider(self):
                return "test"

            def _get_default_model(self):
                return "test-model"

            def rerank(self, query, documents, **kwargs):
                pass

            async def arerank(self, query, documents, **kwargs):
                pass

            def to_langchain(self):
                pass

        model = TestReranker(model_name="test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = model.models

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


class TestSTTModelDeprecation:
    """Test deprecation warnings for SpeechToTextModel.models property."""

    def test_models_property_deprecated(self):
        """Test that .models property emits deprecation warning."""

        class TestSTT(SpeechToTextModel):
            def _get_models(self):
                return [Model(id="test-stt", owned_by="test")]

            @property
            def provider(self):
                return "test"

            def _get_default_model(self):
                return "test-model"

            def transcribe(self, audio_file, **kwargs):
                pass

            async def atranscribe(self, audio_file, **kwargs):
                pass

        model = TestSTT(model_name="test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = model.models

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


class TestTTSModelDeprecation:
    """Test deprecation warnings for TextToSpeechModel.models property."""

    def test_models_property_deprecated(self):
        """Test that .models property emits deprecation warning."""

        class TestTTS(TextToSpeechModel):
            def _get_models(self):
                return [Model(id="test-tts", owned_by="test")]

            @property
            def provider(self):
                return "test"

            @property
            def available_voices(self):
                return {}

            def _get_default_model(self):
                return "test-model"

            def generate_speech(self, text, voice, **kwargs):
                pass

            async def agenerate_speech(self, text, voice, **kwargs):
                pass

        model = TestTTS(model_name="test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = model.models

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


class TestDeprecationMessage:
    """Test the specific content of deprecation messages."""

    def test_deprecation_message_includes_provider_name(self):
        """Test that deprecation message includes the provider name."""

        class TestLLM(LanguageModel):
            def _get_models(self):
                return []

            @property
            def provider(self):
                return "custom-provider"

            def _get_default_model(self):
                return "test-model"

            def chat_complete(self, messages, **kwargs):
                pass

            async def achat_complete(self, messages, **kwargs):
                pass

            def to_langchain(self):
                pass

        model = TestLLM(model_name="test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = model.models

            message = str(w[0].message)
            assert "custom-provider" in message
            assert "AIFactory.get_provider_models('custom-provider')" in message

    def test_deprecation_message_recommends_alternative(self):
        """Test that deprecation message recommends AIFactory method."""

        class TestLLM(LanguageModel):
            def _get_models(self):
                return []

            @property
            def provider(self):
                return "test"

            def _get_default_model(self):
                return "test-model"

            def chat_complete(self, messages, **kwargs):
                pass

            async def achat_complete(self, messages, **kwargs):
                pass

            def to_langchain(self):
                pass

        model = TestLLM(model_name="test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = model.models

            message = str(w[0].message)
            assert "AIFactory.get_provider_models" in message
            assert "static" in message.lower() or "without creating provider instances" in message


class TestBackwardCompatibility:
    """Test that deprecated functionality still works for backward compatibility."""

    def test_old_code_continues_to_work(self):
        """Test that existing code using .models still works (with warnings)."""

        class TestLLM(LanguageModel):
            def _get_models(self):
                return [
                    Model(id="gpt-4", owned_by="openai"),
                    Model(id="gpt-3.5-turbo", owned_by="openai"),
                ]

            @property
            def provider(self):
                return "openai"

            def _get_default_model(self):
                return "gpt-4"

            def chat_complete(self, messages, **kwargs):
                pass

            async def achat_complete(self, messages, **kwargs):
                pass

            def to_langchain(self):
                pass

        model = TestLLM(model_name="gpt-4")

        # Old code pattern that should still work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # This is how users currently access models
            available_models = model.models
            model_ids = [m.id for m in available_models]

            assert "gpt-4" in model_ids
            assert "gpt-3.5-turbo" in model_ids
            assert len(model_ids) == 2


class TestWarningStackLevel:
    """Test that warnings point to the correct code location."""

    def test_warning_points_to_user_code(self):
        """Test that warning stacklevel points to user's code, not library internals."""

        class TestLLM(LanguageModel):
            def _get_models(self):
                return []

            @property
            def provider(self):
                return "test"

            def _get_default_model(self):
                return "test-model"

            def chat_complete(self, messages, **kwargs):
                pass

            async def achat_complete(self, messages, **kwargs):
                pass

            def to_langchain(self):
                pass

        model = TestLLM(model_name="test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            models = model.models  # This line should be in the warning

            assert len(w) == 1
            # The warning should have been raised from this test file, not from base.py
            assert "test_deprecation_warnings.py" in w[0].filename


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
