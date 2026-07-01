"""Tests for the PayPerQ (PPQ) provider classes (embedding, STT, TTS).

The PPQ language model is covered by the profile system (see
tests/providers/llm/test_profiles.py). These tests cover the dedicated
embedding/STT/TTS classes, which subclass the OpenAI-compatible providers
and inject PPQ defaults (base URL, PPQ_API_KEY, provider name).
"""

import os
from unittest.mock import patch

import pytest

from esperanto.factory import AIFactory
from esperanto.providers.embedding.ppq import PPQEmbeddingModel
from esperanto.providers.stt.ppq import PPQSpeechToTextModel
from esperanto.providers.tts.ppq import PPQTextToSpeechModel

DEFAULT_BASE_URL = "https://api.ppq.ai/v1"


class TestPPQEmbedding:
    def test_defaults(self):
        model = PPQEmbeddingModel(api_key="test-key")
        assert model.provider == "ppq"
        assert model.base_url == DEFAULT_BASE_URL
        assert model._get_default_model() == "openai/text-embedding-3-small"

    def test_env_var_api_key(self):
        with patch.dict(os.environ, {"PPQ_API_KEY": "env-key"}, clear=False):
            model = PPQEmbeddingModel()
            assert model.api_key == "env-key"

    def test_env_var_base_url(self):
        with patch.dict(
            os.environ,
            {"PPQ_API_KEY": "key", "PPQ_BASE_URL": "https://proxy.ppq.ai/v1"},
            clear=False,
        ):
            model = PPQEmbeddingModel()
            assert model.base_url == "https://proxy.ppq.ai/v1"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="PayPerQ API key not found"):
                PPQEmbeddingModel()

    def test_factory_creates_ppq(self):
        model = AIFactory.create_embedding(
            "ppq", "openai/text-embedding-3-large", config={"api_key": "test-key"}
        )
        assert model.provider == "ppq"
        assert model.base_url == DEFAULT_BASE_URL


class TestPPQSpeechToText:
    def test_defaults(self):
        model = PPQSpeechToTextModel(api_key="test-key")
        assert model.provider == "ppq"
        assert model.base_url == DEFAULT_BASE_URL
        assert model._get_default_model() == "nova-3"

    def test_env_var_api_key(self):
        with patch.dict(os.environ, {"PPQ_API_KEY": "env-key"}, clear=False):
            model = PPQSpeechToTextModel()
            assert model.api_key == "env-key"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="PayPerQ API key not found"):
                PPQSpeechToTextModel()

    def test_factory_creates_ppq(self):
        model = AIFactory.create_speech_to_text(
            "ppq", "nova-2", config={"api_key": "test-key"}
        )
        assert model.provider == "ppq"
        assert model.base_url == DEFAULT_BASE_URL


class TestPPQTextToSpeech:
    def test_defaults(self):
        model = PPQTextToSpeechModel(api_key="test-key")
        assert model.provider == "ppq"
        assert model.base_url == DEFAULT_BASE_URL
        assert model._get_default_model() == "deepgram_aura_2"

    def test_env_var_api_key(self):
        with patch.dict(os.environ, {"PPQ_API_KEY": "env-key"}, clear=False):
            model = PPQTextToSpeechModel()
            assert model.api_key == "env-key"

    def test_env_var_base_url(self):
        with patch.dict(
            os.environ,
            {"PPQ_API_KEY": "key", "PPQ_BASE_URL": "https://proxy.ppq.ai/v1"},
            clear=False,
        ):
            model = PPQTextToSpeechModel()
            assert model.base_url == "https://proxy.ppq.ai/v1"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="PayPerQ API key not found"):
                PPQTextToSpeechModel()

    def test_factory_creates_ppq(self):
        model = AIFactory.create_text_to_speech(
            "ppq", "eleven_multilingual_v2", config={"api_key": "test-key"}
        )
        assert model.provider == "ppq"
        assert model.base_url == DEFAULT_BASE_URL
