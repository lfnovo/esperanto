"""Tests for the AI factory module."""

import pytest
from unittest.mock import patch

from esperanto.factory import AIFactory
from esperanto.providers.llm import (
    OpenAILanguageModel,
    AnthropicLanguageModel,
    GeminiLanguageModel,
)
from esperanto.providers.speech_to_text import OpenAISpeechToTextModel
from esperanto.providers.text_to_speech import ElevenLabsTextToSpeechModel


class TestAIFactory:
    """Test suite for AIFactory."""

    def test_create_llm_with_valid_provider(self):
        """Test creating LLM with valid provider."""
        model = AIFactory.create_llm("openai", "gpt-4")
        assert isinstance(model, OpenAILanguageModel)
        assert model.model_name == "gpt-4"

    def test_create_llm_with_config(self):
        """Test creating LLM with configuration."""
        config = {"temperature": 0.7, "max_tokens": 100}
        model = AIFactory.create_llm("openai", "gpt-4", config)
        assert isinstance(model, OpenAILanguageModel)
        assert model.config["temperature"] == 0.7
        assert model.config["max_tokens"] == 100

    def test_create_llm_with_invalid_provider(self):
        """Test creating LLM with invalid provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            AIFactory.create_llm("invalid_provider", "model")
        assert "Unsupported LLM provider" in str(exc_info.value)

    def test_create_llm_case_insensitive(self):
        """Test provider name is case insensitive."""
        model = AIFactory.create_llm("OPENAI", "gpt-4")
        assert isinstance(model, OpenAILanguageModel)

    def test_create_stt_with_valid_provider(self):
        """Test creating STT with valid provider."""
        model = AIFactory.create_stt("openai", "whisper-1")
        assert isinstance(model, OpenAISpeechToTextModel)
        assert model.model_name == "whisper-1"

    def test_create_stt_requires_model_name(self):
        """Test that STT requires a model name."""
        with pytest.raises(ValueError) as exc_info:
            AIFactory.create_stt("openai")
        assert "model_name must be specified" in str(exc_info.value)

    def test_create_stt_with_invalid_provider(self):
        """Test creating STT with invalid provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            AIFactory.create_stt("invalid_provider")
        assert "Unsupported speech-to-text provider" in str(exc_info.value)

    def test_create_tts_with_valid_provider(self):
        """Test creating TTS with valid provider."""
        model = AIFactory.create_tts("elevenlabs")
        assert isinstance(model, ElevenLabsTextToSpeechModel)

    def test_create_tts_with_config(self):
        """Test creating TTS with configuration."""
        config = {"voice": "Adam"}
        model = AIFactory.create_tts("elevenlabs", config=config)
        assert isinstance(model, ElevenLabsTextToSpeechModel)
        assert model.config["voice"] == "Adam"

    def test_create_tts_with_invalid_provider(self):
        """Test creating TTS with invalid provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            AIFactory.create_tts("invalid_provider")
        assert "Unsupported text-to-speech provider" in str(exc_info.value)

    @pytest.mark.parametrize(
        "provider,model_name,model_class",
        [
            ("openai", "gpt-4", OpenAILanguageModel),
            ("anthropic", "claude-3", AnthropicLanguageModel),
            ("gemini", "gemini-pro", GeminiLanguageModel),
        ],
    )
    def test_create_llm_all_providers(self, provider, model_name, model_class):
        """Test creating LLM with different providers."""
        model = AIFactory.create_llm(provider, model_name)
        assert isinstance(model, model_class)
        assert model.model_name == model_name

    def test_config_is_optional(self):
        """Test that config parameter is optional for all factory methods."""
        # LLM
        model = AIFactory.create_llm("openai", "gpt-4")
        assert isinstance(model, OpenAILanguageModel)
        assert isinstance(model.config, dict)

        # STT (requires model name)
        model = AIFactory.create_stt("openai", "whisper-1")
        assert isinstance(model, OpenAISpeechToTextModel)
        assert isinstance(model.config, dict)

        # TTS
        model = AIFactory.create_tts("elevenlabs")
        assert isinstance(model, ElevenLabsTextToSpeechModel)
        assert isinstance(model.config, dict)

    def test_empty_config_handling(self):
        """Test that empty config is handled correctly."""
        model = AIFactory.create_llm("openai", "gpt-4", {})
        assert isinstance(model.config, dict)
        assert len(model.config) == 0
