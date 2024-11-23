"""Tests for the AI factory module."""


import pytest

from esperanto.factory import AIFactory
from esperanto.providers.llm import (
    AnthropicLanguageModel,
    GeminiLanguageModel,
    OpenAILanguageModel,
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
        assert "Provider 'invalid_provider' not supported for llm" in str(exc_info.value)

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
        with pytest.raises(TypeError) as exc_info:
            AIFactory.create_stt("openai")
        assert "missing 1 required positional argument: 'model_name'" in str(exc_info.value)

    def test_create_stt_with_invalid_provider(self):
        """Test creating STT with invalid provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            AIFactory.create_stt("invalid_provider", "model")
        assert "Provider 'invalid_provider' not supported for stt" in str(exc_info.value)

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
        assert "Provider 'invalid_provider' not supported for tts" in str(exc_info.value)

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
        config = {
            "api_key": "test-api-key",  # Mock API key for testing
        }
        model = AIFactory.create_llm(provider, model_name, config)
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
