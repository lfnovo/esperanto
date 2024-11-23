"""Tests for the OpenRouter language model provider."""

import os
import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from esperanto.providers.llm.openrouter import OpenRouterLanguageModel


@pytest.fixture
def model():
    """Create a test OpenRouter model instance."""
    return OpenRouterLanguageModel(
        model_name="mistralai/mistral-7b",
        config={
            "temperature": 0.7,
            "max_tokens": 500,
            "streaming": False,
            "top_p": 0.8,
        },
    )


class TestOpenRouterLanguageModel:
    """Test suite for the OpenRouter language model."""

    def test_initialization(self):
        """Test model initialization with custom config."""
        model = OpenRouterLanguageModel(
            model_name="mistralai/mistral-7b",
            config={
                "temperature": 0.7,
                "max_tokens": 500,
                "streaming": False,
                "top_p": 0.8,
            },
        )
        assert model.model_name == "mistralai/mistral-7b"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8

    def test_initialization_default_values(self):
        """Test model initialization with default values."""
        model = OpenRouterLanguageModel(model_name="mistralai/mistral-7b")
        assert model.model_name == "mistralai/mistral-7b"
        assert model.temperature == 1.0
        assert model.max_tokens == 850
        assert model.streaming is True
        assert model.top_p == 0.9

    def test_to_langchain(self, model, monkeypatch):
        """Test conversion to LangChain model."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://custom.openrouter.ai/api/v1")

        langchain_model = model.to_langchain()
        assert isinstance(langchain_model, ChatOpenAI)
        assert langchain_model.model_name == "mistralai/mistral-7b"
        assert langchain_model.temperature == 0.7
        assert langchain_model.max_tokens == 500
        assert langchain_model.streaming is False
        assert langchain_model.top_p == 0.8
        assert langchain_model.openai_api_base == "https://custom.openrouter.ai/api/v1"
        assert langchain_model.openai_api_key == SecretStr("test-api-key")

    def test_to_langchain_default_values(self, monkeypatch):
        """Test conversion to LangChain model with default values."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

        model = OpenRouterLanguageModel(model_name="mistralai/mistral-7b")
        langchain_model = model.to_langchain()

        assert langchain_model.openai_api_base == "https://openrouter.ai/api/v1"
        assert langchain_model.openai_api_key == SecretStr("openrouter")

    def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exception

    def test_validate_config_failure_no_model_name(self):
        """Test config validation with missing model name."""
        model = OpenRouterLanguageModel(model_name="")
        with pytest.raises(ValueError, match="model_name must be specified"):
            model.validate_config()

    def test_provider_name(self, model):
        """Test provider name."""
        assert model.provider == "openrouter"

    def test_model_parameters(self, model):
        """Test model parameters are correctly set."""
        assert model.model_name == "mistralai/mistral-7b"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8
