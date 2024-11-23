"""Tests for the Ollama language model provider."""

import os

import pytest
from langchain_ollama.chat_models import ChatOllama

from esperanto.providers.llm.ollama import OllamaLanguageModel


@pytest.fixture
def model():
    """Create a test Ollama model instance."""
    return OllamaLanguageModel(
        model_name="llama2",
        config={
            "temperature": 0.7,
            "max_tokens": 500,
            "streaming": False,
            "top_p": 0.8,
            "base_url": "http://localhost:11434",
        },
    )


class TestOllamaLanguageModel:
    """Test suite for the Ollama language model."""

    def test_initialization(self):
        """Test model initialization with custom config."""
        model = OllamaLanguageModel(
            model_name="llama2",
            config={
                "temperature": 0.7,
                "max_tokens": 500,
                "streaming": False,
                "top_p": 0.8,
                "base_url": "http://localhost:11434",
            },
        )
        assert model.model_name == "llama2"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8
        assert model.base_url == "http://localhost:11434"

    def test_initialization_default_values(self):
        """Test model initialization with default values."""
        model = OllamaLanguageModel(model_name="llama2")
        assert model.model_name == "llama2"
        assert model.temperature == 1.0
        assert model.max_tokens == 850
        assert model.streaming is False
        assert model.top_p == 0.9
        # base_url defaults to OLLAMA_API_BASE env var or http://localhost:11434
        assert model.base_url in ["http://localhost:11434", "http://10.20.30.20:11434"]

    def test_initialization_env_base_url(self, monkeypatch):
        """Test model initialization with base_url from environment variable."""
        monkeypatch.setenv("OLLAMA_API_BASE", "http://custom:11434")
        model = OllamaLanguageModel(model_name="llama2")
        assert model.base_url == "http://custom:11434"

    def test_to_langchain(self, model):
        """Test conversion to LangChain model."""
        langchain_model = model.to_langchain()
        assert isinstance(langchain_model, ChatOllama)
        assert langchain_model.model == "llama2"
        assert langchain_model.temperature == 0.7
        assert langchain_model.base_url == "http://localhost:11434"

    def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exception

    def test_validate_config_failure_no_model_name(self):
        """Test config validation with missing model name."""
        model = OllamaLanguageModel(model_name="")
        with pytest.raises(ValueError, match="model_name must be specified"):
            model.validate_config()

    def test_validate_config_failure_no_base_url(self, monkeypatch):
        """Test config validation with missing base_url."""
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
        model = OllamaLanguageModel(
            model_name="llama2",
            config={"base_url": ""},
        )
        model.base_url = ""
        with pytest.raises(ValueError, match="base_url must be specified"):
            model.validate_config()

    def test_provider_name(self, model):
        """Test provider name."""
        assert model.provider == "ollama"

    def test_model_parameters(self, model):
        """Test model parameters are correctly set."""
        assert model.model_name == "llama2"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8
        assert model.base_url == "http://localhost:11434"
