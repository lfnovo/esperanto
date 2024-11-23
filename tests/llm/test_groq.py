"""Tests for Groq language model implementation."""

import pytest
from langchain_groq import ChatGroq

from esperanto.providers.llm import GroqLanguageModel


@pytest.fixture
def model():
    """Create a test model instance."""
    return GroqLanguageModel(
        model_name="mixtral-8x7b-v1",
        config={
            "temperature": 0.7,
            "max_tokens": 500,
            "streaming": False,
            "top_p": 0.8,
            "api_key": "test-api-key",  # Mock API key for testing
        }
    )


class TestGroqLanguageModel:
    """Test suite for Groq language model."""

    async def test_initialization(self, model):
        """Test model initialization with valid config."""
        assert model.model_name == "mixtral-8x7b-v1"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8

    async def test_initialization_default_values(self):
        """Test model initialization with default values."""
        model = GroqLanguageModel(model_name="mixtral-8x7b-v1")
        assert model.model_name == "mixtral-8x7b-v1"
        assert model.temperature == 1.0
        assert model.max_tokens == 850
        assert model.streaming is True
        assert model.top_p == 0.9

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified for Groq language model"):
            GroqLanguageModel(model_name="")

    async def test_to_langchain(self, model):
        """Test conversion to LangChain model."""
        langchain_model = model.to_langchain()
        assert isinstance(langchain_model, ChatGroq)
        assert langchain_model.model_name == "mixtral-8x7b-v1"
        assert langchain_model.max_tokens == 500
        assert langchain_model.temperature == 0.7
        assert langchain_model.streaming is False
        assert langchain_model.model_kwargs["top_p"] == 0.8

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exception

    async def test_validate_config_failure(self):
        """Test config validation with invalid config."""
        model = GroqLanguageModel(model_name="mixtral-8x7b-v1")
        model.model_name = ""  # Invalidate the model name after construction
        with pytest.raises(ValueError, match="model_name must be specified for Groq language model"):
            model.validate_config()

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "groq"

    async def test_model_parameters(self, model):
        """Test model parameters are correctly set."""
        assert model.model_name == "mixtral-8x7b-v1"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8
