"""Tests for LiteLLM language model implementation."""

import pytest
from langchain_community.chat_models import ChatLiteLLM

from esperanto.providers.llm import LiteLLMLanguageModel


@pytest.fixture
def model():
    """Create a test model instance."""
    return LiteLLMLanguageModel(
        model_name="gpt-3.5-turbo",
        config={
            "temperature": 0.7,
            "max_tokens": 500,
            "streaming": False,
            "top_p": 0.8,
            "api_base": "https://api.litellm.ai",
        }
    )


class TestLiteLLMLanguageModel:
    """Test suite for LiteLLM language model."""

    async def test_initialization(self, model):
        """Test model initialization with valid config."""
        assert model.model_name == "gpt-3.5-turbo"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8
        assert model.api_base == "https://api.litellm.ai"

    async def test_initialization_default_values(self):
        """Test model initialization with default values."""
        model = LiteLLMLanguageModel(model_name="gpt-3.5-turbo")
        assert model.model_name == "gpt-3.5-turbo"
        assert model.temperature == 1.0
        assert model.max_tokens == 850
        assert model.streaming is True
        assert model.top_p == 0.9
        assert model.api_base is None

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified for LiteLLM language model"):
            LiteLLMLanguageModel(model_name="")

    async def test_to_langchain(self, model):
        """Test conversion to LangChain model."""
        langchain_model = model.to_langchain()
        assert isinstance(langchain_model, ChatLiteLLM)
        assert langchain_model.model == "gpt-3.5-turbo"
        assert langchain_model.max_tokens == 500
        assert langchain_model.temperature == 0.7
        assert langchain_model.streaming is False
        assert langchain_model.top_p == 0.8
        assert langchain_model.api_base == "https://api.litellm.ai"

    async def test_to_langchain_no_api_base(self):
        """Test conversion to LangChain model without api_base."""
        model = LiteLLMLanguageModel(
            model_name="gpt-3.5-turbo",
            config={
                "temperature": 0.7,
                "max_tokens": 500,
                "streaming": False,
                "top_p": 0.8,
            }
        )
        langchain_model = model.to_langchain()
        assert isinstance(langchain_model, ChatLiteLLM)
        assert langchain_model.model == "gpt-3.5-turbo"
        assert langchain_model.max_tokens == 500
        assert langchain_model.temperature == 0.7
        assert langchain_model.streaming is False
        assert langchain_model.top_p == 0.8
        assert langchain_model.api_base is None  # ChatLiteLLM sets None for api_base when not provided

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exception

    async def test_validate_config_failure(self):
        """Test config validation with invalid config."""
        model = LiteLLMLanguageModel(model_name="gpt-3.5-turbo")
        model.model_name = ""  # Invalidate the model name after construction
        with pytest.raises(ValueError, match="model_name must be specified for LiteLLM language model"):
            model.validate_config()

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "litellm"

    async def test_model_parameters(self, model):
        """Test model parameters are correctly set."""
        assert model.model_name == "gpt-3.5-turbo"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8
        assert model.api_base == "https://api.litellm.ai"
