"""Tests for Anthropic language model implementation."""

import pytest
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from esperanto.providers.llm import AnthropicLanguageModel


@pytest.fixture
def model():
    """Create a test model instance."""
    return AnthropicLanguageModel(
        model_name="claude-3-opus-20240229",
        config={"temperature": 0.7, "api_key": SecretStr("test-api-key")},
    )


class TestAnthropicLanguageModel:
    """Test suite for Anthropic language model."""

    async def test_initialization(self, model):
        """Test model initialization with valid config."""
        assert model.temperature == 0.7
        assert model.streaming is True
        assert model.top_p == 0.9
        assert model.max_tokens == 850

    async def test_to_langchain(self, model):
        """Test conversion to LangChain model."""
        langchain_model = model.to_langchain()
        assert isinstance(langchain_model, ChatAnthropic)
        assert langchain_model.model == "claude-3-opus-20240229"
        assert langchain_model.temperature == 0.7
        assert langchain_model.streaming is True
        assert langchain_model.top_p == 0.9
        assert langchain_model.max_tokens == 850

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exception

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "anthropic"

    async def test_model_parameters(self, model):
        """Test model parameters are correctly set."""
        assert model.temperature == 0.7
        assert model.streaming is True
        assert model.top_p == 0.9
        assert model.max_tokens == 850
