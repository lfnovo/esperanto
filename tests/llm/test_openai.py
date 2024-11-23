"""Tests for OpenAI language model implementation."""

import os
import pytest
from langchain_openai.chat_models import ChatOpenAI
from pydantic import SecretStr

from esperanto.providers.llm import OpenAILanguageModel


@pytest.fixture
def model():
    """Create a test model instance."""
    return OpenAILanguageModel(
        model_name="gpt-3.5-turbo",
        config={
            "api_key": SecretStr("test-key"),
            "temperature": 0.7,
            "max_tokens": 500,
            "streaming": False,
            "top_p": 0.8,
            "json": True,
        }
    )


class TestOpenAILanguageModel:
    """Test suite for OpenAI language model."""

    async def test_initialization(self, model):
        """Test model initialization with valid config."""
        assert model.model_name == "gpt-3.5-turbo"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8
        assert model.json_mode is True

    async def test_initialization_default_values(self):
        """Test model initialization with default values."""
        model = OpenAILanguageModel(
            model_name="gpt-3.5-turbo",
            config={"api_key": SecretStr("test-key")}
        )
        assert model.model_name == "gpt-3.5-turbo"
        assert model.temperature == 1.0
        assert model.max_tokens == 850
        assert model.streaming is True
        assert model.top_p == 0.9
        assert model.json_mode is False

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified"):
            OpenAILanguageModel(model_name="")

    async def test_to_langchain(self, model):
        """Test conversion to LangChain model."""
        langchain_model = model.to_langchain()
        assert isinstance(langchain_model, ChatOpenAI)
        assert langchain_model.model_name == "gpt-3.5-turbo"
        assert langchain_model.max_tokens == 500
        assert langchain_model.temperature == 0.7
        assert langchain_model.streaming is False
        assert langchain_model.top_p == 0.8
        assert langchain_model.model_kwargs == {
            "response_format": {"type": "json"}
        }

    async def test_to_langchain_without_json(self):
        """Test conversion to LangChain model without JSON mode."""
        model = OpenAILanguageModel(
            model_name="gpt-3.5-turbo",
            config={
                "api_key": SecretStr("test-key"),
                "json": False
            }
        )
        langchain_model = model.to_langchain()
        assert langchain_model.model_kwargs == {
            "response_format": None
        }

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exception

    async def test_validate_config_failure(self):
        """Test config validation with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified for OpenAI language model"):
            OpenAILanguageModel(model_name="")

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "openai"

    async def test_model_parameters(self, model):
        """Test model parameters are correctly set."""
        assert model.model_name == "gpt-3.5-turbo"
        assert model.temperature == 0.7
        assert model.max_tokens == 500
        assert model.streaming is False
        assert model.top_p == 0.8
        assert model.json_mode is True

    async def test_custom_api_key(self, mock_env_vars):
        """Test initialization with custom API key."""
        model = OpenAILanguageModel(
            model_name="gpt-3.5-turbo",
            config={"api_key": SecretStr("custom-key")}
        )
        assert model.api_key.get_secret_value() == "custom-key"

    async def test_custom_base_url(self):
        """Test initialization with custom base URL."""
        model = OpenAILanguageModel(
            model_name="gpt-3.5-turbo",
            config={
                "api_key": SecretStr("test-key"),
                "openai_api_base": "https://custom.openai.api"
            }
        )
        assert model.base_url == "https://custom.openai.api"

    async def test_custom_organization(self):
        """Test initialization with custom organization."""
        model = OpenAILanguageModel(
            model_name="gpt-3.5-turbo",
            config={
                "api_key": SecretStr("test-key"),
                "organization": "org-123"
            }
        )
        assert model.organization == "org-123"
