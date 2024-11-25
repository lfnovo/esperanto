"""Tests for Gemini language model."""

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from esperanto.providers.llm.gemini import GeminiLanguageModel


def test_initialization():
    """Test initialization of Gemini language model."""
    model = GeminiLanguageModel(
        model_name="gemini-pro",
        config={"api_key": SecretStr("test-key")},
    )

    assert model.model_name == "gemini-pro"
    assert model.provider == "gemini"
    assert model.temperature == 0.7
    assert model.top_p == 0.9
    assert model.max_tokens == 850
    assert isinstance(model._api_key, SecretStr)
    assert model._api_key.get_secret_value() == "test-key"


def test_initialization_empty_model_name():
    """Test initialization with empty model name."""
    with pytest.raises(ValueError, match="model_name must be specified"):
        GeminiLanguageModel(
            model_name="",
            config={"api_key": SecretStr("test-key")},
        )


def test_validate_config():
    """Test configuration validation."""
    model = GeminiLanguageModel(
        model_name="gemini-pro",
        config={"api_key": SecretStr("test-key")},
    )
    model.validate_config()


def test_validate_config_invalid_temperature():
    """Test configuration validation with invalid temperature."""
    model = GeminiLanguageModel(
        model_name="gemini-pro",
        config={
            "api_key": SecretStr("test-key"),
            "temperature": 1.5,
        },
    )
    with pytest.raises(ValueError, match="temperature must be between 0 and 1"):
        model.validate_config()


def test_validate_config_invalid_top_p():
    """Test configuration validation with invalid top_p."""
    model = GeminiLanguageModel(
        model_name="gemini-pro",
        config={
            "api_key": SecretStr("test-key"),
            "top_p": 1.5,
        },
    )
    with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
        model.validate_config()


def test_validate_config_invalid_max_tokens():
    """Test configuration validation with invalid max_tokens."""
    model = GeminiLanguageModel(
        model_name="gemini-pro",
        config={
            "api_key": SecretStr("test-key"),
            "max_tokens": "invalid",
        },
    )
    with pytest.raises(ValueError, match="max_tokens must be an integer"):
        model.validate_config()


# def test_to_langchain():
#     """Test conversion to LangChain model."""
#     model = GeminiLanguageModel(
#         model_name="gemini-pro",
#         config={"api_key": SecretStr("test-key")},
#     )
#     langchain_model = model.to_langchain()

#     assert isinstance(langchain_model, ChatGoogleGenerativeAI)
#     assert langchain_model.model == "models/gemini-pro"
#     assert langchain_model.google_api_key.get_secret_value() == "test-key"
