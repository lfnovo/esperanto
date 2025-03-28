"""Tests for base model."""

from typing import AsyncGenerator, Dict, Generator, List, Union

from langchain_core.language_models.chat_models import BaseChatModel

from esperanto import LanguageModel
from esperanto.common_types import ChatCompletion, ChatCompletionChunk


class TestLanguageModel(LanguageModel):
    """Test implementation of LanguageModel."""

    @property
    def models(self):
        """List all available models for this provider."""
        return []

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "test"

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "test-default-model"

    def chat_complete(
        self, messages: List[Dict[str, str]], stream: bool = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        return ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            provider="test",
            response="test response",
            messages=messages,
            usage={"total_tokens": 10},
        )

    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: bool = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        return ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            provider="test",
            response="test response",
            messages=messages,
            usage={"total_tokens": 10},
        )

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        from langchain_core.chat_models.fake import FakeListChatModel

        return FakeListChatModel(responses=["test response"])


def test_client_properties():
    """Test that client properties are available in base class."""
    model = TestLanguageModel()
    assert hasattr(model, "client")
    assert hasattr(model, "async_client")
    assert model.client is None  # Default value should be None
    assert model.async_client is None  # Default value should be None


def test_language_model_config():
    """Test language model configuration initialization."""
    config = {
        "model_name": "test-model",
        "api_key": "test-key",
        "base_url": "test-url",
        "max_tokens": 1000,
        "temperature": 0.8,
        "streaming": True,
        "top_p": 0.95,
        "structured": {"format": "json"},
        "organization": "test-org",
    }
    model = TestLanguageModel(config=config)

    # Test that all config values are set correctly
    assert model.model_name == "test-model"
    assert model.api_key == "test-key"
    assert model.base_url == "test-url"
    assert model.max_tokens == 1000
    assert model.temperature == 0.8
    assert model.streaming is True
    assert model.top_p == 0.95
    assert model.structured == {"format": "json"}
    assert model.organization == "test-org"


def test_language_model_clean_config():
    """Test clean_config method."""
    model = TestLanguageModel(
        model_name="test-model",
        api_key="test-key",
        base_url=None,  # This should be excluded
        max_tokens=1000,
        temperature=0.8,
    )

    config = model.clean_config()
    assert "model_name" in config
    assert "api_key" in config
    assert "base_url" not in config
    assert config["max_tokens"] == 1000
    assert config["temperature"] == 0.8


def test_language_model_get_completion_kwargs():
    """Test get_completion_kwargs method."""
    model = TestLanguageModel(
        model_name="test-model",
        max_tokens=1000,
        temperature=0.8,
        top_p=0.95,
        streaming=True,
    )

    # Test without override
    kwargs = model.get_completion_kwargs()
    assert kwargs["max_tokens"] == 1000
    assert kwargs["temperature"] == 0.8
    assert kwargs["top_p"] == 0.95
    assert kwargs["streaming"] is True

    # Test with override
    override = {"max_tokens": 500, "temperature": 0.5}
    kwargs = model.get_completion_kwargs(override)
    assert kwargs["max_tokens"] == 500
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.95
    assert kwargs["streaming"] is True


def test_language_model_get_model_name():
    """Test get_model_name method."""
    # Test with model name in config
    model = TestLanguageModel(model_name="test-model")
    assert model.get_model_name() == "test-model"

    # Test fallback to default model
    model = TestLanguageModel()
    assert model.get_model_name() == "test-default-model"
