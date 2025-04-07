"""Tests for the Perplexity AI language model provider."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from esperanto.providers.llm.perplexity import PerplexityLanguageModel


@pytest.fixture
def perplexity_provider():
    """Fixture for PerplexityLanguageModel."""
    # Set dummy API key for testing
    os.environ["PERPLEXITY_API_KEY"] = "test_api_key"
    provider = PerplexityLanguageModel(model_name="llama-3-sonar-large-32k-online")
    # Clean up env var after test
    yield provider
    del os.environ["PERPLEXITY_API_KEY"]


@pytest.fixture
def mock_openai_client():
    """Fixture for mocking the OpenAI client."""
    with patch("esperanto.providers.llm.perplexity.OpenAI") as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_async_openai_client():
    """Fixture for mocking the AsyncOpenAI client."""
    with patch(
        "esperanto.providers.llm.perplexity.AsyncOpenAI"
    ) as mock_async_client_class:
        mock_instance = AsyncMock()
        mock_async_client_class.return_value = mock_instance
        yield mock_instance


def test_perplexity_provider_initialization(perplexity_provider):
    """Test initialization of PerplexityLanguageModel."""
    assert perplexity_provider.provider == "perplexity"
    assert (
        perplexity_provider.get_model_name() == "llama-3-sonar-large-32k-online"
    )  # Default model
    assert perplexity_provider.api_key == "test_api_key"
    assert perplexity_provider.base_url == "https://api.perplexity.ai/chat/completions"


def test_perplexity_provider_initialization_no_api_key():
    """Test initialization raises error if API key is missing."""
    if "PERPLEXITY_API_KEY" in os.environ:
        del os.environ["PERPLEXITY_API_KEY"]  # Ensure key is not set
    with pytest.raises(ValueError, match="Perplexity API key not found"):
        PerplexityLanguageModel(model_name="test-model")


def test_perplexity_get_api_kwargs(perplexity_provider):
    """Test _get_api_kwargs includes standard and perplexity-specific args."""
    perplexity_provider.temperature = 0.8
    perplexity_provider.max_tokens = 500
    perplexity_provider.search_domain_filter = ["example.com"]
    perplexity_provider.return_images = True
    perplexity_provider.web_search_options = {"search_context_size": "medium"}

    kwargs = perplexity_provider._get_api_kwargs()
    extra_body = perplexity_provider._get_perplexity_extra_body()

    # Test standard kwargs
    assert kwargs["temperature"] == 0.8
    assert kwargs["max_tokens"] == 500
    assert kwargs["stream"] is False  # Should be present and False by default
    # Ensure Perplexity params are NOT in standard kwargs
    assert "search_domain_filter" not in kwargs
    assert "return_images" not in kwargs
    assert "web_search_options" not in kwargs

    # Test extra_body kwargs
    assert extra_body["search_domain_filter"] == ["example.com"]
    assert extra_body["return_images"] is True
    assert "return_related_questions" not in extra_body  # Not set
    assert "search_recency_filter" not in extra_body  # Not set
    assert extra_body["web_search_options"] == {"search_context_size": "medium"}


def test_perplexity_get_api_kwargs_exclude_stream(perplexity_provider):
    """Test _get_api_kwargs excludes stream when requested."""
    perplexity_provider.streaming = True
    kwargs = perplexity_provider._get_api_kwargs(exclude_stream=True)
    assert "stream" not in kwargs


@pytest.mark.asyncio
async def test_perplexity_async_call(perplexity_provider, mock_async_openai_client):
    """Test the asynchronous call method."""
    # Pass messages as dicts, not LangChain objects
    messages = [{"role": "user", "content": "Hello"}]
    expected_response_text = "Hi there!"

    # Mock the async client's chat completions create method
    mock_completion = ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=expected_response_text, role="assistant"
                ),
            )
        ],
        created=1677652288,
        model=perplexity_provider.get_model_name(),
        object="chat.completion",
    )
    mock_async_openai_client.chat.completions.create = AsyncMock(
        return_value=mock_completion
    )
    perplexity_provider.async_client = mock_async_openai_client

    response = await perplexity_provider.achat_complete(
        messages
    )  # Use correct method name

    assert (
        response.choices[0].message.content == expected_response_text
    )  # Access content correctly
    mock_async_openai_client.chat.completions.create.assert_awaited_once()
    call_args = mock_async_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == perplexity_provider.get_model_name()
    assert call_args["messages"] == messages
    assert call_args["extra_body"] == {}  # Should be passed as empty dict


def test_perplexity_call(perplexity_provider, mock_openai_client):
    """Test the synchronous call method."""
    # Pass messages as dicts, not LangChain objects
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    expected_response_text = "Hello."

    # Mock the client's chat completions create method
    mock_completion = ChatCompletion(
        id="chatcmpl-test-sync",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=expected_response_text, role="assistant"
                ),
            )
        ],
        created=1677652289,
        model=perplexity_provider.get_model_name(),
        object="chat.completion",
    )
    mock_openai_client.chat.completions.create.return_value = mock_completion
    perplexity_provider.client = mock_openai_client

    response = perplexity_provider.chat_complete(messages)  # Use correct method name

    assert (
        response.choices[0].message.content == expected_response_text
    )  # Access content correctly
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == perplexity_provider.get_model_name()
    assert call_args["messages"] == messages
    assert call_args["extra_body"] == {}  # Should be passed as empty dict


def test_perplexity_call_with_extra_params(perplexity_provider, mock_openai_client):
    """Test synchronous call with extra Perplexity parameters."""
    perplexity_provider.search_domain_filter = ["test.com"]
    perplexity_provider.return_images = True
    messages = [{"role": "user", "content": "Hi"}]
    expected_response_text = "Hello."

    mock_completion = ChatCompletion(
        id="chatcmpl-test-sync-extra",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=expected_response_text, role="assistant"
                ),
            )
        ],
        created=1677652290,
        model=perplexity_provider.get_model_name(),
        object="chat.completion",
    )
    mock_openai_client.chat.completions.create.return_value = mock_completion
    perplexity_provider.client = mock_openai_client

    response = perplexity_provider.chat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == perplexity_provider.get_model_name()
    assert call_args["messages"] == messages
    assert call_args["extra_body"]["search_domain_filter"] == ["test.com"]
    assert call_args["extra_body"]["return_images"] is True


def test_perplexity_to_langchain(perplexity_provider):
    """Test conversion to LangChain model."""
    perplexity_provider.temperature = 0.7
    perplexity_provider.max_tokens = 100
    perplexity_provider.search_domain_filter = ["test.dev"]
    perplexity_provider.return_related_questions = True

    langchain_model = perplexity_provider.to_langchain()

    assert isinstance(langchain_model, ChatOpenAI)
    assert langchain_model.model_name == perplexity_provider.get_model_name()
    assert (
        langchain_model.openai_api_key.get_secret_value() == "test_api_key"
    )  # Compare secret value
    assert (
        langchain_model.openai_api_base == "https://api.perplexity.ai/chat/completions"
    )
    assert langchain_model.temperature == 0.7
    assert langchain_model.max_tokens == 100
    assert langchain_model.model_kwargs["search_domain_filter"] == ["test.dev"]
    assert langchain_model.model_kwargs["return_related_questions"] is True
    assert "return_images" not in langchain_model.model_kwargs  # Not set


def test_perplexity_to_langchain_structured(perplexity_provider):
    """Test conversion to LangChain model with structured output."""
    perplexity_provider.structured = {"type": "json_object"}
    langchain_model = perplexity_provider.to_langchain()

    assert langchain_model.model_kwargs["response_format"] == {"type": "json_object"}


def test_perplexity_models_property(perplexity_provider, mock_openai_client):
    """Test the models property (currently hardcoded)."""
    # Mocking the client is not strictly necessary here as it's hardcoded,
    # but good practice if it were to change to an API call.
    perplexity_provider.client = mock_openai_client
    models = perplexity_provider.models
    assert isinstance(models, list)
    assert len(models) > 5  # Check if it returns a reasonable number of models
    assert all(model.owned_by == "Perplexity" for model in models)
    assert "llama-3-sonar-large-32k-online" in [m.id for m in models]
