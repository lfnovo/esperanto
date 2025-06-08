"""Tests for the Perplexity AI language model provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_openai import ChatOpenAI

from esperanto.providers.llm.perplexity import PerplexityLanguageModel
from esperanto.common_types import ChatCompletion, Choice, Message, Usage


@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for Perplexity API."""
    def create_response():
        return {
            "id": "cmpl-123",
            "object": "chat.completion",
            "created": 123,
            "model": "llama-3-sonar-large-32k-online",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5
            }
        }
    return create_response

@pytest.fixture
def perplexity_provider(mock_httpx_response):
    """Fixture for PerplexityLanguageModel."""
    # Set dummy API key for testing
    os.environ["PERPLEXITY_API_KEY"] = "test_api_key"
    provider = PerplexityLanguageModel(model_name="llama-3-sonar-large-32k-online")
    
    # Mock the HTTP clients
    mock_client = Mock()
    mock_async_client = AsyncMock()
    
    def mock_post(url, **kwargs):
        response_data = mock_httpx_response()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        return mock_response
    
    async def mock_async_post(url, **kwargs):
        response_data = mock_httpx_response()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        return mock_response
    
    mock_client.post = mock_post
    mock_async_client.post = mock_async_post
    
    provider.client = mock_client
    provider.async_client = mock_async_client
    
    # Clean up env var after test
    yield provider
    del os.environ["PERPLEXITY_API_KEY"]




def test_perplexity_provider_initialization(perplexity_provider):
    """Test initialization of PerplexityLanguageModel."""
    assert perplexity_provider.provider == "perplexity"
    assert (
        perplexity_provider.get_model_name() == "llama-3-sonar-large-32k-online"
    )  # Default model
    assert perplexity_provider.api_key == "test_api_key"
    assert perplexity_provider.base_url == "https://api.perplexity.ai"


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
    perplexity_params = perplexity_provider._get_perplexity_params()

    # Test standard kwargs
    assert kwargs["temperature"] == 0.8
    assert kwargs["max_tokens"] == 500
    # Ensure Perplexity params are NOT in standard kwargs
    assert "search_domain_filter" not in kwargs
    assert "return_images" not in kwargs
    assert "web_search_options" not in kwargs

    # Test perplexity params
    assert perplexity_params["search_domain_filter"] == ["example.com"]
    assert perplexity_params["return_images"] is True
    assert "return_related_questions" not in perplexity_params  # Not set
    assert "search_recency_filter" not in perplexity_params  # Not set
    assert perplexity_params["web_search_options"] == {"search_context_size": "medium"}


def test_perplexity_get_api_kwargs_exclude_stream(perplexity_provider):
    """Test _get_api_kwargs excludes stream when requested."""
    perplexity_provider.streaming = True
    kwargs = perplexity_provider._get_api_kwargs(exclude_stream=True)
    assert "stream" not in kwargs


@pytest.mark.asyncio
async def test_perplexity_async_call(perplexity_provider):
    """Test the asynchronous call method."""
    # Pass messages as dicts, not LangChain objects
    messages = [{"role": "user", "content": "Hello"}]
    expected_response_text = "Hello!"

    response = await perplexity_provider.achat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()


def test_perplexity_call(perplexity_provider):
    """Test the synchronous call method."""
    # Pass messages as dicts, not LangChain objects
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    expected_response_text = "Hello!"

    response = perplexity_provider.chat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()


def test_perplexity_call_with_extra_params(perplexity_provider):
    """Test synchronous call with extra Perplexity parameters."""
    perplexity_provider.search_domain_filter = ["test.com"]
    perplexity_provider.return_images = True
    messages = [{"role": "user", "content": "Hi"}]
    expected_response_text = "Hello!"

    response = perplexity_provider.chat_complete(messages)

    assert response.choices[0].message.content == expected_response_text
    assert response.model == perplexity_provider.get_model_name()
    
    # Test that perplexity params are available
    params = perplexity_provider._get_perplexity_params()
    assert params["search_domain_filter"] == ["test.com"]
    assert params["return_images"] is True


def test_perplexity_to_langchain(perplexity_provider):
    """Test conversion to LangChain model."""
    perplexity_provider.temperature = 0.7
    perplexity_provider.max_tokens = 100
    perplexity_provider.search_domain_filter = ["test.dev"]
    perplexity_provider.return_related_questions = True

    langchain_model = perplexity_provider.to_langchain()

    assert isinstance(langchain_model, ChatOpenAI)
    assert langchain_model.model_name == perplexity_provider.get_model_name()
    # Skip API key and base_url checks since they may be private attributes in LangChain
    assert langchain_model.temperature == 0.7
    assert langchain_model.max_tokens == 100
    assert langchain_model.model_kwargs["search_domain_filter"] == ["test.dev"]
    assert langchain_model.model_kwargs["return_related_questions"] is True
    assert "return_images" not in langchain_model.model_kwargs  # Not set


def test_perplexity_to_langchain_structured(perplexity_provider):
    """Test conversion to LangChain model with structured output."""
    perplexity_provider.structured = {"type": "json_object"}
    langchain_model = perplexity_provider.to_langchain()

    assert langchain_model.model_kwargs["response_format"] == {"type": "text"}


def test_perplexity_models_property(perplexity_provider):
    """Test the models property (currently hardcoded)."""
    models = perplexity_provider.models
    assert isinstance(models, list)
    assert len(models) > 5  # Check if it returns a reasonable number of models
    assert all(model.owned_by == "Perplexity" for model in models)
    # Check for some known models
    model_ids = [m.id for m in models]
    assert "sonar-pro" in model_ids
    assert "sonar" in model_ids
