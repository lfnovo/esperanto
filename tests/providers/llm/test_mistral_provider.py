import os
from unittest.mock import AsyncMock, Mock, patch
import pytest
from esperanto.providers.llm.mistral import MistralLanguageModel
from esperanto.common_types import ChatCompletion, Choice, Message, Usage

@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for Mistral API."""
    def create_response():
        return {
            "id": "cmpl-123",
            "object": "chat.completion",
            "created": 123,
            "model": "mistral-large-latest",
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
def mistral_model(mock_httpx_response):
    """Create MistralLanguageModel with mocked HTTP client."""
    model = MistralLanguageModel(api_key="test-key", model_name="mistral-large-latest")
    
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
    
    model.client = mock_client
    model.async_client = mock_async_client
    return model

def test_provider_name(mistral_model):
    assert mistral_model.provider == "mistral"

def test_initialization_with_api_key():
    model = MistralLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"

def test_initialization_with_env_var(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "env-test-key")
    model = MistralLanguageModel()
    assert model.api_key == "env-test-key"

def test_initialization_without_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Mistral API key not found"):
        MistralLanguageModel()

def test_chat_complete(mistral_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = mistral_model.chat_complete(messages)
    assert response.choices[0].message.content == "Hello!"

async def test_achat_complete(mistral_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = await mistral_model.achat_complete(messages)
    assert response.choices[0].message.content == "Hello!"

def test_to_langchain(mistral_model):
    # Only run if langchain_mistralai is installed
    try:
        lc = mistral_model.to_langchain()
        assert lc is not None
    except ImportError:
        pytest.skip("langchain_mistralai not installed")