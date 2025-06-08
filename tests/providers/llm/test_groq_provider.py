import os
from unittest.mock import patch

import pytest

try:
    from langchain_groq import ChatGroq

    from esperanto.providers.llm.groq import GroqLanguageModel

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    pytestmark = pytest.mark.skip("Groq not installed")


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_provider_name(groq_model):
    assert groq_model.provider == "groq"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_client_properties(groq_model):
    """Test that client properties are properly initialized."""
    # Verify clients are not None
    assert groq_model.client is not None
    assert groq_model.async_client is not None

    # Verify clients have expected HTTP methods (httpx)
    assert hasattr(groq_model.client, "post")
    assert hasattr(groq_model.async_client, "post")
    
    # Verify API key is set
    assert groq_model.api_key == "test-key"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_initialization_with_api_key():
    model = GroqLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_initialization_with_env_var():
    with patch.dict(os.environ, {"GROQ_API_KEY": "env-test-key"}):
        model = GroqLanguageModel()
        assert model.api_key == "env-test-key"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Groq API key not found"):
            GroqLanguageModel()


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_chat_complete(groq_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = groq_model.chat_complete(messages)

    # Verify the client was called with correct parameters
    groq_model.client.post.assert_called_once()
    call_args = groq_model.client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.groq.com/openai/v1/chat/completions"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["messages"] == messages
    assert json_payload["model"] == "mixtral-8x7b-32768"
    assert json_payload["temperature"] == 1.0
    assert not json_payload["stream"]
    
    # Check response
    assert response.choices[0].message.content == "Test response"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
@pytest.mark.asyncio
async def test_achat_complete(groq_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = await groq_model.achat_complete(messages)

    # Verify the async client was called with correct parameters
    groq_model.async_client.post.assert_called_once()
    call_args = groq_model.async_client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.groq.com/openai/v1/chat/completions"
    
    # Check request payload
    json_payload = call_args[1]["json"]
    assert json_payload["messages"] == messages
    assert json_payload["model"] == "mixtral-8x7b-32768"
    assert json_payload["temperature"] == 1.0
    assert not json_payload["stream"]
    
    # Check response
    assert response.choices[0].message.content == "Test response"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_to_langchain(groq_model):
    langchain_model = groq_model.to_langchain()

    assert isinstance(langchain_model, ChatGroq)
    assert langchain_model.model_name == "mixtral-8x7b-32768"
    assert langchain_model.temperature == 1.0
    assert langchain_model.max_tokens == 850
    # assert langchain_model.model_kwargs["top_p"] == 0.9 # top_p is not stored in model_kwargs by default
    assert langchain_model.streaming == False
    assert langchain_model.groq_api_key.get_secret_value() == "test-key"


@pytest.mark.skipif(not HAS_GROQ, reason="Groq not installed")
def test_response_normalization(groq_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = groq_model.chat_complete(messages)

    assert response.id == "chatcmpl-123"
    assert response.created == 1677858242
    assert response.model == "mixtral-8x7b-32768"
    assert response.provider == "groq"
    assert len(response.choices) == 1

    choice = response.choices[0]
    assert choice.index == 0
    assert choice.message.content == "Test response"
    assert choice.message.role == "assistant"
    assert choice.finish_reason == "stop"

    assert response.usage.completion_tokens == 10
    assert response.usage.prompt_tokens == 8
    assert response.usage.total_tokens == 18
