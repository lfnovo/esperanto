import os
from unittest.mock import patch

import pytest

from esperanto.providers.llm.openai import OpenAILanguageModel


def test_provider_name(openai_model):
    assert openai_model.provider == "openai"

def test_initialization_with_api_key():
    model = OpenAILanguageModel(api_key="test-key")
    assert model.api_key == "test-key"

def test_initialization_with_env_var():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
        model = OpenAILanguageModel()
        assert model.api_key == "env-test-key"

def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAILanguageModel()

def test_chat_complete(openai_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    response = openai_model.chat_complete(messages)
    
    # Verify the client was called with correct parameters
    openai_model.client.chat.completions.create.assert_called_once()
    call_kwargs = openai_model.client.chat.completions.create.call_args[1]
    
    assert call_kwargs["messages"] == messages
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["temperature"] == 1.0
    assert not call_kwargs["stream"]
    
    # Verify response structure
    assert response.id.startswith("chatcmpl-")
    assert response.created > 0
    assert response.model == "gpt-4"
    assert response.provider == "openai"
    assert response.object == "chat.completion"
    
    # Verify choices
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.index == 0
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)
    assert choice.message.function_call is None
    assert choice.message.tool_calls is None
    
    # Verify usage
    assert response.usage.completion_tokens > 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens == response.usage.completion_tokens + response.usage.prompt_tokens

@pytest.mark.asyncio
async def test_achat_complete(openai_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    response = await openai_model.achat_complete(messages)
    
    # Verify the async client was called with correct parameters
    openai_model.async_client.chat.completions.create.assert_called_once()
    call_kwargs = openai_model.async_client.chat.completions.create.call_args[1]
    
    assert call_kwargs["messages"] == messages
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["temperature"] == 1.0
    assert not call_kwargs["stream"]
    
    # Verify response structure
    assert response.id.startswith("chatcmpl-")
    assert response.created > 0
    assert response.model == "gpt-4"
    assert response.provider == "openai"
    assert response.object == "chat.completion"
    
    # Verify choices
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.index == 0
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)
    assert choice.message.function_call is None
    assert choice.message.tool_calls is None
    
    # Verify usage
    assert response.usage.completion_tokens > 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens == response.usage.completion_tokens + response.usage.prompt_tokens

def test_json_structured_output(openai_model):
    openai_model.structured = "json_object"
    messages = [{"role": "user", "content": "Hello!"}]
    
    response = openai_model.chat_complete(messages)
    
    call_kwargs = openai_model.client.chat.completions.create.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}

@pytest.mark.asyncio
async def test_json_structured_output_async(openai_model):
    openai_model.structured = "json_object"
    messages = [{"role": "user", "content": "Hello!"}]
    
    response = await openai_model.achat_complete(messages)
    
    call_kwargs = openai_model.async_client.chat.completions.create.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}

def test_to_langchain(openai_model):
    # Test with structured output
    openai_model.structured = "json"
    langchain_model = openai_model.to_langchain()
    assert langchain_model.model_kwargs == {"response_format": {"type": "json_object"}}
    
    # Test model configuration
    assert langchain_model.model_name == "gpt-4"
    assert langchain_model.temperature == 1.0
    # Skip API key check since it's masked in SecretStr

def test_to_langchain_with_base_url(openai_model):
    openai_model.base_url = "https://custom.openai.com"
    langchain_model = openai_model.to_langchain()
    assert langchain_model.openai_api_base == "https://custom.openai.com"

def test_to_langchain_with_organization(openai_model):
    openai_model.organization = "test-org"
    langchain_model = openai_model.to_langchain()
    assert langchain_model.openai_organization == "test-org"
