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
    openai_model.structured = {"type": "json_object"}
    messages = [{"role": "user", "content": "Hello!"}]
    
    response = openai_model.chat_complete(messages)
    
    call_kwargs = openai_model.client.chat.completions.create.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}

@pytest.mark.asyncio
async def test_json_structured_output_async(openai_model):
    openai_model.structured = {"type": "json_object"}
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

def test_o1_model_transformations(openai_model):
    """Test that o1 models correctly transform parameters and messages."""
    openai_model.model_name = "o1-model"  # Set model to o1
    openai_model._config["model_name"] = "o1-model"  # Update config as well
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Test synchronous completion
    response = openai_model.chat_complete(messages)
    call_kwargs = openai_model.client.chat.completions.create.call_args[1]
    
    # Check message transformation
    assert call_kwargs["messages"] == [
        {"role": "user", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Check parameter transformations
    assert call_kwargs["temperature"] == 1.0
    assert "top_p" not in call_kwargs
    assert "max_tokens" not in call_kwargs
    if "max_completion_tokens" in call_kwargs:
        assert call_kwargs["max_completion_tokens"] == openai_model.max_tokens

@pytest.mark.asyncio
async def test_o1_model_transformations_async(openai_model):
    """Test that o1 models correctly transform parameters and messages in async mode."""
    openai_model.model_name = "o1-model"  # Set model to o1
    openai_model._config["model_name"] = "o1-model"  # Update config as well
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Test async completion
    await openai_model.achat_complete(messages)
    call_kwargs = openai_model.async_client.chat.completions.create.call_args[1]
    
    # Check message transformation
    assert call_kwargs["messages"] == [
        {"role": "user", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Check parameter transformations
    assert call_kwargs["temperature"] == 1.0
    assert "top_p" not in call_kwargs
    assert "max_tokens" not in call_kwargs
    if "max_completion_tokens" in call_kwargs:
        assert call_kwargs["max_completion_tokens"] == openai_model.max_tokens

def test_non_o1_model_unchanged(openai_model):
    """Test that non-o1 models don't apply the special transformations."""
    openai_model.model_name = "gpt-4"  # Set model to non-o1
    openai_model._config["model_name"] = "gpt-4"  # Update config as well
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Test completion
    response = openai_model.chat_complete(messages)
    call_kwargs = openai_model.client.chat.completions.create.call_args[1]
    
    # Check messages remain unchanged
    assert call_kwargs["messages"] == messages
    
    # Check o1-specific parameters are not present
    assert "max_completion_tokens" not in call_kwargs
    
    # Test with o1 model
    openai_model.model_name = "o1-test"
    openai_model._config["model_name"] = "o1-test"
    
    response = openai_model.chat_complete(messages)
    o1_call_kwargs = openai_model.client.chat.completions.create.call_args[1]
    
    # Check that top_p is removed for o1 models
    assert "top_p" not in o1_call_kwargs
