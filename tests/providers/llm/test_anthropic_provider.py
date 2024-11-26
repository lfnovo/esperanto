import io
import os
from unittest.mock import patch

import pytest
from esperanto.utils.logging import logger

from esperanto.providers.llm.anthropic import AnthropicLanguageModel


def test_provider_name(anthropic_model):
    assert anthropic_model.provider == "anthropic"

def test_initialization_with_api_key():
    model = AnthropicLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"

def test_initialization_with_env_var():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-test-key"}):
        model = AnthropicLanguageModel()
        assert model.api_key == "env-test-key"

def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Anthropic API key not found"):
            AnthropicLanguageModel()

def test_prepare_messages(anthropic_model):
    # Test with system message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    system, msgs = anthropic_model._prepare_messages(messages)
    assert system == "You are a helpful assistant."
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Hello!"
    
    # Test without system message
    messages = [{"role": "user", "content": "Hello!"}]
    system, msgs = anthropic_model._prepare_messages(messages)
    assert system is None
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Hello!"

def test_chat_complete(anthropic_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    anthropic_model.chat_complete(messages)
    
    # Verify the client was called with correct parameters
    anthropic_model.client.messages.create.assert_called_once()
    call_kwargs = anthropic_model.client.messages.create.call_args[1]
    
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello!"}]
    assert call_kwargs["system"] == "You are a helpful assistant."
    assert call_kwargs["model"] == "claude-3-opus-20240229"
    assert call_kwargs["max_tokens"] == 850
    assert call_kwargs["temperature"] == 0.7
    assert not call_kwargs["stream"]

@pytest.mark.asyncio
async def test_achat_complete(anthropic_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    await anthropic_model.achat_complete(messages)
    
    # Verify the async client was called with correct parameters
    anthropic_model.async_client.messages.create.assert_called_once()
    call_kwargs = anthropic_model.async_client.messages.create.call_args[1]
    
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello!"}]
    assert call_kwargs["system"] == "You are a helpful assistant."
    assert call_kwargs["model"] == "claude-3-opus-20240229"
    assert call_kwargs["max_tokens"] == 850
    assert call_kwargs["temperature"] == 0.7
    assert not call_kwargs["stream"]


    

def test_to_langchain(anthropic_model):
    # Test with structured output warning
    anthropic_model.structured = "json"
    
    langchain_model = anthropic_model.to_langchain()
    
    
    # Test model configuration
    assert langchain_model.model == "claude-3-opus-20240229"
    assert langchain_model.temperature == 0.7
    # Skip API key check since it's masked in SecretStr

def test_to_langchain_with_base_url(anthropic_model):
    anthropic_model.base_url = "https://custom.anthropic.com"
    langchain_model = anthropic_model.to_langchain()
    assert langchain_model.anthropic_api_url == "https://custom.anthropic.com"

@pytest.fixture
def mock_stream_events():
    """Create mock stream events for testing."""
    class MockEvent:
        def __init__(self, type_, index, delta):
            self.type = type_
            self.index = index
            self.delta = delta

    class MockDelta:
        def __init__(self, text=None, stop_reason=None):
            self.text = text
            self.stop_reason = stop_reason

    return [
        MockEvent("content_block_delta", 0, MockDelta(text="Hello")),
        MockEvent("content_block_delta", 1, MockDelta(text=" there")),
        MockEvent("message_delta", 2, MockDelta(stop_reason="end_turn")),
    ]

def test_chat_complete_streaming(anthropic_model, mock_stream_events):
    """Test streaming chat completion."""
    messages = [{"role": "user", "content": "Hello!"}]
    anthropic_model.client.messages.create.return_value = mock_stream_events
    
    # Test streaming
    generator = anthropic_model.chat_complete(messages, stream=True)
    chunks = list(generator)
    
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta["content"] == "Hello"
    assert chunks[1].choices[0].delta["content"] == " there"
    assert chunks[2].choices[0].finish_reason == "end_turn"

@pytest.mark.asyncio
async def test_achat_complete_streaming(anthropic_model, mock_stream_events):
    """Test async streaming chat completion."""
    messages = [{"role": "user", "content": "Hello!"}]
    
    # Mock async stream response
    class AsyncIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0
            
        def __aiter__(self):
            return self
            
        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item
    
    anthropic_model.async_client.messages.create.return_value = AsyncIterator(mock_stream_events)
    
    # Test streaming
    generator = await anthropic_model.achat_complete(messages, stream=True)
    chunks = []
    async for chunk in generator:
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta["content"] == "Hello"
    assert chunks[1].choices[0].delta["content"] == " there"
    assert chunks[2].choices[0].finish_reason == "end_turn"

def test_api_kwargs_handling(anthropic_model):
    """Test API kwargs handling."""
    # Test temperature clamping
    kwargs = anthropic_model._get_api_kwargs()
    assert kwargs["temperature"] == 0.7  # Default
    
    anthropic_model.temperature = 1.5
    kwargs = anthropic_model._get_api_kwargs()
    assert kwargs["temperature"] == 1.0  # Clamped to max
    
    anthropic_model.temperature = -0.5
    kwargs = anthropic_model._get_api_kwargs()
    assert kwargs["temperature"] == 0.0  # Clamped to min
    
    # Test max_tokens conversion
    anthropic_model.max_tokens = "1000"
    kwargs = anthropic_model._get_api_kwargs()
    assert kwargs["max_tokens"] == 1000  # Converted to int
    
    # Test streaming parameter
    anthropic_model.streaming = True
    kwargs = anthropic_model._get_api_kwargs()
    assert kwargs["stream"] is True
    
    kwargs = anthropic_model._get_api_kwargs(exclude_stream=True)
    assert "stream" not in kwargs

def test_to_langchain_with_custom_params():
    """Test LangChain conversion with custom parameters."""
    model = AnthropicLanguageModel(
        api_key="test-key",
        base_url="https://custom.anthropic.com",
        model_name="claude-3-sonnet",
        max_tokens=1000,
        temperature=0.8,
        top_p=0.95,
        streaming=True,
    )
    
    langchain_model = model.to_langchain()
    
    assert langchain_model.max_tokens == 1000
    assert langchain_model.temperature == 0.8
    assert langchain_model.top_p == 0.95
    assert langchain_model.streaming is True
    assert langchain_model.anthropic_api_url == "https://custom.anthropic.com"
    assert langchain_model.model == "claude-3-sonnet"

@pytest.mark.asyncio
async def test_achat_complete_error_handling(anthropic_model):
    """Test async chat completion error handling."""
    messages = [{"role": "user", "content": "Hello!"}]
    
    # Mock API error
    class MockError(Exception):
        def __init__(self):
            super().__init__("Rate limit exceeded")
            self.status_code = 429
            self.response = type("Response", (), {"text": "Rate limit exceeded"})
    
    anthropic_model.async_client.messages.create.side_effect = MockError()
    
    with pytest.raises(Exception) as exc_info:
        await anthropic_model.achat_complete(messages)
    
    assert "Rate limit exceeded" in str(exc_info.value)
