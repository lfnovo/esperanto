from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.providers.llm.anthropic import AnthropicLanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel

try:
    from esperanto.providers.llm.groq import GroqLanguageModel
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

@pytest.fixture
def mock_openai_response():
    class Choice:
        def __init__(self):
            self.index = 0
            self.message = type('Message', (), {
                'content': "Test response",
                'role': "assistant",
                'function_call': None,
                'tool_calls': None
            })
            self.finish_reason = "stop"

    class Usage:
        def __init__(self):
            self.completion_tokens = 10
            self.prompt_tokens = 8
            self.total_tokens = 18

    class Response:
        def __init__(self):
            self.id = "chatcmpl-123"
            self.created = 1677858242
            self.model = "gpt-3.5-turbo-0613"
            self.choices = [Choice()]
            self.usage = Usage()

    return Response()

@pytest.fixture
def mock_anthropic_response():
    class TextBlock:
        def __init__(self, text):
            self.text = text
            self.type = 'text'

    class Usage:
        def __init__(self, input_tokens, output_tokens):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class Message:
        def __init__(self):
            self.id = "msg_123"
            self.content = [TextBlock("Test response")]
            self.model = "claude-3-opus-20240229"
            self.role = "assistant"
            self.stop_reason = "end_turn"
            self.stop_sequence = None
            self.type = "message"
            self.usage = Usage(input_tokens=57, output_tokens=40)

    return Message()

@pytest.fixture
def mock_groq_response():
    if not HAS_GROQ:
        pytest.skip("Groq not installed")

    class Choice:
        def __init__(self):
            self.index = 0
            self.message = type('Message', (), {
                'content': "Test response",
                'role': "assistant",
            })
            self.finish_reason = "stop"

    class Usage:
        def __init__(self):
            self.input_tokens = 8
            self.output_tokens = 10

    class Response:
        def __init__(self):
            self.id = "1234"
            self.created = 1677858242
            self.model = "mixtral-8x7b-32768"
            self.choices = [Choice()]
            self.usage = Usage()

    return Response()

@pytest.fixture
def mock_openai_client(mock_openai_response):
    client = Mock()
    async_client = AsyncMock()
    
    # Mock synchronous completion
    mock_completion = Mock()
    mock_completion.configure_mock(**mock_openai_response.__dict__)
    client.chat.completions.create.return_value = mock_completion
    
    # Mock async completion
    mock_async_completion = AsyncMock()
    mock_async_completion.configure_mock(**mock_openai_response.__dict__)
    async_client.chat.completions.create.return_value = mock_async_completion
    
    return client, async_client

@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    client = Mock()
    async_client = AsyncMock()
    
    # Mock synchronous completion
    mock_completion = Mock()
    mock_completion.configure_mock(**mock_anthropic_response.__dict__)
    client.messages.create.return_value = mock_completion
    
    # Mock async completion
    mock_async_completion = AsyncMock()
    mock_async_completion.configure_mock(**mock_anthropic_response.__dict__)
    async_client.messages.create.return_value = mock_async_completion
    
    return client, async_client

@pytest.fixture
def mock_groq_client(mock_groq_response):
    if not HAS_GROQ:
        pytest.skip("Groq not installed")

    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_groq_response)
    return mock_client

@pytest.fixture
def openai_model(mock_openai_client):
    model = OpenAILanguageModel(
        api_key="test-key",
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    model.client, model.async_client = mock_openai_client
    return model

@pytest.fixture
def anthropic_model(mock_anthropic_client):
    model = AnthropicLanguageModel(
        api_key="test-key",
        model_name="claude-3-opus-20240229",
        temperature=0.7
    )
    model.client, model.async_client = mock_anthropic_client
    return model

@pytest.fixture
def groq_model(mock_groq_client):
    if not HAS_GROQ:
        pytest.skip("Groq not installed")

    model = GroqLanguageModel(
        api_key="test-key",
        model_name="mixtral-8x7b-32768",
        temperature=1.0,
        max_tokens=850,
        top_p=0.9,
    )
    model.client = mock_groq_client
    return model
