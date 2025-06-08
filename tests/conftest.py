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
    return {
        "id": "msg_123",
        "content": [
            {
                "text": "Test response",
                "type": "text"
            }
        ],
        "model": "claude-3-opus-20240229",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {
            "input_tokens": 57,
            "output_tokens": 40
        }
    }

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
def mock_anthropic_client(mock_anthropic_response):
    """Mock httpx clients for Anthropic API."""
    client = Mock()
    async_client = AsyncMock()
    
    # Mock HTTP response objects
    def make_response(status_code, data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data
        return response
    
    def make_async_response(status_code, data):
        response = AsyncMock()
        response.status_code = status_code
        response.json = Mock(return_value=data)
        return response
    
    # Configure responses
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/messages"):
            return make_response(200, mock_anthropic_response)
        elif url.endswith("/models"):
            return make_response(200, {"data": [{"id": "claude-3-opus-20240229", "max_tokens": 200000}]})
        return make_response(404, {"error": "Not found"})
    
    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_response(200, {"data": [{"id": "claude-3-opus-20240229", "max_tokens": 200000}]})
        return make_response(404, {"error": "Not found"})
    
    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/messages"):
            return make_async_response(200, mock_anthropic_response)
        elif url.endswith("/models"):
            return make_async_response(200, {"data": [{"id": "claude-3-opus-20240229", "max_tokens": 200000}]})
        return make_async_response(404, {"error": "Not found"})
    
    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect
    
    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    
    return client, async_client

@pytest.fixture
def mock_groq_client(mock_groq_response):
    if not HAS_GROQ:
        pytest.skip("Groq not installed")

    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_groq_response)
    return mock_client

@pytest.fixture
def openai_model():
    """Create OpenAILanguageModel with mocked HTTP client."""
    model = OpenAILanguageModel(
        api_key="test-key",
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Mock the HTTP clients
    mock_client = Mock()
    mock_async_client = AsyncMock()
    
    # Create mock HTTP response data
    mock_response_data = {
        "id": "chatcmpl-123",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0613",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": "Test response",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "completion_tokens": 10,
            "prompt_tokens": 8,
            "total_tokens": 18
        }
    }
    
    def mock_post_side_effect(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        # Add streaming capability
        mock_response.iter_text = Mock(return_value=iter([
            'data: {"choices":[{"delta":{"content":"Test"},"index":0}]}\n',
            'data: [DONE]\n'
        ]))
        return mock_response
    
    async def mock_async_post_side_effect(url, **kwargs):
        mock_response = Mock()  # Use regular Mock, not AsyncMock
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        # Add async streaming capability
        async def async_iter():
            yield 'data: {"choices":[{"delta":{"content":"Test"},"index":0}]}\n'
            yield 'data: [DONE]\n'
        mock_response.aiter_text = Mock(return_value=async_iter())
        return mock_response
    
    def mock_get_side_effect(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "gpt-3.5-turbo", "owned_by": "openai"}]}
        return mock_response
    
    mock_client.post.side_effect = mock_post_side_effect
    mock_client.get.side_effect = mock_get_side_effect
    mock_async_client.post.side_effect = mock_async_post_side_effect
    
    model.client = mock_client
    model.async_client = mock_async_client
    
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
