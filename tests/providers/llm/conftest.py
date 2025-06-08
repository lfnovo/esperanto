from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from esperanto.providers.llm.google import GoogleLanguageModel
from esperanto.providers.llm.groq import GroqLanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel


@pytest.fixture
def mock_openai_response():
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.created = 1677858242
    mock_response.model = "gpt-4"

    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_message.role = "assistant"

    mock_choice = MagicMock()
    mock_choice.index = 0
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.completion_tokens = 10
    mock_usage.prompt_tokens = 20
    mock_usage.total_tokens = 30
    mock_response.usage = mock_usage

    return mock_response


@pytest.fixture
def openai_model():
    """Create OpenAILanguageModel with mocked HTTP client."""
    model = OpenAILanguageModel(api_key="test-key")
    
    # Mock the HTTP clients
    mock_client = MagicMock()
    mock_async_client = AsyncMock()
    
    # Create mock HTTP response data
    mock_response_data = {
        "id": "chatcmpl-123",
        "created": 1677858242,
        "model": "gpt-4",
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
            "prompt_tokens": 20,
            "total_tokens": 30
        }
    }
    
    def mock_post_side_effect(url, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        # Add streaming capability
        mock_response.iter_text = MagicMock(return_value=iter([
            'data: {"choices":[{"delta":{"content":"Test"},"index":0}]}\n',
            'data: [DONE]\n'
        ]))
        return mock_response
    
    async def mock_async_post_side_effect(url, **kwargs):
        mock_response = MagicMock()  # Use regular Mock, not AsyncMock
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        # Add async streaming capability
        async def async_iter():
            yield 'data: {"choices":[{"delta":{"content":"Test"},"index":0}]}\n'
            yield 'data: [DONE]\n'
        mock_response.aiter_text = MagicMock(return_value=async_iter())
        return mock_response
    
    def mock_get_side_effect(url, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "gpt-4", "owned_by": "openai"}]}
        return mock_response
    
    mock_client.post.side_effect = mock_post_side_effect
    mock_client.get.side_effect = mock_get_side_effect
    mock_async_client.post.side_effect = mock_async_post_side_effect
    
    model.client = mock_client
    model.async_client = mock_async_client
    
    yield model


@pytest.fixture
def google_model():
    with patch("google.genai.Client") as mock_client:
        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock the models object (sync)
        mock_models = MagicMock()
        mock_client_instance.models = mock_models
        # Mock the aio.models object (async)
        mock_aio = MagicMock()
        mock_client_instance.aio = MagicMock()
        mock_client_instance.aio.models = mock_aio

        # Mock generate_content method for sync
        mock_part = MagicMock()
        mock_part.text = "Hello! How can I help you today?"
        mock_part.strip = lambda: "Hello! How can I help you today?"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.prompt_feedback = MagicMock()
        mock_response.prompt_feedback.block_reason = None

        # Async response
        mock_async_response = MagicMock()
        mock_async_response.candidates = [mock_candidate]

        # Streaming async generator
        async def async_stream_response(*args, **kwargs):
            yield mock_response

        # Assign the mocks
        mock_models.generate_content = MagicMock(return_value=mock_response)
        mock_models.generate_content_async = AsyncMock(return_value=mock_async_response)
        mock_models.list_models = MagicMock(return_value=[])
        # Patch aio.models for new SDK async usage
        mock_aio.generate_content = AsyncMock(return_value=mock_async_response)
        mock_aio.generate_content_stream = AsyncMock(return_value=async_stream_response())

        # Initialize model with test key
        model = GoogleLanguageModel(api_key="test-key")
        model._client = mock_client_instance

        yield model


@pytest.fixture
def mock_groq_response():
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.created = 1677858242
    mock_response.model = "mixtral-8x7b-32768"

    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_message.role = "assistant"

    mock_choice = MagicMock()
    mock_choice.index = 0
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.completion_tokens = 10
    mock_usage.prompt_tokens = 20
    mock_usage.total_tokens = 30
    mock_response.usage = mock_usage

    return mock_response


@pytest.fixture
def groq_model():
    """Create GroqLanguageModel with mocked HTTP client."""
    model = GroqLanguageModel(api_key="test-key")
    
    # Mock the HTTP clients
    mock_client = MagicMock()
    mock_async_client = AsyncMock()
    
    # Create mock HTTP response data
    mock_response_data = {
        "id": "chatcmpl-123",
        "created": 1677858242,
        "model": "mixtral-8x7b-32768",
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
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        return mock_response
    
    async def mock_async_post_side_effect(url, **kwargs):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value=mock_response_data)
        return mock_response
    
    mock_client.post.side_effect = mock_post_side_effect
    mock_async_client.post.side_effect = mock_async_post_side_effect
    
    model.client = mock_client
    model.async_client = mock_async_client
    
    yield model
