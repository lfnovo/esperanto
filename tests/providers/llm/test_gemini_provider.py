import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from esperanto.providers.llm.google import GoogleLanguageModel


def test_provider_name():
    """Test provider name."""
    from esperanto.providers.llm.google import GoogleLanguageModel
    
    model = GoogleLanguageModel(api_key="test-key")
    assert model.provider == "google"


def test_initialization_with_api_key():
    model = GoogleLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"


def test_initialization_with_env_var():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-test-key"}):
        model = GoogleLanguageModel()
        assert model.api_key == "env-test-key"


def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Google API key not found"):
            GoogleLanguageModel()


def test_chat_complete():
    """Test chat completion with httpx mocking."""
    from unittest.mock import Mock
    from esperanto.providers.llm.google import GoogleLanguageModel
    
    # Create fresh model instance without old fixtures
    model = GoogleLanguageModel(api_key="test-key")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # Mock Google API response data
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Hello! How can I help you today?"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": []
            }
        ],
        "promptFeedback": {
            "safetyRatings": []
        }
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    result = model.chat_complete(messages)

    assert result.choices[0].message.content == "Hello! How can I help you today?"
    assert result.choices[0].finish_reason == "stop"
    assert result.provider == "google"


@pytest.mark.asyncio
async def test_achat_complete():
    """Test async chat completion with httpx mocking."""
    from unittest.mock import Mock, AsyncMock
    from esperanto.providers.llm.google import GoogleLanguageModel
    
    # Create fresh model instance
    model = GoogleLanguageModel(api_key="test-key")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # Mock Google API response data
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Hello! How can I help you today?"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": []
            }
        ],
        "promptFeedback": {
            "safetyRatings": []
        }
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the async client
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    model.async_client = mock_async_client

    result = await model.achat_complete(messages)

    assert result.choices[0].message.content == "Hello! How can I help you today?"
    assert result.choices[0].finish_reason == "stop"
    assert result.provider == "google"


def test_json_structured_output():
    """Test structured JSON output with httpx mocking."""
    from unittest.mock import Mock
    from esperanto.providers.llm.google import GoogleLanguageModel
    
    # Create fresh model instance
    model = GoogleLanguageModel(api_key="test-key")
    model.structured = {"type": "json"}
    
    messages = [{"role": "user", "content": "Hello!"}]

    # Mock Google API response data for JSON
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"response": "Hello! How can I help you today?"}'
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": []
            }
        ],
        "promptFeedback": {
            "safetyRatings": []
        }
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    response = model.chat_complete(messages)

    # Verify the HTTP request was made correctly
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    
    # Check that JSON structure is properly set in the request
    json_payload = call_args[1]["json"]
    assert json_payload["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.asyncio
async def test_json_structured_output_async():
    """Test async structured JSON output with httpx mocking."""
    from unittest.mock import Mock, AsyncMock
    from esperanto.providers.llm.google import GoogleLanguageModel
    
    # Create fresh model instance
    model = GoogleLanguageModel(api_key="test-key")
    model.structured = {"type": "json"}
    
    messages = [{"role": "user", "content": "Hello!"}]

    # Mock Google API response data for JSON
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"greeting": "Hello!", "response": "How can I help?"}'
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": []
            }
        ],
        "promptFeedback": {
            "safetyRatings": []
        }
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the async client
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    model.async_client = mock_async_client

    response = await model.achat_complete(messages)

    # Verify the HTTP request was made correctly
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    
    # Check that JSON structure is properly set in the request
    json_payload = call_args[1]["json"]
    assert json_payload["generationConfig"]["responseMimeType"] == "application/json"


def test_to_langchain():
    """Test LangChain conversion."""
    from unittest.mock import Mock, patch
    from esperanto.providers.llm.google import GoogleLanguageModel
    
    # Mock the LangChain classes to avoid credential issues
    mock_chat_google = Mock()
    mock_chat_google.model = "gemini-2.0-flash"  # Match what the provider actually uses
    mock_chat_google.temperature = 1.0
    mock_chat_google.top_p = 0.9
    
    with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_chat_class:
        mock_chat_class.return_value = mock_chat_google
        
        # Create fresh model instance
        model = GoogleLanguageModel(api_key="test-key")
        
        langchain_model = model.to_langchain()

        # Test model configuration  
        assert langchain_model.model == "gemini-2.0-flash"
        
        # Verify ChatGoogleGenerativeAI was called with correct parameters
        mock_chat_class.assert_called_once_with(
            model="gemini-2.0-flash",
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            top_p=model.top_p,
        )
    assert langchain_model.temperature == 1.0
    assert langchain_model.top_p == 0.9
    # Skip API key check since it's masked
