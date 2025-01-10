import os
from unittest.mock import AsyncMock, MagicMock, patch

import google.generativeai as genai
import pytest

from esperanto.providers.llm.google import GoogleLanguageModel


def test_provider_name(google_model):
    assert google_model.provider == "google"


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


def test_chat_complete(google_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Mock response
    mock_response = MagicMock()
    mock_response.text = "Hello! How can I help you today?"
    mock_response.prompt_feedback.block_reason = None
    google_model._client.generate_content.return_value = mock_response
    
    result = google_model.chat_complete(messages)
    
    # Verify the client was called with correct parameters
    google_model._client.generate_content.assert_called_once()
    call_args = google_model._client.generate_content.call_args[1]
    
    # Check generation config
    assert isinstance(call_args["generation_config"], genai.GenerationConfig)
    assert call_args["generation_config"].temperature == 1.0
    assert call_args["generation_config"].top_p == 0.9
    
    # Check response format
    assert result.choices[0].message.content == "Hello! How can I help you today?"
    assert result.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_achat_complete(google_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Create a mock response with the correct structure
    mock_text = "Hello! How can I help you today?"
    mock_part = MagicMock()
    mock_part.text = mock_text
    
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    mock_candidate.finish_reason = "STOP"
    
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    
    # Use AsyncMock for async method
    google_model._client.generate_content_async = AsyncMock(return_value=mock_response)
    
    result = await google_model.achat_complete(messages)
    
    # Verify the async client was called with correct parameters
    google_model._client.generate_content_async.assert_called_once()
    call_args = google_model._client.generate_content_async.call_args[1]
    
    # Check generation config
    assert isinstance(call_args["generation_config"], genai.GenerationConfig)
    assert call_args["generation_config"].temperature == 1.0
    assert call_args["generation_config"].top_p == 0.9
    
    # Check response format
    assert result.choices[0].message.content == mock_text
    assert result.choices[0].finish_reason == "stop"


def test_json_structured_output(google_model):
    google_model.structured = {"type": "json"}
    messages = [{"role": "user", "content": "Hello!"}]
    
    response = google_model.chat_complete(messages)
    
    call_args = google_model._client.generate_content.call_args
    assert call_args[1]["generation_config"].response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_json_structured_output_async(google_model):
    google_model.structured = {"type": "json"}
    messages = [{"role": "user", "content": "Hello!"}]
    
    # Mock the async response
    mock_response = MagicMock()
    mock_response.text = '{"greeting": "Hello!", "response": "How can I help?"}'
    mock_response.prompt_feedback.block_reason = None
    google_model._client.generate_content_async = AsyncMock(return_value=mock_response)
    
    response = await google_model.achat_complete(messages)
    
    call_args = google_model._client.generate_content_async.call_args
    assert call_args[1]["generation_config"].response_mime_type == "application/json"


def test_to_langchain(google_model):
    langchain_model = google_model.to_langchain()
    
    # Test model configuration
    assert langchain_model.model == "models/gemini-1.5-pro"
    assert langchain_model.temperature == 1.0
    assert langchain_model.top_p == 0.9
    # Skip API key check since it's masked in SecretStr
