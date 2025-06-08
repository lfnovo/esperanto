"""Tests for OpenAI speech-to-text provider."""

import os
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from esperanto.common_types import TranscriptionResponse
from esperanto.factory import AIFactory
from esperanto.providers.stt.openai import OpenAISpeechToTextModel


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"mock audio content")
    return str(audio_file)


@pytest.fixture
def mock_openai_transcription_response():
    """Mock HTTP response for OpenAI transcription API."""
    return {
        "text": "This is a test transcription"
    }


@pytest.fixture
def mock_openai_models_response():
    """Mock HTTP response for OpenAI models API."""
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "owned_by": "openai-internal"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "owned_by": "openai"
            }
        ]
    }


@pytest.fixture
def mock_httpx_clients(mock_openai_transcription_response, mock_openai_models_response):
    """Mock httpx clients for OpenAI STT."""
    from unittest.mock import Mock, AsyncMock
    
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

    # Configure responses based on URL
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/audio/transcriptions"):
            return make_response(200, mock_openai_transcription_response)
        return make_response(404, {"error": "Not found"})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_response(200, mock_openai_models_response)
        return make_response(404, {"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/audio/transcriptions"):
            return make_async_response(200, mock_openai_transcription_response)
        return make_async_response(404, {"error": "Not found"})

    async def mock_async_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_async_response(200, mock_openai_models_response)
        return make_async_response(404, {"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    async_client.get.side_effect = mock_async_get_side_effect

    return client, async_client


@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


def test_factory_creates_openai_stt():
    """Test that AIFactory creates OpenAI STT model."""
    model = AIFactory.create_stt("openai")
    assert isinstance(model, OpenAISpeechToTextModel)


def test_openai_transcribe(audio_file, mock_httpx_clients):
    """Test OpenAI transcribe method."""
    model = OpenAISpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients
    
    response = model.transcribe(audio_file)
    
    # Verify HTTP POST was called
    model.client.post.assert_called_once()
    call_args = model.client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/audio/transcriptions"
    
    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"
    
    # Check files and data
    assert "files" in call_args[1]
    assert "data" in call_args[1]
    assert call_args[1]["data"]["model"] == "whisper-1"
    
    # Check response
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"
    assert response.model == "whisper-1"


@pytest.mark.asyncio
async def test_openai_atranscribe(audio_file, mock_httpx_clients):
    """Test OpenAI async transcribe method."""
    model = OpenAISpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients
    
    response = await model.atranscribe(audio_file)
    
    # Verify async HTTP POST was called
    model.async_client.post.assert_called_once()
    call_args = model.async_client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/audio/transcriptions"
    
    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"
    
    # Check files and data
    assert "files" in call_args[1]
    assert "data" in call_args[1]
    assert call_args[1]["data"]["model"] == "whisper-1"
    
    # Check response
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"
    assert response.model == "whisper-1"


def test_openai_transcribe_with_options(audio_file, mock_httpx_clients):
    """Test OpenAI transcribe with language and prompt."""
    model = OpenAISpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients
    
    response = model.transcribe(
        audio_file,
        language="en",
        prompt="This is a podcast about AI",
    )
    
    # Verify HTTP POST was called
    model.client.post.assert_called_once()
    call_args = model.client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/audio/transcriptions"
    
    # Check data includes language and prompt
    data = call_args[1]["data"]
    assert data["model"] == "whisper-1"
    assert data["language"] == "en"
    assert data["prompt"] == "This is a podcast about AI"
    
    # Check response
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


def test_openai_transcribe_file_object(mock_httpx_clients):
    """Test OpenAI transcribe with file object."""
    model = OpenAISpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients
    
    with open(__file__, "rb") as f:
        response = model.transcribe(f)
    
    # Verify HTTP POST was called
    model.client.post.assert_called_once()
    call_args = model.client.post.call_args
    
    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/audio/transcriptions"
    
    # Check files and data
    assert "files" in call_args[1]
    assert "data" in call_args[1]
    
    # Check response
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


def test_openai_models(mock_httpx_clients):
    """Test that the models property works with HTTP."""
    model = OpenAISpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients
    
    models = model.models
    
    # Verify HTTP GET was called
    model.client.get.assert_called_with(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": "Bearer test-key"
        }
    )
    
    # Check that only whisper models are returned
    assert len(models) == 1
    assert models[0].id == "whisper-1"
    assert models[0].type == "speech_to_text"
