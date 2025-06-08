"""Tests for the OpenAI TTS provider."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

from esperanto.providers.tts.openai import OpenAITextToSpeechModel


@pytest.fixture
def mock_tts_audio_response():
    """Mock binary audio response data."""
    return b"mock audio data for testing"


@pytest.fixture
def mock_openai_models_response():
    """Mock HTTP response for OpenAI models API."""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "owned_by": "openai-internal"
            },
            {
                "id": "tts-1-hd",
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
def mock_httpx_clients(mock_tts_audio_response, mock_openai_models_response):
    """Mock httpx clients for OpenAI TTS."""
    client = Mock()
    async_client = AsyncMock()

    # Mock HTTP response objects
    def make_response(status_code, content=None, json_data=None):
        response = Mock()
        response.status_code = status_code
        if content is not None:
            response.content = content
        if json_data is not None:
            response.json.return_value = json_data
        return response

    def make_async_response(status_code, content=None, json_data=None):
        response = AsyncMock()
        response.status_code = status_code
        if content is not None:
            response.content = content
        if json_data is not None:
            response.json = Mock(return_value=json_data)
        return response

    # Configure responses based on URL
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_response(200, content=mock_tts_audio_response)
        return make_response(404, json_data={"error": "Not found"})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_response(200, json_data=mock_openai_models_response)
        return make_response(404, json_data={"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_async_response(200, content=mock_tts_audio_response)
        return make_async_response(404, json_data={"error": "Not found"})

    async def mock_async_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_async_response(200, json_data=mock_openai_models_response)
        return make_async_response(404, json_data={"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    async_client.get.side_effect = mock_async_get_side_effect

    return client, async_client


@pytest.fixture
def tts_model(mock_httpx_clients):
    """Create a TTS model instance with mocked HTTP clients."""
    model = OpenAITextToSpeechModel(
        api_key="test-key",
        model_name="tts-1"
    )
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.model_name == "tts-1"
    assert tts_model.PROVIDER == "openai"


def test_generate_speech(tts_model):
    """Test synchronous speech generation."""
    response = tts_model.generate_speech(
        text="Hello world",
        voice="alloy"
    )

    # Verify HTTP POST was called
    tts_model.client.post.assert_called_once()
    call_args = tts_model.client.post.call_args

    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/audio/speech"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"

    # Check JSON payload
    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "tts-1"
    assert json_payload["voice"] == "alloy"
    assert json_payload["input"] == "Hello world"

    # Check response
    assert response.audio_data == b"mock audio data for testing"
    assert response.content_type == "audio/mp3"
    assert response.model == "tts-1"
    assert response.voice == "alloy"
    assert response.provider == "openai"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    """Test asynchronous speech generation."""
    response = await tts_model.agenerate_speech(
        text="Hello world",
        voice="nova"
    )

    # Verify async HTTP POST was called
    tts_model.async_client.post.assert_called_once()
    call_args = tts_model.async_client.post.call_args

    # Check URL
    assert call_args[0][0] == "https://api.openai.com/v1/audio/speech"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"

    # Check JSON payload
    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "tts-1"
    assert json_payload["voice"] == "nova"
    assert json_payload["input"] == "Hello world"

    # Check response
    assert response.audio_data == b"mock audio data for testing"
    assert response.content_type == "audio/mp3"
    assert response.model == "tts-1"
    assert response.voice == "nova"
    assert response.provider == "openai"


def test_available_voices(tts_model):
    """Test getting available voices."""
    voices = tts_model.available_voices

    assert len(voices) == 6  # OpenAI has 6 default voices
    
    # Test one voice
    voice = voices["alloy"]
    assert voice.name == "alloy"
    assert voice.id == "alloy"
    assert voice.gender == "NEUTRAL"
    assert voice.language_code == "en-US"


def test_models(tts_model):
    """Test that the models property works with HTTP."""
    models = tts_model.models
    
    # Verify HTTP GET was called
    tts_model.client.get.assert_called_with(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }
    )
    
    # Check that only TTS models are returned
    assert len(models) == 2
    assert models[0].id == "tts-1"
    assert models[1].id == "tts-1-hd"
    assert models[0].type == "text_to_speech"
    assert models[1].type == "text_to_speech"
