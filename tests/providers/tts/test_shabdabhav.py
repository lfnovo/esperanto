"""Tests for the Google TTS provider."""
import base64
from unittest.mock import AsyncMock, Mock
import os
import pytest

from esperanto.providers.tts.shabdabhav import ShabdabhavTextToSpeechModel


@pytest.fixture
def mock_httpx_response():

    return {
        "audio": b"test audio data",
        "voices": []
    }

@pytest.fixture
def tts_model(mock_httpx_response):
    """Create a TTS model instance with mocked clients."""
    # Set API key in environment for testing
    os.environ["SHABDABHAV_API_KEY"] = "test-key"
    
    model = ShabdabhavTextToSpeechModel(
        model_name="piper-tts"
    )
    
    # Mock the HTTP clients
    mock_client = Mock()
    mock_async_client = AsyncMock()
    
    def mock_post(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_httpx_response["audio"]()
        return mock_response
    
    async def mock_async_post(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_httpx_response["audio"]()
        return mock_response
    
    # def mock_get(url, **kwargs):
    #     mock_response = Mock()
    #     mock_response.status_code = 200
    #     mock_response.json.return_value = mock_httpx_response["voices"]()
    #     return mock_response
    
    mock_client.post = mock_post
    # mock_client.get = mock_get
    mock_async_client.post = mock_async_post
    
    model.client = mock_client
    model.async_client = mock_async_client
    
    yield model
    
    # Clean up environment variable
    # if "ELEVENLABS_API_KEY" in os.environ:
    #     del os.environ["ELEVENLABS_API_KEY"]

def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.model_name == "piper-tts"
    assert tts_model.PROVIDER == "shabdabhav"
    # assert tts_model.api_key == "test-key"


def test_generate_speech(tts_model):
    """Test synchronous speech generation."""
    # Test generation
    response = tts_model.generate_speech(
        text="Hello world",
        voice="en/en_US/amy/medium/en_US-amy-medium.onnx"
    )

    assert response.audio_data == b"test audio data"
    assert response.content_type == "audio/wav"
    assert response.model == "piper-tts"
    assert response.voice == "en/en_US/amy/medium/en_US-amy-medium.onnx"
    assert response.provider == "shabdabhav"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    """Test asynchronous speech generation."""
    # Test generation
    response = await tts_model.agenerate_speech(
        text="Hello world",
        voice="en/en_US/amy/medium/en_US-amy-medium.onnx"
    )

    assert response.audio_data == b"test audio data"
    assert response.content_type == "audio/wav"
    assert response.model == "piper-tts"
    assert response.voice == "en/en_US/amy/medium/en_US-amy-medium.onnx"
    assert response.provider == "shabdabhav"

