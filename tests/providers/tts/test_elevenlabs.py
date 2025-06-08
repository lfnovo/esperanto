"""Tests for the ElevenLabs TTS provider."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import os

from esperanto.providers.tts.elevenlabs import ElevenLabsTextToSpeechModel


@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for ElevenLabs API."""
    def create_audio_response():
        return b"test audio data"
    
    def create_voices_response():
        return {
            "voices": [
                {
                    "voice_id": "test-voice-1",
                    "name": "Test Voice",
                    "labels": {"gender": "female", "language": "en"},
                    "description": "A test voice",
                    "preview_url": "https://example.com/preview.mp3"
                }
            ]
        }
    
    return {
        "audio": create_audio_response,
        "voices": create_voices_response
    }


@pytest.fixture
def tts_model(mock_httpx_response):
    """Create a TTS model instance with mocked clients."""
    # Set API key in environment for testing
    os.environ["ELEVENLABS_API_KEY"] = "test-key"
    
    model = ElevenLabsTextToSpeechModel(
        api_key="test-key",
        model_name="eleven_multilingual_v2"
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
    
    def mock_get(url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_httpx_response["voices"]()
        return mock_response
    
    mock_client.post = mock_post
    mock_client.get = mock_get
    mock_async_client.post = mock_async_post
    
    model.client = mock_client
    model.async_client = mock_async_client
    
    yield model
    
    # Clean up environment variable
    if "ELEVENLABS_API_KEY" in os.environ:
        del os.environ["ELEVENLABS_API_KEY"]


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.model_name == "eleven_multilingual_v2"
    assert tts_model.PROVIDER == "elevenlabs"
    assert tts_model.api_key == "test-key"


def test_generate_speech(tts_model):
    """Test synchronous speech generation."""
    # Test generation
    response = tts_model.generate_speech(
        text="Hello world",
        voice="test_voice"
    )

    assert response.audio_data == b"test audio data"
    assert response.content_type == "audio/mp3"
    assert response.model == "eleven_multilingual_v2"
    assert response.voice == "test_voice"
    assert response.provider == "elevenlabs"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    """Test asynchronous speech generation."""
    # Test generation
    response = await tts_model.agenerate_speech(
        text="Hello world",
        voice="test_voice"
    )

    assert response.audio_data == b"test audio data"
    assert response.content_type == "audio/mp3"
    assert response.model == "eleven_multilingual_v2"
    assert response.voice == "test_voice"
    assert response.provider == "elevenlabs"


def test_available_voices(tts_model):
    """Test getting available voices."""
    # Test getting voices
    voices = tts_model.available_voices
    assert len(voices) == 1
    assert voices["test-voice-1"].name == "Test Voice"
    assert voices["test-voice-1"].id == "test-voice-1"
    assert voices["test-voice-1"].gender == "FEMALE"
    assert voices["test-voice-1"].language_code == "en"
    assert voices["test-voice-1"].description == "A test voice"
    assert voices["test-voice-1"].preview_url == "https://example.com/preview.mp3"
