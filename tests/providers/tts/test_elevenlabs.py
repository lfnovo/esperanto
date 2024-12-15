"""Tests for the ElevenLabs TTS provider."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from esperanto.providers.tts.elevenlabs import ElevenLabsTextToSpeechModel


@pytest.fixture
def mock_elevenlabs_client():
    """Mock ElevenLabs client."""
    with patch("esperanto.providers.tts.elevenlabs.ElevenLabs") as mock_client:
        yield mock_client


@pytest.fixture
def mock_async_elevenlabs_client():
    """Mock async ElevenLabs client."""
    with patch("esperanto.providers.tts.elevenlabs.AsyncElevenLabs") as mock_client:
        yield mock_client


@pytest.fixture
def tts_model(mock_elevenlabs_client, mock_async_elevenlabs_client):
    """Create a TTS model instance with mocked clients."""
    return ElevenLabsTextToSpeechModel(
        api_key="test-key",
        model_name="eleven_multilingual_v2"
    )


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.model_name == "eleven_multilingual_v2"
    assert tts_model.PROVIDER == "elevenlabs"


def test_generate_speech(tts_model, mock_elevenlabs_client):
    """Test synchronous speech generation."""
    # Mock response
    mock_response = [b"test ", b"audio ", b"data"]
    mock_elevenlabs_client.return_value.text_to_speech.convert.return_value = mock_response

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
async def test_agenerate_speech(tts_model, mock_async_elevenlabs_client):
    """Test asynchronous speech generation."""
    # Mock async response
    mock_response = AsyncMock()
    mock_response.__aiter__.return_value = [b"test ", b"audio ", b"data"]
    mock_async_elevenlabs_client.return_value.text_to_speech.convert.return_value = mock_response

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


def test_available_voices(tts_model, mock_elevenlabs_client):
    """Test getting available voices."""
    # Mock response
    mock_voice = MagicMock()
    mock_voice.name = "Test Voice"
    mock_voice.voice_id = "test-voice-1"
    mock_voice.labels = {"gender": "female", "language": "en"}
    mock_voice.description = "A test voice"
    mock_voice.preview_url = "https://example.com/preview.mp3"

    mock_response = MagicMock()
    mock_response.voices = [mock_voice]
    mock_elevenlabs_client.return_value.voices.get_all.return_value = mock_response

    # Test getting voices
    voices = tts_model.available_voices
    assert len(voices) == 1
    assert voices["test-voice-1"].name == "Test Voice"
    assert voices["test-voice-1"].id == "test-voice-1"
    assert voices["test-voice-1"].gender == "FEMALE"
    assert voices["test-voice-1"].language_code == "en"
    assert voices["test-voice-1"].description == "A test voice"
    assert voices["test-voice-1"].preview_url == "https://example.com/preview.mp3"
