"""Tests for the Google TTS provider."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from esperanto.providers.tts.google import GoogleTextToSpeechModel


@pytest.fixture
def mock_google_client(mocker):
    """Mock the Google Cloud TTS client."""
    mock = mocker.patch("google.cloud.texttospeech.TextToSpeechClient", autospec=True)
    return mock


@pytest.fixture
def mock_async_google_client(mocker):
    """Mock the Google Cloud TTS async client."""
    mock = mocker.patch("google.cloud.texttospeech.TextToSpeechAsyncClient", autospec=True)
    return mock


@pytest.fixture
def tts_model(mock_google_client, mock_async_google_client):
    """Create a TTS model instance with mocked clients."""
    mock_credentials = MagicMock()
    mock_credentials.before_request = MagicMock()

    model = GoogleTextToSpeechModel(
        model_name="en-US-Standard-A",
        api_key=None,
        base_url=None,
        config={"credentials": mock_credentials}
    )

    # Mock the clients to avoid actual API calls
    model.client = mock_google_client.return_value
    model.async_client = mock_async_google_client.return_value

    return model


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.model_name == "en-US-Standard-A"
    assert tts_model.PROVIDER == "google"


def test_generate_speech(tts_model, mock_google_client):
    """Test synchronous speech generation."""
    # Mock response
    mock_response = MagicMock()
    mock_response.audio_content = b"test audio data"
    mock_google_client.return_value.synthesize_speech.return_value = mock_response

    # Test generation
    response = tts_model.generate_speech(
        text="Hello world",
        voice="en-US-Standard-A"
    )

    assert response.audio_data == b"test audio data"
    assert response.content_type == "audio/mp3"
    assert response.model == "en-US-Standard-A"
    assert response.voice == "en-US-Standard-A"
    assert response.provider == "google"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model, mock_async_google_client):
    """Test asynchronous speech generation."""
    # Mock response
    mock_response = MagicMock()
    mock_response.audio_content = b"test audio data"
    mock_async_google_client.return_value.synthesize_speech = AsyncMock(
        return_value=mock_response
    )

    # Test generation
    response = await tts_model.agenerate_speech(
        text="Hello world",
        voice="en-US-Standard-A"
    )

    assert response.audio_data == b"test audio data"
    assert response.content_type == "audio/mp3"
    assert response.model == "en-US-Standard-A"
    assert response.voice == "en-US-Standard-A"
    assert response.provider == "google"


def test_available_voices(tts_model, mock_google_client):
    """Test getting available voices."""
    # Mock response
    mock_voice = MagicMock()
    mock_voice.name = "en-US-Standard-A"
    mock_voice.language_codes = ["en-US"]
    mock_voice.ssml_gender = "FEMALE"
    mock_voice.natural_sample_rate_hertz = 24000

    mock_response = MagicMock()
    mock_response.voices = [mock_voice]
    mock_google_client.return_value.list_voices.return_value = mock_response

    # Test getting voices
    voices = tts_model.available_voices
    assert len(voices) == 1
    assert voices["en-US-Standard-A"].name == "en-US-Standard-A"
    assert voices["en-US-Standard-A"].id == "en-US-Standard-A"
    assert voices["en-US-Standard-A"].gender == "FEMALE"
    assert voices["en-US-Standard-A"].language_code == "en-US"
