"""Tests for the OpenAI TTS provider."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from esperanto.providers.tts.openai import OpenAITextToSpeechModel


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.audio.speech.create.return_value = MagicMock(content=b"test audio data")
    return client


@pytest.fixture
def mock_async_openai_client():
    client = MagicMock()
    client.audio.speech.create = AsyncMock(return_value=MagicMock(content=b"test audio data"))
    return client


@pytest.fixture
def tts_model(mock_openai_client, mock_async_openai_client):
    model = OpenAITextToSpeechModel(api_key="test-key", model_name="tts-1")
    model.client = mock_openai_client
    model.async_client = mock_async_openai_client
    return model


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.model_name == "tts-1"
    assert tts_model.PROVIDER == "openai"


def test_generate_speech(tts_model, mock_openai_client):
    """Test synchronous speech generation."""
    # Mock response
    mock_response = MagicMock()
    mock_response.content = b"test audio data"
    mock_openai_client.audio.speech.create.return_value = mock_response

    # Test generation
    response = tts_model.generate_speech(
        text="Hello world",
        voice="alloy"
    )

    assert response.audio_data == b"test audio data"
    assert response.content_type == "audio/mp3"
    assert response.model == "tts-1"
    assert response.voice == "alloy"
    assert response.provider == "openai"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model, mock_async_openai_client):
    """Test asynchronous speech generation."""
    # Mock async response
    mock_response = AsyncMock()
    mock_response.content = b"test audio data"
    mock_async_openai_client.audio.speech.create = AsyncMock(return_value=mock_response)

    # Test generation
    response = await tts_model.agenerate_speech(
        text="Hello world",
        voice="alloy"
    )

    assert response.audio_data == b"test audio data"
    assert response.content_type == "audio/mp3"
    assert response.model == "tts-1"
    assert response.voice == "alloy"
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
