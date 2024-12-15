"""Tests for Groq speech-to-text provider."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from esperanto.factory import AIFactory
from esperanto.providers.stt.groq import GroqSpeechToTextModel
from esperanto.types import TranscriptionResponse


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"mock audio content")
    return str(audio_file)


@pytest.fixture
def mock_groq_response():
    """Mock Groq transcription response."""
    response = MagicMock()
    response.text = "This is a test transcription"
    return response


@pytest.fixture
def mock_groq_client(mock_groq_response):
    """Mock Groq client."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_groq_response
    return mock_client


@pytest.fixture
def mock_async_groq_client(mock_groq_response):
    """Mock async Groq client."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_groq_response)
    return mock_client


@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        yield


def test_factory_creates_groq_stt():
    """Test that AIFactory creates Groq STT model."""
    model = AIFactory.create_stt("groq")
    assert isinstance(model, GroqSpeechToTextModel)


def test_groq_transcribe(audio_file, mock_groq_client):
    """Test Groq transcribe method."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        with patch("groq.Groq.__init__", return_value=None) as mock_init:
            with patch("groq.AsyncGroq.__init__", return_value=None) as mock_async_init:
                model = GroqSpeechToTextModel()
                model.client.audio = mock_groq_client.audio
                model.async_client.audio = mock_groq_client.audio
                response = model.transcribe(audio_file)
                assert isinstance(response, TranscriptionResponse)
                assert response.text == "This is a test transcription"


@pytest.mark.asyncio
async def test_groq_atranscribe(audio_file, mock_async_groq_client):
    """Test Groq async transcribe method."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        with patch("groq.Groq.__init__", return_value=None) as mock_init:
            with patch("groq.AsyncGroq.__init__", return_value=None) as mock_async_init:
                model = GroqSpeechToTextModel()
                model.client.audio = mock_async_groq_client.audio
                model.async_client.audio = mock_async_groq_client.audio
                response = await model.atranscribe(audio_file)
                assert isinstance(response, TranscriptionResponse)
                assert response.text == "This is a test transcription"


def test_groq_transcribe_with_options(audio_file, mock_groq_client):
    """Test Groq transcribe with language and prompt."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        with patch("groq.Groq.__init__", return_value=None) as mock_init:
            with patch("groq.AsyncGroq.__init__", return_value=None) as mock_async_init:
                model = GroqSpeechToTextModel()
                model.client.audio = mock_groq_client.audio
                model.async_client.audio = mock_groq_client.audio
                response = model.transcribe(
                    audio_file,
                    language="en",
                    prompt="This is a podcast about AI",
                )
                assert isinstance(response, TranscriptionResponse)
                assert response.text == "This is a test transcription"


def test_groq_transcribe_file_object(mock_groq_client):
    """Test Groq transcribe with file object."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        with patch("groq.Groq.__init__", return_value=None) as mock_init:
            with patch("groq.AsyncGroq.__init__", return_value=None) as mock_async_init:
                model = GroqSpeechToTextModel()
                model.client.audio = mock_groq_client.audio
                model.async_client.audio = mock_groq_client.audio
                with open(__file__, "rb") as f:
                    response = model.transcribe(f)
                assert isinstance(response, TranscriptionResponse)
                assert response.text == "This is a test transcription"
