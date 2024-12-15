"""Tests for OpenAI speech-to-text provider."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from esperanto.factory import AIFactory
from esperanto.providers.stt.openai import OpenAISpeechToTextModel
from esperanto.types import TranscriptionResponse


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"mock audio content")
    return str(audio_file)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI transcription response."""
    response = MagicMock()
    response.text = "This is a test transcription"
    return response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_openai_response
    return mock_client


@pytest.fixture
def mock_async_openai_client(mock_openai_response):
    """Mock async OpenAI client."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_openai_response)
    return mock_client


@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


def test_factory_creates_openai_stt():
    """Test that AIFactory creates OpenAI STT model."""
    model = AIFactory.create_stt("openai")
    assert isinstance(model, OpenAISpeechToTextModel)


def test_openai_transcribe(audio_file, mock_openai_client):
    """Test OpenAI transcribe method."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI.__init__", return_value=None) as mock_init:
            with patch("openai.AsyncOpenAI.__init__", return_value=None) as mock_async_init:
                model = OpenAISpeechToTextModel()
                model.client.audio = mock_openai_client.audio
                model.async_client.audio = mock_openai_client.audio
                response = model.transcribe(audio_file)
                assert isinstance(response, TranscriptionResponse)
                assert response.text == "This is a test transcription"


@pytest.mark.asyncio
async def test_openai_atranscribe(audio_file, mock_async_openai_client):
    """Test OpenAI async transcribe method."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI.__init__", return_value=None) as mock_init:
            with patch("openai.AsyncOpenAI.__init__", return_value=None) as mock_async_init:
                model = OpenAISpeechToTextModel()
                model.client.audio = mock_async_openai_client.audio
                model.async_client.audio = mock_async_openai_client.audio
                response = await model.atranscribe(audio_file)
                assert isinstance(response, TranscriptionResponse)
                assert response.text == "This is a test transcription"


def test_openai_transcribe_with_options(audio_file, mock_openai_client):
    """Test OpenAI transcribe with language and prompt."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI.__init__", return_value=None) as mock_init:
            with patch("openai.AsyncOpenAI.__init__", return_value=None) as mock_async_init:
                model = OpenAISpeechToTextModel()
                model.client.audio = mock_openai_client.audio
                model.async_client.audio = mock_openai_client.audio
                response = model.transcribe(
                    audio_file,
                    language="en",
                    prompt="This is a podcast about AI",
                )
                assert isinstance(response, TranscriptionResponse)
                assert response.text == "This is a test transcription"


def test_openai_transcribe_file_object(mock_openai_client):
    """Test OpenAI transcribe with file object."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI.__init__", return_value=None) as mock_init:
            with patch("openai.AsyncOpenAI.__init__", return_value=None) as mock_async_init:
                model = OpenAISpeechToTextModel()
                model.client.audio = mock_openai_client.audio
                model.async_client.audio = mock_openai_client.audio
                with open(__file__, "rb") as f:
                    response = model.transcribe(f)
                assert isinstance(response, TranscriptionResponse)
                assert response.text == "This is a test transcription"
