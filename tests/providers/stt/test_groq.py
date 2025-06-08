"""Tests for Groq speech-to-text provider."""

import os

import pytest

from esperanto.common_types import TranscriptionResponse
from esperanto.factory import AIFactory
from esperanto.providers.stt.groq import GroqSpeechToTextModel


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"mock audio content")
    return str(audio_file)




def test_factory_creates_groq_stt():
    """Test that AIFactory creates Groq STT model."""
    from unittest.mock import patch
    import os
    
    # Mock the environment variable to provide an API key
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}):
        model = AIFactory.create_speech_to_text("groq")
        assert isinstance(model, GroqSpeechToTextModel)


def test_groq_transcribe(audio_file):
    """Test Groq transcribe method with httpx mocking."""
    from unittest.mock import Mock
    
    # Create fresh model instance
    model = GroqSpeechToTextModel(api_key="test-key")
    
    # Mock OpenAI API response data
    mock_response_data = {
        "text": "This is a test transcription"
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    response = model.transcribe(audio_file)
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


@pytest.mark.asyncio
async def test_groq_atranscribe(audio_file):
    """Test Groq async transcribe method with httpx mocking."""
    from unittest.mock import Mock, AsyncMock
    
    # Create fresh model instance
    model = GroqSpeechToTextModel(api_key="test-key")
    
    # Mock OpenAI API response data
    mock_response_data = {
        "text": "This is a test transcription"
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the async client
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    model.async_client = mock_async_client

    response = await model.atranscribe(audio_file)
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


def test_groq_transcribe_with_options(audio_file):
    """Test Groq transcribe with language and prompt using httpx mocking."""
    from unittest.mock import Mock
    
    # Create fresh model instance
    model = GroqSpeechToTextModel(api_key="test-key")
    
    # Mock OpenAI API response data
    mock_response_data = {
        "text": "This is a test transcription"
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    response = model.transcribe(
        audio_file,
        language="en",
        prompt="This is a podcast about AI",
    )
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


def test_groq_transcribe_file_object():
    """Test Groq transcribe with file object using httpx mocking."""
    from unittest.mock import Mock
    
    # Create fresh model instance
    model = GroqSpeechToTextModel(api_key="test-key")
    
    # Mock OpenAI API response data
    mock_response_data = {
        "text": "This is a test transcription"
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    with open(__file__, "rb") as f:
        response = model.transcribe(f)
    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"
