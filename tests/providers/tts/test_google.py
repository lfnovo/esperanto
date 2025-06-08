"""Tests for the Google TTS provider."""
import base64
from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.providers.tts.google import GoogleTextToSpeechModel


def test_init():
    """Test model initialization."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    assert model.provider == "google"


def test_generate_speech():
    """Test synchronous speech generation with httpx mocking."""
    from unittest.mock import Mock

    from esperanto.providers.tts.google import GoogleTextToSpeechModel
    
    # Create fresh model instance
    model = GoogleTextToSpeechModel(api_key="test-key")
    
    # Mock Google API response data
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "data": base64.b64encode(b"test audio data").decode()
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the client
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    model.client = mock_client

    response = model.generate_speech(
        text="Hello world",
        voice="en-US-Standard-A"
    )

    assert b"test audio data" in response.audio_data  # Original data is embedded in WAV format
    assert response.content_type == "audio/wav"
    assert response.model == "gemini-2.5-flash-preview-tts"
    assert response.voice == "en-US-Standard-A"
    assert response.provider == "google"


@pytest.mark.asyncio
async def test_agenerate_speech():
    """Test asynchronous speech generation with httpx mocking."""
    from unittest.mock import AsyncMock, Mock

    from esperanto.providers.tts.google import GoogleTextToSpeechModel
    
    # Create fresh model instance
    model = GoogleTextToSpeechModel(api_key="test-key")
    
    # Mock Google API response data
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "data": base64.b64encode(b"test audio data").decode()
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    
    # Mock the async client
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    model.async_client = mock_async_client

    response = await model.agenerate_speech(
        text="Hello world",
        voice="en-US-Standard-A"
    )

    assert b"test audio data" in response.audio_data  # Original data is embedded in WAV format
    assert response.content_type == "audio/wav"
    assert response.model == "gemini-2.5-flash-preview-tts"
    assert response.voice == "en-US-Standard-A"
    assert response.provider == "google"


def test_available_voices():
    """Test getting available voices (predefined list)."""
    from esperanto.providers.tts.google import GoogleTextToSpeechModel
    
    # Create fresh model instance
    model = GoogleTextToSpeechModel(api_key="test-key")

    # Test getting voices (Google TTS uses predefined voices)
    voices = model.available_voices
    assert len(voices) == 30  # Google has 30 predefined voices
    
    # Test a few specific voices to ensure structure is correct
    assert "achernar" in voices
    assert voices["achernar"].name == "UpbeatAchernar"
    assert voices["achernar"].id == "achernar"
    assert voices["achernar"].gender == "FEMALE"
    
    assert "charon" in voices
    assert voices["charon"].name == "UpbeatCharon"
    assert voices["charon"].id == "charon"
    assert voices["charon"].gender == "MALE"
