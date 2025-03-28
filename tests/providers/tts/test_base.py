"""Tests for the base TTS provider."""

from pathlib import Path

import pytest

from esperanto.common_types.tts import AudioResponse, Voice
from esperanto.providers.tts.base import TextToSpeechModel


def test_voice_class():
    """Test the Voice class attributes and normalization."""
    voice = Voice(
        name="Test Voice",
        id="test-voice-1",
        gender="FEMALE",
        language_code="en-US",
        description="A test voice",
        accent="American",
        age="young",
        use_case="general",
        preview_url="https://example.com/preview.mp3",
    )

    assert voice.name == "Test Voice"
    assert voice.id == "test-voice-1"
    assert voice.gender == "FEMALE"
    assert voice.language_code == "en-US"
    assert voice.description == "A test voice"
    assert voice.accent == "American"
    assert voice.age == "young"
    assert voice.use_case == "general"
    assert voice.preview_url == "https://example.com/preview.mp3"


def test_audio_response():
    """Test AudioResponse class."""
    audio_data = b"test audio data"
    response = AudioResponse(
        audio_data=audio_data,
        content_type="audio/mp3",
        model="test-model",
        voice="test-voice",
        provider="test-provider",
        metadata={"test": "metadata"},
    )

    assert response.audio_data == audio_data
    assert response.content_type == "audio/mp3"
    assert response.model == "test-model"
    assert response.voice == "test-voice"
    assert response.provider == "test-provider"
    assert response.metadata == {"test": "metadata"}
