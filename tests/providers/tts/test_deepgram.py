"""Tests for the Deepgram TTS provider."""
import os
from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.providers.tts.deepgram import DEEPGRAM_VOICES, DeepgramTextToSpeechModel


@pytest.fixture
def mock_audio_bytes() -> bytes:
    return b"mock audio data for testing"


@pytest.fixture
def mock_httpx_clients(mock_audio_bytes: bytes):
    client = Mock()
    async_client = AsyncMock()

    def make_sync_response(status_code: int, content: bytes | None = None, json_data: dict | None = None):
        response = Mock()
        response.status_code = status_code
        if content is not None:
            response.content = content
        if json_data is not None:
            response.json.return_value = json_data
        return response

    def make_async_response(status_code: int, content: bytes | None = None, json_data: dict | None = None):
        response = AsyncMock()
        response.status_code = status_code
        if content is not None:
            response.content = content
        if json_data is not None:
            response.json = Mock(return_value=json_data)
        return response

    client.post.return_value = make_sync_response(200, content=mock_audio_bytes)
    async_client.post.return_value = make_async_response(200, content=mock_audio_bytes)

    return client, async_client


@pytest.fixture
def tts_model(mock_httpx_clients):
    model = DeepgramTextToSpeechModel(
        api_key="test-key",
        model_name="aura-2-thalia-en",
    )
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.model_name == "aura-2-thalia-en"
    assert tts_model.PROVIDER == "deepgram"
    assert tts_model.api_key == "test-key"


def test_generate_speech_default(tts_model, mock_audio_bytes):
    """Test default sync generate_speech POSTs to the correct URL with correct params."""
    response = tts_model.generate_speech(text="Hello world", voice="aura-2-thalia-en")

    tts_model.client.post.assert_called_once()
    call_args = tts_model.client.post.call_args

    # URL
    assert call_args[0][0] == "https://api.deepgram.com/v1/speak"

    # Auth header
    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Token test-key"

    # Query params
    params = call_args[1]["params"]
    assert params["model"] == "aura-2-thalia-en"
    assert params["encoding"] == "mp3"

    # JSON body
    assert call_args[1]["json"] == {"text": "Hello world"}

    # Response
    assert response.audio_data == mock_audio_bytes
    assert response.content_type == "audio/mp3"
    assert response.provider == "deepgram"


def test_generate_speech_voice_override(tts_model):
    """Voice override sends model=aura-2-orpheus-en in the query string and AudioResponse.model reports the actually-used voice."""
    response = tts_model.generate_speech(text="Hello world", voice="aura-2-orpheus-en")

    # AudioResponse.model must match what we actually sent upstream, not the instance default.
    assert response.model == "aura-2-orpheus-en"
    assert response.voice == "aura-2-orpheus-en"

    call_args = tts_model.client.post.call_args
    params = call_args[1]["params"]
    assert params["model"] == "aura-2-orpheus-en"


def test_generate_speech_voice_defaults_to_model_name(mock_httpx_clients):
    """When voice is omitted, the provider falls back to self.model_name (not a hard-coded default)."""
    model = DeepgramTextToSpeechModel(api_key="test-key", model_name="aura-2-orpheus-en")
    model.client, model.async_client = mock_httpx_clients

    model.generate_speech(text="Hello world")

    params = model.client.post.call_args[1]["params"]
    assert params["model"] == "aura-2-orpheus-en"


def test_generate_speech_encoding_override(tts_model):
    """Encoding override sends encoding=wav and response content_type=audio/wav."""
    response = tts_model.generate_speech(text="Hello world", voice="aura-2-thalia-en", encoding="wav")

    call_args = tts_model.client.post.call_args
    params = call_args[1]["params"]
    assert params["encoding"] == "wav"
    assert response.content_type == "audio/wav"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model, mock_audio_bytes):
    """Async generate_speech behaves identically to sync version."""
    response = await tts_model.agenerate_speech(text="Hello world", voice="aura-2-thalia-en")

    tts_model.async_client.post.assert_called_once()
    call_args = tts_model.async_client.post.call_args

    # URL
    assert call_args[0][0] == "https://api.deepgram.com/v1/speak"

    # Auth header
    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Token test-key"

    # Query params
    params = call_args[1]["params"]
    assert params["model"] == "aura-2-thalia-en"
    assert params["encoding"] == "mp3"

    # JSON body
    assert call_args[1]["json"] == {"text": "Hello world"}

    # Response
    assert response.audio_data == mock_audio_bytes
    assert response.content_type == "audio/mp3"
    assert response.provider == "deepgram"


def test_missing_api_key_raises_value_error():
    """Constructing without API key raises ValueError mentioning DEEPGRAM_API_KEY."""
    env_key = "DEEPGRAM_API_KEY"
    original = os.environ.pop(env_key, None)
    try:
        with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
            DeepgramTextToSpeechModel()
    finally:
        if original is not None:
            os.environ[env_key] = original


def test_error_response_raises_runtime_error(tts_model):
    """A non-200 Deepgram response raises RuntimeError with 'Deepgram API error'."""
    error_response = Mock()
    error_response.status_code = 400
    error_response.json.return_value = {
        "err_code": "BAD_REQUEST",
        "err_msg": "Invalid model specified",
    }
    tts_model.client.post.return_value = error_response

    with pytest.raises(RuntimeError, match="Deepgram API error"):
        tts_model.generate_speech(text="Hello world", voice="aura-2-thalia-en")


def test_available_voices_returns_dict():
    """available_voices returns a non-empty dict of Voice objects."""
    model = DeepgramTextToSpeechModel(api_key="test-key")
    voices = model.available_voices

    assert isinstance(voices, dict)
    assert len(voices) > 0

    # Spot-check a known voice
    assert "aura-2-thalia-en" in voices
    voice = voices["aura-2-thalia-en"]
    assert voice.id == "aura-2-thalia-en"
    assert voice.name == "Thalia"

    # Spot-check a non-English voice
    assert "aura-2-celeste-es" in voices

    # Verify all entries are from the static catalog
    assert voices == DEEPGRAM_VOICES
