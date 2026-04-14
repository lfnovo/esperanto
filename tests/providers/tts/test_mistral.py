"""Tests for the Mistral TTS provider."""
import base64

import pytest
from unittest.mock import AsyncMock, Mock

from esperanto.providers.tts.mistral import MistralTextToSpeechModel


MOCK_AUDIO_BYTES = b"mock audio data for testing"


@pytest.fixture
def mock_tts_audio_response():
    """Mock base64-encoded JSON audio response data."""
    return {"audio_data": base64.b64encode(MOCK_AUDIO_BYTES).decode()}


@pytest.fixture
def mock_httpx_clients(mock_tts_audio_response):
    """Mock httpx clients for Mistral TTS."""
    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        return response

    def make_async_response(status_code, json_data=None):
        response = AsyncMock()
        response.status_code = status_code
        if json_data is not None:
            response.json = Mock(return_value=json_data)
        return response

    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_response(200, json_data=mock_tts_audio_response)
        return make_response(404, json_data={"error": {"message": "Not found"}})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_async_response(200, json_data=mock_tts_audio_response)
        return make_async_response(404, json_data={"error": {"message": "Not found"}})

    client.post.side_effect = mock_post_side_effect
    async_client.post.side_effect = mock_async_post_side_effect

    return client, async_client


@pytest.fixture
def tts_model(mock_httpx_clients):
    """Create a TTS model instance with mocked HTTP clients."""
    model = MistralTextToSpeechModel(
        api_key="test-key",
        model_name="voxtral-mini-tts-2603"
    )
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.model_name == "voxtral-mini-tts-2603"
    assert tts_model.PROVIDER == "mistral"


def test_generate_speech(tts_model):
    """Test synchronous speech generation."""
    response = tts_model.generate_speech(
        text="Hello world",
        voice="neutral_female"
    )

    tts_model.client.post.assert_called_once()
    call_args = tts_model.client.post.call_args

    assert call_args[0][0] == "https://api.mistral.ai/v1/audio/speech"

    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key"

    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "voxtral-mini-tts-2603"
    assert json_payload["voice_id"] == "neutral_female"
    assert json_payload["input"] == "Hello world"
    assert "voice" not in json_payload

    assert response.audio_data == MOCK_AUDIO_BYTES
    assert response.content_type == "audio/mp3"
    assert response.model == "voxtral-mini-tts-2603"
    assert response.voice == "neutral_female"
    assert response.provider == "mistral"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    """Test asynchronous speech generation."""
    response = await tts_model.agenerate_speech(
        text="Hello world",
        voice="fr_female"
    )

    tts_model.async_client.post.assert_called_once()
    call_args = tts_model.async_client.post.call_args

    assert call_args[0][0] == "https://api.mistral.ai/v1/audio/speech"

    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key"

    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "voxtral-mini-tts-2603"
    assert json_payload["voice_id"] == "fr_female"
    assert json_payload["input"] == "Hello world"

    assert response.audio_data == MOCK_AUDIO_BYTES
    assert response.content_type == "audio/mp3"
    assert response.model == "voxtral-mini-tts-2603"
    assert response.voice == "fr_female"
    assert response.provider == "mistral"


def test_generate_speech_with_response_format(tts_model):
    """Test speech generation with explicit response_format."""
    response = tts_model.generate_speech(
        text="Hello world",
        voice="neutral_female",
        response_format="wav"
    )

    call_args = tts_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["response_format"] == "wav"

    assert response.content_type == "audio/wav"


def test_available_voices(tts_model):
    """Test getting available voices."""
    voices = tts_model.available_voices

    assert len(voices) == 20

    voice = voices["neutral_female"]
    assert voice.name == "neutral_female"
    assert voice.id == "neutral_female"
    assert voice.gender == "FEMALE"

    voice_male = voices["fr_male"]
    assert voice_male.gender == "MALE"
    assert voice_male.language_code == "fr"


def test_models(tts_model):
    """Test that _get_models returns hardcoded model list."""
    models = tts_model._get_models()

    assert len(models) == 1
    assert models[0].id == "voxtral-mini-tts-2603"
    assert models[0].owned_by == "mistralai"


def test_missing_api_key(monkeypatch):
    """Test that missing API key raises ValueError or creates model with None key."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    model = MistralTextToSpeechModel()
    assert model.api_key is None
