"""Tests for the Mistral TTS provider."""
import base64
from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.providers.tts.mistral import MistralTextToSpeechModel

RAW_AUDIO = b"mock audio data"
B64_AUDIO = base64.b64encode(RAW_AUDIO).decode()

VOICES_RESPONSE = {
    "items": [
        {
            "id": "gb_jane_neutral",
            "name": "Jane",
            "gender": "NEUTRAL",
            "languages": ["en"],
        }
    ],
    "page": 1,
    "page_size": 1,
    "total": 1,
    "total_pages": 1,
}


@pytest.fixture
def mock_httpx_clients():
    """Mock httpx clients for Mistral TTS."""
    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        else:
            response.json.return_value = {}
        response.text = ""
        return response

    def make_async_response(status_code, json_data=None):
        response = AsyncMock()
        response.status_code = status_code
        if json_data is not None:
            response.json = Mock(return_value=json_data)
        else:
            response.json = Mock(return_value={})
        response.text = ""
        return response

    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_response(200, json_data={"audio_data": B64_AUDIO})
        return make_response(404, json_data={"error": {"message": "Not found"}})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/audio/voices"):
            return make_response(200, json_data=VOICES_RESPONSE)
        return make_response(404, json_data={"error": {"message": "Not found"}})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_async_response(200, json_data={"audio_data": B64_AUDIO})
        return make_async_response(404, json_data={"error": {"message": "Not found"}})

    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect
    async_client.post.side_effect = mock_async_post_side_effect

    return client, async_client


@pytest.fixture
def tts_model(mock_httpx_clients):
    model = MistralTextToSpeechModel(
        api_key="test-key",
        model_name="voxtral-mini-tts-2603",
    )
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init(tts_model):
    assert tts_model.model_name == "voxtral-mini-tts-2603"
    assert tts_model.provider == "mistral"
    assert tts_model.base_url == "https://api.mistral.ai/v1"


def test_generate_speech(tts_model):
    response = tts_model.generate_speech(
        text="Hello world",
        voice="gb_jane_neutral",
    )

    tts_model.client.post.assert_called_once()
    call_args = tts_model.client.post.call_args

    assert call_args[0][0] == "https://api.mistral.ai/v1/audio/speech"

    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key"

    payload = call_args[1]["json"]
    assert payload["input"] == "Hello world"
    assert payload["voice_id"] == "gb_jane_neutral"
    assert payload["model"] == "voxtral-mini-tts-2603"
    assert payload["response_format"] == "mp3"

    assert response.audio_data == RAW_AUDIO
    assert response.content_type == "audio/mp3"
    assert response.model == "voxtral-mini-tts-2603"
    assert response.voice == "gb_jane_neutral"
    assert response.provider == "mistral"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    response = await tts_model.agenerate_speech(
        text="Hello world",
        voice="en_paul_neutral",
    )

    tts_model.async_client.post.assert_called_once()
    call_args = tts_model.async_client.post.call_args

    assert call_args[0][0] == "https://api.mistral.ai/v1/audio/speech"

    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key"

    payload = call_args[1]["json"]
    assert payload["input"] == "Hello world"
    assert payload["voice_id"] == "en_paul_neutral"
    assert payload["model"] == "voxtral-mini-tts-2603"
    assert payload["response_format"] == "mp3"

    assert response.audio_data == RAW_AUDIO
    assert response.content_type == "audio/mp3"
    assert response.model == "voxtral-mini-tts-2603"
    assert response.voice == "en_paul_neutral"
    assert response.provider == "mistral"


def test_generate_speech_with_response_format(tts_model):
    response = tts_model.generate_speech(
        text="Hello world",
        voice="gb_jane_neutral",
        response_format="wav",
    )

    call_args = tts_model.client.post.call_args
    payload = call_args[1]["json"]
    assert payload["response_format"] == "wav"
    assert response.content_type == "audio/wav"


def test_available_voices(tts_model):
    voices = tts_model.available_voices

    tts_model.client.get.assert_called_once_with(
        "https://api.mistral.ai/v1/audio/voices",
        headers=tts_model._get_headers(),
        params={"page": 1},
    )

    assert "gb_jane_neutral" in voices
    voice = voices["gb_jane_neutral"]
    assert voice.id == "gb_jane_neutral"
    assert voice.name == "Jane"
    assert voice.gender == "NEUTRAL"
    assert voice.language_code == "en"


def test_available_voices_cached(tts_model):
    _ = tts_model.available_voices
    _ = tts_model.available_voices
    assert tts_model.client.get.call_count == 1


def test_error_handling_4xx(tts_model):
    tts_model.client.post.side_effect = None
    error_response = Mock()
    error_response.status_code = 400
    error_response.json.return_value = {"error": {"message": "Bad request"}}
    error_response.text = "Bad request"
    tts_model.client.post.return_value = error_response

    with pytest.raises(RuntimeError, match="Mistral API error"):
        tts_model.generate_speech(text="Hello", voice="gb_jane_neutral")


def test_error_handling_5xx(tts_model):
    tts_model.client.post.side_effect = None
    error_response = Mock()
    error_response.status_code = 500
    error_response.json.side_effect = Exception("no json")
    error_response.text = "Internal Server Error"
    tts_model.client.post.return_value = error_response

    with pytest.raises(RuntimeError, match="Mistral API error"):
        tts_model.generate_speech(text="Hello", voice="gb_jane_neutral")


def test_available_voices_paginated():
    voice_a = {"id": "voice_a", "name": "Voice A", "gender": "FEMALE", "languages": ["en"]}
    voice_b = {"id": "voice_b", "name": "Voice B", "gender": "MALE", "languages": ["fr"]}

    page1_response = Mock()
    page1_response.status_code = 200
    page1_response.json.return_value = {
        "items": [voice_a],
        "page": 1,
        "page_size": 1,
        "total": 2,
        "total_pages": 2,
    }

    page2_response = Mock()
    page2_response.status_code = 200
    page2_response.json.return_value = {
        "items": [voice_b],
        "page": 2,
        "page_size": 1,
        "total": 2,
        "total_pages": 2,
    }

    model = MistralTextToSpeechModel(api_key="test-key")
    model.client = Mock()
    model.client.get.side_effect = [page1_response, page2_response]

    voices = model.available_voices

    assert "voice_a" in voices
    assert "voice_b" in voices
    assert model.client.get.call_count == 2
