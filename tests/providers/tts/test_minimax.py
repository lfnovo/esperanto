"""Tests for the MiniMax text-to-speech provider."""

from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.factory import AIFactory
from esperanto.providers.tts.minimax import MiniMaxTextToSpeechModel

RAW_AUDIO = b"mock minimax audio"
TTS_RESPONSE = {
    "data": {
        "audio": RAW_AUDIO.hex(),
        "status": 2,
        "subtitle_file": "https://example.test/subtitles.json",
    },
    "extra_info": {
        "audio_length": 1250,
        "audio_format": "mp3",
        "usage_characters": 12,
    },
    "trace_id": "trace-123",
    "base_resp": {"status_code": 0, "status_msg": "success"},
}

VOICES_RESPONSE = {
    "system_voice": [
        {
            "voice_id": "English_Graceful_Lady",
            "voice_name": "Graceful Lady",
            "description": ["Expressive English system voice"],
        }
    ],
    "voice_cloning": [
        {
            "voice_id": "cloned-voice",
            "description": ["Account voice clone"],
        }
    ],
    "voice_generation": [
        {
            "voice_id": "generated-voice",
            "description": [],
        }
    ],
    "base_resp": {"status_code": 0, "status_msg": "success"},
}


def _response(status_code: int, payload: dict) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = ""
    return response


@pytest.fixture
def tts_model() -> MiniMaxTextToSpeechModel:
    model = MiniMaxTextToSpeechModel(api_key="test-key")
    model.client = Mock()
    model.async_client = AsyncMock()

    def post(url: str, **kwargs):
        if url.endswith("/v1/t2a_v2"):
            return _response(200, TTS_RESPONSE)
        if url.endswith("/v1/get_voice"):
            return _response(200, VOICES_RESPONSE)
        return _response(404, {"message": "not found"})

    async def async_post(url: str, **kwargs):
        return post(url, **kwargs)

    model.client.post.side_effect = post
    model.async_client.post.side_effect = async_post
    return model


def test_init_defaults(tts_model):
    assert tts_model.provider == "minimax"
    assert tts_model.model_name == "speech-2.8-hd"
    assert tts_model.base_url == "https://api.minimax.io"


def test_init_from_environment(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://custom.minimax.test/v1")

    model = MiniMaxTextToSpeechModel()

    assert model.api_key == "env-key"
    assert model.base_url == "https://custom.minimax.test/v1"
    assert model._build_url("t2a_v2") == "https://custom.minimax.test/v1/t2a_v2"


def test_mainland_china_base_url_is_normalized():
    model = MiniMaxTextToSpeechModel(
        api_key="test-key",
        base_url="https://api.minimaxi.com/v1",
    )
    assert model._build_url("t2a_v2") == "https://api.minimaxi.com/v1/t2a_v2"


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MiniMax API key not found"):
        MiniMaxTextToSpeechModel()


def test_factory_registration():
    model = AIFactory.create_text_to_speech(
        "minimax",
        config={"api_key": "test-key"},
    )
    assert isinstance(model, MiniMaxTextToSpeechModel)


def test_available_providers_include_minimax_tts():
    providers = AIFactory.get_available_providers()
    assert "minimax" in providers["language"]
    assert "minimax" in providers["text_to_speech"]
    assert "minimax" not in providers["speech_to_text"]


def test_get_models(tts_model):
    models = tts_model._get_models()
    assert models[0].id == "speech-2.8-hd"
    assert all(model.type == "text_to_speech" for model in models)


def test_available_voices(tts_model):
    voices = tts_model.available_voices

    assert set(voices) == {
        "English_Graceful_Lady",
        "cloned-voice",
        "generated-voice",
    }
    assert voices["English_Graceful_Lady"].name == "Graceful Lady"
    assert voices["cloned-voice"].description == "Account voice clone"
    assert voices["generated-voice"].description == "Generated voice"

    _ = tts_model.available_voices
    assert tts_model.client.post.call_count == 1
    assert tts_model.client.post.call_args.kwargs["json"] == {"voice_type": "all"}


def test_generate_speech_maps_common_and_native_parameters(tts_model):
    response = tts_model.generate_speech(
        "Hello MiniMax",
        voice="English_Graceful_Lady",
        response_format="wav",
        speed=1.2,
        volume=2,
        pitch=1,
        emotion="happy",
        sample_rate=32000,
        bitrate=128000,
        channels=2,
        language_boost="English",
        pronunciation_dict={"tone": ["API/A P I"]},
    )

    call = tts_model.client.post.call_args
    assert call.args[0] == "https://api.minimax.io/v1/t2a_v2"
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"
    payload = call.kwargs["json"]
    assert payload["model"] == "speech-2.8-hd"
    assert payload["text"] == "Hello MiniMax"
    assert payload["stream"] is False
    assert payload["output_format"] == "hex"
    assert payload["voice_setting"] == {
        "voice_id": "English_Graceful_Lady",
        "speed": 1.2,
        "vol": 2,
        "pitch": 1,
        "emotion": "happy",
    }
    assert payload["audio_setting"] == {
        "format": "wav",
        "sample_rate": 32000,
        "bitrate": 128000,
        "channel": 2,
    }
    assert payload["language_boost"] == "English"

    assert response.audio_data == RAW_AUDIO
    assert response.duration == 1.25
    assert response.content_type == "audio/wav"
    assert response.model == "speech-2.8-hd"
    assert response.voice == "English_Graceful_Lady"
    assert response.provider == "minimax"
    assert response.metadata["trace_id"] == "trace-123"
    assert response.metadata["subtitle_file"].endswith("subtitles.json")


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    response = await tts_model.agenerate_speech(
        "Hello MiniMax",
        voice="English_Graceful_Lady",
    )

    assert response.audio_data == RAW_AUDIO
    assert response.content_type == "audio/mpeg"
    assert tts_model.async_client.post.call_args.args[0].endswith("/v1/t2a_v2")


def test_generate_speech_saves_output(tts_model, tmp_path):
    output_file = tmp_path / "speech.mp3"
    tts_model.generate_speech(
        "Hello MiniMax",
        voice="English_Graceful_Lady",
        output_file=output_file,
    )
    assert output_file.read_bytes() == RAW_AUDIO


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stream": True}, "streaming TTS is not exposed"),
        ({"output_format": "url"}, "output_format must be 'hex'"),
    ],
)
def test_unsupported_response_modes(tts_model, kwargs, message):
    with pytest.raises(ValueError, match=message):
        tts_model.generate_speech(
            "Hello MiniMax",
            voice="English_Graceful_Lady",
            **kwargs,
        )


def test_api_error_in_successful_http_response(tts_model):
    tts_model.client.post.side_effect = None
    tts_model.client.post.return_value = _response(
        200,
        {"base_resp": {"status_code": 2013, "status_msg": "invalid input"}},
    )
    with pytest.raises(RuntimeError, match="MiniMax API error: invalid input"):
        tts_model.generate_speech(
            "Hello MiniMax",
            voice="English_Graceful_Lady",
        )


def test_missing_audio_data(tts_model):
    tts_model.client.post.side_effect = None
    tts_model.client.post.return_value = _response(
        200,
        {"data": None, "base_resp": {"status_code": 0, "status_msg": "success"}},
    )
    with pytest.raises(RuntimeError, match="did not include audio data"):
        tts_model.generate_speech(
            "Hello MiniMax",
            voice="English_Graceful_Lady",
        )
