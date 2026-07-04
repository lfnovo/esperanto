"""Tests for the OpenRouter STT provider."""
import base64
from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.providers.stt.openrouter import OpenRouterSpeechToTextModel

SAMPLE_AUDIO = b"fake audio bytes for testing"
TRANSCRIPTION_JSON = {
    "text": "Hello from OpenRouter.",
    "usage": {
        "seconds": 9.2,
        "total_tokens": 113,
        "input_tokens": 83,
        "output_tokens": 30,
        "cost": 0.000508,
    },
}


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)


@pytest.fixture
def mock_httpx_clients():
    """Mock httpx clients for OpenRouter STT."""
    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    def make_async_response(status_code, json_data):
        response = AsyncMock()
        response.status_code = status_code
        response.json = Mock(return_value=json_data)
        return response

    def mock_post(url, **kwargs):
        if url.endswith("/audio/transcriptions"):
            return make_response(200, TRANSCRIPTION_JSON)
        return make_response(404, {"error": {"message": "Not found"}})

    async def mock_async_post(url, **kwargs):
        if url.endswith("/audio/transcriptions"):
            return make_async_response(200, TRANSCRIPTION_JSON)
        return make_async_response(404, {"error": {"message": "Not found"}})

    client.post.side_effect = mock_post
    async_client.post.side_effect = mock_async_post
    return client, async_client


@pytest.fixture
def stt_model(mock_httpx_clients):
    model = OpenRouterSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients
    return model


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "sample.mp3"
    path.write_bytes(SAMPLE_AUDIO)
    return str(path)


def test_init(stt_model):
    assert stt_model.api_key == "test-key"
    assert stt_model.provider == "openrouter"


def test_init_missing_api_key():
    with pytest.raises(ValueError, match="OpenRouter API key not found"):
        OpenRouterSpeechToTextModel()


def test_init_from_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-test-key")
    model = OpenRouterSpeechToTextModel()
    assert model.api_key == "env-test-key"


def test_init_default_base_url(stt_model):
    assert stt_model.base_url == "https://openrouter.ai/api/v1"


def test_init_custom_base_url():
    model = OpenRouterSpeechToTextModel(api_key="test-key", base_url="https://custom.api.com/v1")
    assert model.base_url == "https://custom.api.com/v1"


def test_init_base_url_from_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://env-base.api.com/v1")
    model = OpenRouterSpeechToTextModel()
    assert model.base_url == "https://env-base.api.com/v1"


def test_init_base_url_strips_trailing_slash():
    model = OpenRouterSpeechToTextModel(api_key="test-key", base_url="https://openrouter.ai/api/v1/")
    assert model.base_url == "https://openrouter.ai/api/v1"


def test_get_headers(stt_model):
    headers = stt_model._get_headers()
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["HTTP-Referer"] == "https://github.com/lfnovo/esperanto"
    assert headers["X-Title"] == "Esperanto"


def test_handle_error_with_json(stt_model):
    response = Mock()
    response.status_code = 401
    response.json.return_value = {"error": {"message": "Invalid API key"}}
    with pytest.raises(RuntimeError, match="OpenRouter API error: Invalid API key"):
        stt_model._handle_error(response)


def test_handle_error_without_json(stt_model):
    response = Mock()
    response.status_code = 500
    response.json.side_effect = Exception("Not JSON")
    response.text = "Internal Server Error"
    with pytest.raises(RuntimeError, match="OpenRouter API error: HTTP 500"):
        stt_model._handle_error(response)


def test_default_model(stt_model):
    assert stt_model._get_default_model() == "openai/whisper-1"


def test_transcribe_posts_base64_json(stt_model, audio_path):
    """Transcription sends OpenRouter's JSON + base64 input_audio contract."""
    result = stt_model.transcribe(audio_path)

    stt_model.client.post.assert_called_once()
    call_args = stt_model.client.post.call_args
    assert call_args[0][0] == "https://openrouter.ai/api/v1/audio/transcriptions"

    payload = call_args[1]["json"]
    assert payload["model"] == "openai/whisper-1"
    assert payload["input_audio"]["data"] == base64.b64encode(SAMPLE_AUDIO).decode("ascii")
    assert payload["input_audio"]["format"] == "mp3"

    assert result.text == "Hello from OpenRouter."
    assert result.provider == "openrouter"
    assert result.model == "openai/whisper-1"
    # usage.seconds maps to both duration and input_seconds
    assert result.duration == 9.2
    assert result.usage.input_seconds == 9.2
    assert result.usage.total_tokens == 113
    assert result.usage.input_tokens == 83
    assert result.usage.output_tokens == 30


def test_transcribe_detects_wav_format(stt_model, tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(SAMPLE_AUDIO)
    stt_model.transcribe(str(path))
    payload = stt_model.client.post.call_args[1]["json"]
    assert payload["input_audio"]["format"] == "wav"


def test_transcribe_with_language(stt_model, audio_path):
    stt_model.transcribe(audio_path, language="en")
    payload = stt_model.client.post.call_args[1]["json"]
    assert payload["language"] == "en"


def test_transcribe_binary_io(stt_model, audio_path):
    with open(audio_path, "rb") as f:
        result = stt_model.transcribe(f)
    payload = stt_model.client.post.call_args[1]["json"]
    assert payload["input_audio"]["data"] == base64.b64encode(SAMPLE_AUDIO).decode("ascii")
    assert result.text == "Hello from OpenRouter."


def test_transcribe_api_error(stt_model, audio_path):
    error_response = Mock()
    error_response.status_code = 400
    error_response.json.return_value = {"error": {"message": "Bad request"}}
    stt_model.client.post.side_effect = lambda *a, **kw: error_response
    with pytest.raises(RuntimeError, match="OpenRouter API error: Bad request"):
        stt_model.transcribe(audio_path)


def test_transcribe_no_usage_block(stt_model, audio_path):
    """A response without usage still produces a valid TranscriptionResponse."""
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {"text": "no usage here"}
    stt_model.client.post.side_effect = lambda *a, **kw: resp
    result = stt_model.transcribe(audio_path)
    assert result.text == "no usage here"
    assert result.usage is None
    assert result.duration is None


@pytest.mark.asyncio
async def test_atranscribe(stt_model, audio_path):
    result = await stt_model.atranscribe(audio_path, language="en")

    stt_model.async_client.post.assert_called_once()
    call_args = stt_model.async_client.post.call_args
    assert call_args[0][0] == "https://openrouter.ai/api/v1/audio/transcriptions"
    payload = call_args[1]["json"]
    assert payload["input_audio"]["data"] == base64.b64encode(SAMPLE_AUDIO).decode("ascii")
    assert payload["language"] == "en"

    assert result.text == "Hello from OpenRouter."
    assert result.usage.total_tokens == 113


def test_get_models_returns_list(stt_model):
    """Model discovery is not filterable for STT; returns a list."""
    assert isinstance(stt_model._get_models(), list)
