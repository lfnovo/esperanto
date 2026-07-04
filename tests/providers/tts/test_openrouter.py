"""Tests for the OpenRouter TTS provider."""
from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.providers.tts.openrouter import OpenRouterTextToSpeechModel


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure a clean environment for every test."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)


@pytest.fixture
def mock_tts_audio_response():
    """Mock binary audio response data."""
    return b"mock audio data for testing"


@pytest.fixture
def mock_httpx_clients(mock_tts_audio_response):
    """Mock httpx clients for OpenRouter TTS."""
    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, content=None, json_data=None):
        response = Mock()
        response.status_code = status_code
        if content is not None:
            response.content = content
        if json_data is not None:
            response.json.return_value = json_data
        return response

    def make_async_response(status_code, content=None, json_data=None):
        response = AsyncMock()
        response.status_code = status_code
        if content is not None:
            response.content = content
        if json_data is not None:
            response.json = Mock(return_value=json_data)
        return response

    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_response(200, content=mock_tts_audio_response)
        return make_response(404, json_data={"error": {"message": "Not found"}})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_async_response(200, content=mock_tts_audio_response)
        return make_async_response(404, json_data={"error": {"message": "Not found"}})

    client.post.side_effect = mock_post_side_effect
    async_client.post.side_effect = mock_async_post_side_effect

    return client, async_client


@pytest.fixture
def tts_model(mock_httpx_clients):
    """Create a TTS model instance with mocked HTTP clients."""
    model = OpenRouterTextToSpeechModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.api_key == "test-key"
    assert tts_model.PROVIDER == "openrouter"


def test_init_missing_api_key():
    """Test that missing API key raises ValueError."""
    with pytest.raises(ValueError, match="OpenRouter API key not found"):
        OpenRouterTextToSpeechModel()


def test_init_from_env_var(monkeypatch):
    """Test initialization from environment variable."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-test-key")
    model = OpenRouterTextToSpeechModel()
    assert model.api_key == "env-test-key"


def test_init_default_base_url(tts_model):
    """Test that default base URL is the OpenRouter endpoint."""
    assert tts_model.base_url == "https://openrouter.ai/api/v1"


def test_init_custom_base_url():
    """Test initialization with custom base URL."""
    model = OpenRouterTextToSpeechModel(api_key="test-key", base_url="https://custom.api.com/v1")
    assert model.base_url == "https://custom.api.com/v1"


def test_init_base_url_from_env_var(monkeypatch):
    """Test base URL from environment variable."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://env-base.api.com/v1")
    model = OpenRouterTextToSpeechModel()
    assert model.base_url == "https://env-base.api.com/v1"


def test_init_base_url_strips_trailing_slash():
    """Trailing slash is stripped to avoid double-slash URLs."""
    model = OpenRouterTextToSpeechModel(api_key="test-key", base_url="https://openrouter.ai/api/v1/")
    assert model.base_url == "https://openrouter.ai/api/v1"


def test_get_headers(tts_model):
    """Test request headers include auth and OpenRouter-required headers."""
    headers = tts_model._get_headers()
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["HTTP-Referer"] == "https://github.com/lfnovo/esperanto"
    assert headers["X-Title"] == "Esperanto"


def test_handle_error_with_json(tts_model):
    """Test error handling with JSON error body."""
    response = Mock()
    response.status_code = 401
    response.json.return_value = {"error": {"message": "Invalid API key"}}
    with pytest.raises(RuntimeError, match="OpenRouter API error: Invalid API key"):
        tts_model._handle_error(response)


def test_handle_error_without_json(tts_model):
    """Test error handling when response is not JSON."""
    response = Mock()
    response.status_code = 500
    response.json.side_effect = Exception("Not JSON")
    response.text = "Internal Server Error"
    with pytest.raises(RuntimeError, match="OpenRouter API error: HTTP 500"):
        tts_model._handle_error(response)


def test_provider(tts_model):
    """Test provider property."""
    assert tts_model.provider == "openrouter"


def test_get_default_model(tts_model):
    """Test default model is a real OpenRouter speech model."""
    assert tts_model._get_default_model() == "microsoft/mai-voice-2"


def test_available_voices(tts_model):
    """Voice catalog reflects the default model's Microsoft neural voices."""
    voices = tts_model.available_voices
    assert "en-US-AvaNeural" in voices
    assert voices["en-US-AvaNeural"].id == "en-US-AvaNeural"
    # OpenAI's alloy/nova set does not apply to the default model
    assert "alloy" not in voices


def test_default_voice_used_when_omitted(tts_model):
    """When no voice is passed, a voice valid for the default model is sent."""
    tts_model.generate_speech(text="Hello")
    payload = tts_model.client.post.call_args[1]["json"]
    assert payload["voice"] == "en-US-AvaNeural"


def test_get_models_queries_speech_filter(tts_model):
    """Model discovery hits /models with the output_modalities=speech filter."""
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {"data": [{"id": "microsoft/mai-voice-2"}, {"id": "hexgrad/kokoro-82m"}]}
    client = Mock()
    client.get.return_value = resp
    tts_model.client = client

    models = tts_model._get_models()
    assert [m.id for m in models] == ["microsoft/mai-voice-2", "hexgrad/kokoro-82m"]
    assert client.get.call_args[1]["params"] == {"output_modalities": "speech"}
    assert models[0].owned_by == "microsoft"


def test_generate_speech(tts_model):
    """Test synchronous speech generation posts an OpenAI-compatible payload."""
    response = tts_model.generate_speech(text="Hello world", voice="en-US-AndrewNeural")

    tts_model.client.post.assert_called_once()
    call_args = tts_model.client.post.call_args
    assert call_args[0][0] == "https://openrouter.ai/api/v1/audio/speech"

    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "microsoft/mai-voice-2"
    assert json_payload["voice"] == "en-US-AndrewNeural"
    assert json_payload["input"] == "Hello world"
    assert json_payload["response_format"] == "mp3"

    # OpenRouter-required headers travel with the request
    headers = call_args[1]["headers"]
    assert headers["HTTP-Referer"] == "https://github.com/lfnovo/esperanto"
    assert headers["X-Title"] == "Esperanto"

    assert response.audio_data == b"mock audio data for testing"
    assert response.content_type == "audio/mp3"
    assert response.voice == "en-US-AndrewNeural"
    assert response.provider == "openrouter"


def test_generate_speech_uses_explicit_model():
    """Explicit model_name flows into the request payload."""
    model = OpenRouterTextToSpeechModel(api_key="test-key", model_name="mistralai/voxtral-mini-tts-2603")
    client = Mock()
    resp = Mock()
    resp.status_code = 200
    resp.content = b"audio"
    client.post.return_value = resp
    model.client = client

    model.generate_speech(text="Hi", voice="alloy")
    assert client.post.call_args[1]["json"]["model"] == "mistralai/voxtral-mini-tts-2603"


def test_generate_speech_saves_to_file(tts_model, tmp_path):
    """Test speech generation saves to file when output_file is specified."""
    output_file = tmp_path / "output.mp3"
    tts_model.generate_speech(text="Hello", voice="alloy", output_file=output_file)
    assert output_file.exists()
    assert output_file.read_bytes() == b"mock audio data for testing"


def test_generate_speech_api_error(tts_model):
    """Test that API errors are raised as RuntimeError."""
    error_response = Mock()
    error_response.status_code = 400
    error_response.json.return_value = {"error": {"message": "Bad request"}}
    tts_model.client.post.side_effect = lambda *a, **kw: error_response

    with pytest.raises(RuntimeError, match="Failed to generate speech"):
        tts_model.generate_speech(text="Hello", voice="alloy")


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    """Test asynchronous speech generation."""
    response = await tts_model.agenerate_speech(text="Hello world", voice="echo")

    tts_model.async_client.post.assert_called_once()
    call_args = tts_model.async_client.post.call_args
    assert call_args[0][0] == "https://openrouter.ai/api/v1/audio/speech"

    json_payload = call_args[1]["json"]
    assert json_payload["voice"] == "echo"
    assert json_payload["input"] == "Hello world"

    assert response.audio_data == b"mock audio data for testing"
    assert response.provider == "openrouter"


@pytest.mark.asyncio
async def test_agenerate_speech_saves_to_file(tts_model, tmp_path):
    """Test async speech generation saves to file."""
    output_file = tmp_path / "async_output.mp3"
    await tts_model.agenerate_speech(text="Hello", voice="alloy", output_file=output_file)
    assert output_file.exists()
    assert output_file.read_bytes() == b"mock audio data for testing"
