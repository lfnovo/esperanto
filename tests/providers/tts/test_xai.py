"""Tests for the xAI TTS provider."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from esperanto.providers.tts.xai import XAITextToSpeechModel


@pytest.fixture
def mock_tts_audio_response():
    """Mock binary audio response data."""
    return b"mock audio data for testing"


@pytest.fixture
def mock_httpx_clients(mock_tts_audio_response):
    """Mock httpx clients for xAI TTS."""
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
        if url.endswith("/v1/tts"):
            return make_response(200, content=mock_tts_audio_response)
        return make_response(404, json_data={"error": {"message": "Not found"}})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/v1/tts"):
            return make_async_response(200, content=mock_tts_audio_response)
        return make_async_response(404, json_data={"error": {"message": "Not found"}})

    client.post.side_effect = mock_post_side_effect
    async_client.post.side_effect = mock_async_post_side_effect

    return client, async_client


@pytest.fixture
def tts_model(mock_httpx_clients):
    """Create a TTS model instance with mocked HTTP clients."""
    model = XAITextToSpeechModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init(tts_model):
    """Test model initialization."""
    assert tts_model.api_key == "test-key"
    assert tts_model.PROVIDER == "xai"


def test_init_missing_api_key(monkeypatch):
    """Test that missing API key raises ValueError."""
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY_TTS", raising=False)
    with pytest.raises(ValueError, match="xAI API key not found"):
        XAITextToSpeechModel()


def test_init_from_env_var(monkeypatch):
    """Test initialization from environment variable."""
    monkeypatch.setenv("XAI_API_KEY", "env-test-key")
    model = XAITextToSpeechModel()
    assert model.api_key == "env-test-key"


def test_init_from_tts_env_var(monkeypatch):
    """Test that TTS-specific env var takes priority."""
    monkeypatch.setenv("XAI_API_KEY", "generic-key")
    monkeypatch.setenv("XAI_API_KEY_TTS", "tts-key")
    model = XAITextToSpeechModel()
    assert model.api_key == "tts-key"


def test_init_default_base_url(tts_model):
    """Test that default base URL is used."""
    assert tts_model.base_url == "https://api.x.ai"


def test_init_custom_base_url():
    """Test initialization with custom base URL."""
    model = XAITextToSpeechModel(api_key="test-key", base_url="https://custom.api.com")
    assert model.base_url == "https://custom.api.com"


def test_init_base_url_from_env_var(monkeypatch):
    """Test base URL from environment variable."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://env-base.api.com")
    model = XAITextToSpeechModel()
    assert model.base_url == "https://env-base.api.com"


def test_init_base_url_tts_env_priority(monkeypatch):
    """Test that TTS-specific base URL env var takes priority."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://generic.api.com")
    monkeypatch.setenv("XAI_BASE_URL_TTS", "https://tts-specific.api.com")
    model = XAITextToSpeechModel()
    assert model.base_url == "https://tts-specific.api.com"


def test_get_headers(tts_model):
    """Test request headers."""
    headers = tts_model._get_headers()
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"


def test_build_url(tts_model):
    """Test URL building."""
    url = tts_model.build_url("v1/tts")
    assert url == "https://api.x.ai/v1/tts"


def test_build_url_trailing_slash():
    """Test URL building strips trailing slash from base URL."""
    model = XAITextToSpeechModel(api_key="test-key", base_url="https://custom.api.com/")
    url = model.build_url("v1/tts")
    assert url == "https://custom.api.com/v1/tts"


def test_build_url_strips_v1_suffix():
    """Test URL building strips /v1 suffix to avoid duplication."""
    model = XAITextToSpeechModel(api_key="test-key", base_url="https://api.x.ai/v1")
    url = model.build_url("v1/tts")
    assert url == "https://api.x.ai/v1/tts"


def test_build_url_strips_v1_with_trailing_slash():
    """Test URL building strips /v1/ suffix to avoid duplication."""
    model = XAITextToSpeechModel(api_key="test-key", base_url="https://api.x.ai/v1/")
    url = model.build_url("v1/tts")
    assert url == "https://api.x.ai/v1/tts"


def test_handle_error_success(tts_model):
    """Test that successful responses don't raise."""
    response = Mock()
    response.status_code = 200
    tts_model._handle_error(response)  # Should not raise


def test_handle_error_with_json(tts_model):
    """Test error handling with JSON error body."""
    response = Mock()
    response.status_code = 401
    response.json.return_value = {"error": {"message": "Invalid API key"}}
    with pytest.raises(RuntimeError, match="xAI API error: Invalid API key"):
        tts_model._handle_error(response)


def test_handle_error_without_json(tts_model):
    """Test error handling when response is not JSON."""
    response = Mock()
    response.status_code = 500
    response.json.side_effect = Exception("Not JSON")
    response.text = "Internal Server Error"
    with pytest.raises(RuntimeError, match="xAI API error: HTTP 500: Internal Server Error"):
        tts_model._handle_error(response)


def test_available_voices(tts_model):
    """Test getting available voices."""
    voices = tts_model.available_voices

    assert len(voices) == 5
    assert set(voices.keys()) == {"eve", "ara", "rex", "sal", "leo"}

    eve = voices["eve"]
    assert eve.name == "eve"
    assert eve.id == "eve"
    assert eve.gender == "FEMALE"

    rex = voices["rex"]
    assert rex.gender == "MALE"


def test_provider(tts_model):
    """Test provider property."""
    assert tts_model.provider == "xai"


def test_get_default_model(tts_model):
    """Test default model returns empty string."""
    assert tts_model._get_default_model() == ""


def test_get_models(tts_model):
    """Test models list is empty."""
    assert tts_model._get_models() == []


def test_generate_speech(tts_model):
    """Test synchronous speech generation."""
    response = tts_model.generate_speech(
        text="Hello world",
        voice="eve"
    )

    tts_model.client.post.assert_called_once()
    call_args = tts_model.client.post.call_args

    assert call_args[0][0].endswith("/v1/tts")

    json_payload = call_args[1]["json"]
    assert json_payload["voice_id"] == "eve"
    assert json_payload["text"] == "Hello world"
    assert json_payload["language"] == "auto"
    assert json_payload["output_format"]["codec"] == "mp3"

    assert response.audio_data == b"mock audio data for testing"
    assert response.content_type == "audio/mpeg"
    assert response.voice == "eve"
    assert response.provider == "xai"


def test_generate_speech_default_voice(tts_model):
    """Test that default voice is eve."""
    response = tts_model.generate_speech(text="Hello")

    call_args = tts_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["voice_id"] == "eve"


def test_generate_speech_with_language(tts_model):
    """Test speech generation with explicit language."""
    response = tts_model.generate_speech(
        text="Bonjour",
        voice="eve",
        language="fr"
    )

    call_args = tts_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["language"] == "fr"


def test_generate_speech_with_wav_format(tts_model):
    """Test speech generation with wav format."""
    response = tts_model.generate_speech(
        text="Hello",
        voice="rex",
        response_format="wav"
    )

    call_args = tts_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["output_format"]["codec"] == "wav"
    assert response.content_type == "audio/wav"


def test_generate_speech_codec_kwarg_overrides_format(tts_model):
    """Test that codec kwarg overrides response_format and content_type stays consistent."""
    response = tts_model.generate_speech(
        text="Hello",
        voice="eve",
        codec="wav"
    )

    call_args = tts_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["output_format"]["codec"] == "wav"
    assert response.content_type == "audio/wav"


def test_generate_speech_with_unknown_format(tts_model):
    """Test speech generation with unmapped format falls back."""
    response = tts_model.generate_speech(
        text="Hello",
        voice="eve",
        response_format="ogg"
    )
    assert response.content_type == "audio/ogg"


def test_generate_speech_saves_to_file(tts_model, tmp_path):
    """Test speech generation saves to file when output_file is specified."""
    output_file = tmp_path / "output.mp3"
    response = tts_model.generate_speech(
        text="Hello",
        voice="eve",
        output_file=output_file
    )

    assert output_file.exists()
    assert output_file.read_bytes() == b"mock audio data for testing"


def test_generate_speech_creates_parent_dirs(tts_model, tmp_path):
    """Test that parent directories are created for output file."""
    output_file = tmp_path / "subdir" / "nested" / "output.mp3"
    tts_model.generate_speech(
        text="Hello",
        voice="eve",
        output_file=output_file
    )

    assert output_file.exists()


def test_generate_speech_api_error(tts_model):
    """Test that API errors are raised as RuntimeError."""
    error_response = Mock()
    error_response.status_code = 400
    error_response.json.return_value = {"error": {"message": "Bad request"}}
    tts_model.client.post.side_effect = lambda *a, **kw: error_response

    with pytest.raises(RuntimeError, match="Failed to generate speech"):
        tts_model.generate_speech(text="Hello", voice="eve")


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    """Test asynchronous speech generation."""
    response = await tts_model.agenerate_speech(
        text="Hello world",
        voice="ara"
    )

    tts_model.async_client.post.assert_called_once()
    call_args = tts_model.async_client.post.call_args

    assert call_args[0][0].endswith("/v1/tts")

    json_payload = call_args[1]["json"]
    assert json_payload["voice_id"] == "ara"
    assert json_payload["text"] == "Hello world"
    assert json_payload["language"] == "auto"
    assert json_payload["output_format"]["codec"] == "mp3"

    assert response.audio_data == b"mock audio data for testing"
    assert response.content_type == "audio/mpeg"
    assert response.voice == "ara"
    assert response.provider == "xai"


@pytest.mark.asyncio
async def test_agenerate_speech_with_format(tts_model):
    """Test async speech generation with explicit format."""
    response = await tts_model.agenerate_speech(
        text="Hello",
        voice="sal",
        response_format="pcm"
    )

    call_args = tts_model.async_client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["output_format"]["codec"] == "pcm"
    assert response.content_type == "audio/pcm"


@pytest.mark.asyncio
async def test_agenerate_speech_saves_to_file(tts_model, tmp_path):
    """Test async speech generation saves to file."""
    output_file = tmp_path / "async_output.mp3"
    response = await tts_model.agenerate_speech(
        text="Hello",
        voice="leo",
        output_file=output_file
    )

    assert output_file.exists()
    assert output_file.read_bytes() == b"mock audio data for testing"


@pytest.mark.asyncio
async def test_agenerate_speech_api_error(tts_model):
    """Test that async API errors are raised as RuntimeError."""
    error_response = AsyncMock()
    error_response.status_code = 400
    error_response.json = Mock(return_value={"error": {"message": "Bad request"}})
    tts_model.async_client.post.side_effect = lambda *a, **kw: error_response

    with pytest.raises(RuntimeError, match="Failed to generate speech"):
        await tts_model.agenerate_speech(text="Hello", voice="eve")
