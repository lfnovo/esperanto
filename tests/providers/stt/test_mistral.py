"""Tests for Mistral speech-to-text provider."""

import io
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import TranscriptionResponse
from esperanto.factory import AIFactory
from esperanto.providers.stt.base import _guess_audio_content_type
from esperanto.providers.stt.mistral import MistralSpeechToTextModel


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    f = tmp_path / "test.mp3"
    f.write_bytes(b"mock audio content")
    return str(f)


@pytest.fixture
def wav_file(tmp_path):
    f = tmp_path / "test.wav"
    f.write_bytes(b"mock wav content")
    return str(f)


@pytest.fixture
def flac_file(tmp_path):
    f = tmp_path / "test.flac"
    f.write_bytes(b"mock flac content")
    return str(f)


@pytest.fixture
def ogg_file(tmp_path):
    f = tmp_path / "test.ogg"
    f.write_bytes(b"mock ogg content")
    return str(f)


@pytest.fixture
def unknown_file(tmp_path):
    f = tmp_path / "test.xyz"
    f.write_bytes(b"mock unknown content")
    return str(f)


@pytest.fixture
def mock_transcription_response():
    return {
        "text": "This is a test transcription",
        "model": "voxtral-mini-latest",
        "language": "en",
        "segments": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_httpx_clients(mock_transcription_response):
    """Mock httpx sync and async clients."""
    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, data):
        r = Mock()
        r.status_code = status_code
        r.json.return_value = data
        return r

    def make_async_response(status_code, data):
        r = AsyncMock()
        r.status_code = status_code
        r.json = Mock(return_value=data)
        return r

    def post_side_effect(url, **kwargs):
        if url.endswith("/audio/transcriptions"):
            return make_response(200, mock_transcription_response)
        return make_response(404, {"error": {"message": "Not found"}})

    async def async_post_side_effect(url, **kwargs):
        if url.endswith("/audio/transcriptions"):
            return make_async_response(200, mock_transcription_response)
        return make_async_response(404, {"error": {"message": "Not found"}})

    client.post.side_effect = post_side_effect
    async_client.post.side_effect = async_post_side_effect

    return client, async_client


@pytest.fixture(autouse=True)
def mock_env():
    with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}):
        yield


def test_factory_creates_mistral_stt():
    model = AIFactory.create_stt("mistral")
    assert isinstance(model, MistralSpeechToTextModel)


def test_mistral_transcribe(audio_file, mock_httpx_clients):
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = model.transcribe(audio_file)

    model.client.post.assert_called_once()
    call_args = model.client.post.call_args

    assert call_args[0][0] == "https://api.mistral.ai/v1/audio/transcriptions"

    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Accept"] == "application/json"

    assert "files" in call_args[1]
    assert "data" in call_args[1]
    assert call_args[1]["data"]["model"] == "voxtral-mini-latest"

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"
    assert response.model == "voxtral-mini-latest"
    assert response.provider == "mistral"
    assert response.language == "en"


@pytest.mark.asyncio
async def test_mistral_atranscribe(audio_file, mock_httpx_clients):
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = await model.atranscribe(audio_file)

    model.async_client.post.assert_called_once()
    call_args = model.async_client.post.call_args

    assert call_args[0][0] == "https://api.mistral.ai/v1/audio/transcriptions"

    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Accept"] == "application/json"

    assert "files" in call_args[1]
    assert "data" in call_args[1]
    assert call_args[1]["data"]["model"] == "voxtral-mini-latest"

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"
    assert response.model == "voxtral-mini-latest"
    assert response.provider == "mistral"
    assert response.language == "en"


def test_mistral_transcribe_with_options(audio_file, mock_httpx_clients):
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = model.transcribe(audio_file, language="fr", prompt="Podcast sur l'IA")

    call_args = model.client.post.call_args
    data = call_args[1]["data"]
    assert data["model"] == "voxtral-mini-latest"
    assert data["language"] == "fr"
    assert data["prompt"] == "Podcast sur l'IA"

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


def test_mistral_transcribe_file_object(mock_httpx_clients):
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    with open(__file__, "rb") as f:
        response = model.transcribe(f)

    model.client.post.assert_called_once()
    call_args = model.client.post.call_args
    assert call_args[0][0] == "https://api.mistral.ai/v1/audio/transcriptions"
    assert "files" in call_args[1]
    assert "data" in call_args[1]

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


def test_mistral_error_handling(audio_file):
    """Test that 4xx responses raise RuntimeError with provider message."""
    model = MistralSpeechToTextModel(api_key="test-key")

    error_response = Mock()
    error_response.status_code = 401
    error_response.json.return_value = {"error": {"message": "Unauthorized"}}

    model.client = Mock()
    model.client.post.return_value = error_response

    with pytest.raises(RuntimeError, match="Mistral API error: Unauthorized"):
        model.transcribe(audio_file)


def test_mistral_default_model():
    model = MistralSpeechToTextModel(api_key="test-key")
    assert model._get_default_model() == "voxtral-mini-latest"
    assert model.get_model_name() == "voxtral-mini-latest"


def test_mistral_get_models():
    model = MistralSpeechToTextModel(api_key="test-key")
    models = model._get_models()
    model_ids = [m.id for m in models]
    assert "voxtral-mini-latest" in model_ids
    assert "voxtral-small-latest" in model_ids


def test_mistral_provider_name():
    model = MistralSpeechToTextModel(api_key="test-key")
    assert model.provider == "mistral"


def test_mistral_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Mistral API key not found"):
            MistralSpeechToTextModel(api_key=None)


def test_mistral_transcribe_wav_content_type(wav_file, mock_httpx_clients):
    """Test that .wav file uses correct audio MIME type."""
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(wav_file)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == _guess_audio_content_type(wav_file)
    assert content_type.startswith("audio/")


def test_mistral_transcribe_flac_content_type(flac_file, mock_httpx_clients):
    """Test that .flac file uses correct audio MIME type."""
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(flac_file)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == _guess_audio_content_type(flac_file)
    assert content_type.startswith("audio/")


def test_mistral_transcribe_ogg_content_type(ogg_file, mock_httpx_clients):
    """Test that .ogg file uses audio/ogg MIME type."""
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(ogg_file)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == "audio/ogg"


def test_mistral_transcribe_unknown_extension_falls_back(unknown_file, mock_httpx_clients):
    """Test that unknown extension falls back to audio/mpeg."""
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(unknown_file)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == "audio/mpeg"


def test_mistral_transcribe_binaryio_mp3_name_uses_mpeg(mock_httpx_clients):
    """Test that BinaryIO with .mp3 name uses audio/mpeg."""
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    stream = io.BytesIO(b"mock audio")
    stream.name = "audio.mp3"
    model.transcribe(stream)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == "audio/mpeg"


@pytest.mark.asyncio
async def test_mistral_atranscribe_wav_content_type(wav_file, mock_httpx_clients):
    """Test that async .wav transcribe uses correct audio MIME type."""
    model = MistralSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    await model.atranscribe(wav_file)

    call_args = model.async_client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == _guess_audio_content_type(wav_file)
    assert content_type.startswith("audio/")
