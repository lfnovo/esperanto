"""Tests for ElevenLabs speech-to-text provider."""

import io
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import TranscriptionResponse
from esperanto.factory import AIFactory
from esperanto.providers.stt.base import _guess_audio_content_type
from esperanto.providers.stt.elevenlabs import ElevenLabsSpeechToTextModel


@pytest.fixture(autouse=True)
def mock_env():
    with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
        yield


@pytest.fixture
def audio_file(tmp_path):
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
    return {"text": "This is a test transcription"}


@pytest.fixture
def mock_httpx_clients(mock_transcription_response):
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
        if url.endswith("/speech-to-text"):
            return make_response(200, mock_transcription_response)
        return make_response(404, {"error": {"message": "Not found"}})

    async def async_post_side_effect(url, **kwargs):
        if url.endswith("/speech-to-text"):
            return make_async_response(200, mock_transcription_response)
        return make_async_response(404, {"error": {"message": "Not found"}})

    client.post.side_effect = post_side_effect
    async_client.post.side_effect = async_post_side_effect

    return client, async_client


def test_factory_creates_elevenlabs_stt():
    model = AIFactory.create_stt("elevenlabs")
    assert isinstance(model, ElevenLabsSpeechToTextModel)


def test_elevenlabs_transcribe(audio_file, mock_httpx_clients):
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = model.transcribe(audio_file)

    model.client.post.assert_called_once()
    call_args = model.client.post.call_args
    assert call_args[0][0] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert "files" in call_args[1]
    assert "data" in call_args[1]
    assert call_args[1]["data"]["model_id"] == "scribe_v1"

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


@pytest.mark.asyncio
async def test_elevenlabs_atranscribe(audio_file, mock_httpx_clients):
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = await model.atranscribe(audio_file)

    model.async_client.post.assert_called_once()
    call_args = model.async_client.post.call_args
    assert call_args[0][0] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert "files" in call_args[1]

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"


def test_elevenlabs_transcribe_wav_content_type(wav_file, mock_httpx_clients):
    """Test that .wav file uses correct audio MIME type."""
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(wav_file)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == _guess_audio_content_type(wav_file)
    assert content_type.startswith("audio/")


def test_elevenlabs_transcribe_flac_content_type(flac_file, mock_httpx_clients):
    """Test that .flac file uses correct audio MIME type."""
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(flac_file)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == _guess_audio_content_type(flac_file)
    assert content_type.startswith("audio/")


def test_elevenlabs_transcribe_ogg_content_type(ogg_file, mock_httpx_clients):
    """Test that .ogg file uses audio/ogg MIME type."""
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(ogg_file)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == "audio/ogg"


def test_elevenlabs_transcribe_unknown_extension_falls_back(unknown_file, mock_httpx_clients):
    """Test that unknown extension falls back to audio/mpeg."""
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(unknown_file)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == "audio/mpeg"


def test_elevenlabs_transcribe_binaryio_mp3_name_uses_mpeg(mock_httpx_clients):
    """Test that BinaryIO with .mp3 name uses audio/mpeg."""
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    stream = io.BytesIO(b"mock audio")
    stream.name = "audio.mp3"
    model.transcribe(stream)

    call_args = model.client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == "audio/mpeg"


@pytest.mark.asyncio
async def test_elevenlabs_atranscribe_wav_content_type(wav_file, mock_httpx_clients):
    """Test that async .wav transcribe uses correct audio MIME type."""
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    await model.atranscribe(wav_file)

    call_args = model.async_client.post.call_args
    content_type = call_args[1]["files"]["file"][2]
    assert content_type == _guess_audio_content_type(wav_file)
    assert content_type.startswith("audio/")


def test_elevenlabs_provider_name():
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    assert model.provider == "elevenlabs"


def test_elevenlabs_default_model():
    model = ElevenLabsSpeechToTextModel(api_key="test-key")
    assert model._get_default_model() == "scribe_v1"


def test_elevenlabs_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="ElevenLabs API key not found"):
            ElevenLabsSpeechToTextModel(api_key=None)
