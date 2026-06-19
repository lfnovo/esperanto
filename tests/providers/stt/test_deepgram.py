"""Tests for Deepgram speech-to-text provider."""

import io
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import TranscriptionResponse, TranscriptionSegment
from esperanto.factory import AIFactory
from esperanto.providers.stt.deepgram import DeepgramSpeechToTextModel


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
def mock_transcription_response():
    return {
        "metadata": {"duration": 3.5, "model": "nova-3"},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {"transcript": "This is a test transcription", "confidence": 0.99}
                    ]
                }
            ]
        },
    }


@pytest.fixture
def mock_verbose_response():
    return {
        "metadata": {"duration": 3.5, "model": "nova-3"},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {"transcript": "Hello world. This is a test.", "confidence": 0.99}
                    ]
                }
            ],
            "utterances": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hello world.",
                    "channel": 0,
                    "confidence": 0.97,
                    "id": "utt-001",
                    "speaker": 0,
                },
                {
                    "start": 1.5,
                    "end": 3.5,
                    "text": "This is a test.",
                    "channel": 0,
                    "confidence": 0.95,
                    "id": "utt-002",
                    "speaker": 0,
                },
            ],
        },
    }


def _make_sync_response(status_code: int, data: dict) -> Mock:
    r = Mock()
    r.status_code = status_code
    r.json.return_value = data
    return r


def _make_async_response(status_code: int, data: dict) -> AsyncMock:
    r = AsyncMock()
    r.status_code = status_code
    r.json = Mock(return_value=data)
    return r


@pytest.fixture
def mock_httpx_clients(mock_transcription_response):
    client = Mock()
    async_client = AsyncMock()

    client.post.return_value = _make_sync_response(200, mock_transcription_response)

    async def async_post(*args, **kwargs):
        return _make_async_response(200, mock_transcription_response)

    async_client.post.side_effect = async_post

    return client, async_client


@pytest.fixture
def mock_verbose_clients(mock_verbose_response):
    client = Mock()
    async_client = AsyncMock()

    client.post.return_value = _make_sync_response(200, mock_verbose_response)

    async def async_post(*args, **kwargs):
        return _make_async_response(200, mock_verbose_response)

    async_client.post.side_effect = async_post

    return client, async_client


@pytest.fixture(autouse=True)
def mock_env():
    with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test-key"}):
        yield


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_creates_deepgram_stt():
    model = AIFactory.create_stt("deepgram")
    assert isinstance(model, DeepgramSpeechToTextModel)


def test_factory_creates_deepgram_stt_via_speech_to_text():
    model = AIFactory.create_speech_to_text("deepgram")
    assert isinstance(model, DeepgramSpeechToTextModel)


def test_factory_creates_deepgram_stt_with_model():
    model = AIFactory.create_speech_to_text("deepgram", "nova-3")
    assert isinstance(model, DeepgramSpeechToTextModel)
    assert model.get_model_name() == "nova-3"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_request_url(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(audio_file)

    call_args = model.client.post.call_args
    assert call_args[0][0] == "https://api.deepgram.com/v1/listen"


def test_request_auth_header(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(audio_file)

    call_args = model.client.post.call_args
    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Token test-key"


def test_request_utterances_param(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(audio_file)

    call_args = model.client.post.call_args
    params = call_args[1]["params"]
    assert "utterances" in params
    assert params["utterances"]


def test_request_model_param(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(audio_file)

    call_args = model.client.post.call_args
    params = call_args[1]["params"]
    assert params["model"] == model.get_model_name()


# ---------------------------------------------------------------------------
# Basic transcription (file path)
# ---------------------------------------------------------------------------


def test_transcribe_file_path(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = model.transcribe(audio_file)

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"
    assert response.provider == "deepgram"
    assert response.model == "nova-3"


# ---------------------------------------------------------------------------
# BinaryIO input
# ---------------------------------------------------------------------------


def test_transcribe_binaryio(mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    stream = io.BytesIO(b"mock audio bytes")
    stream.name = "audio.mp3"
    response = model.transcribe(stream)

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"

    call_args = model.client.post.call_args
    headers = call_args[1]["headers"]
    assert headers["Content-Type"] == "audio/mpeg"


def test_transcribe_binaryio_wav(mock_httpx_clients, tmp_path):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    stream = io.BytesIO(b"mock wav audio")
    stream.name = "audio.wav"
    response = model.transcribe(stream)

    assert isinstance(response, TranscriptionResponse)
    call_args = model.client.post.call_args
    headers = call_args[1]["headers"]
    assert headers["Content-Type"].startswith("audio/")


# ---------------------------------------------------------------------------
# Language param forwarding
# ---------------------------------------------------------------------------


def test_transcribe_with_language(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(audio_file, language="es")

    call_args = model.client.post.call_args
    params = call_args[1]["params"]
    assert params["language"] == "es"


def test_transcribe_without_language_no_param(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    model.transcribe(audio_file)

    call_args = model.client.post.call_args
    params = call_args[1]["params"]
    assert "language" not in params


# ---------------------------------------------------------------------------
# Segments present path
# ---------------------------------------------------------------------------


def test_segments_present(audio_file, mock_verbose_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_verbose_clients

    response = model.transcribe(audio_file)

    assert response.segments is not None
    assert len(response.segments) == 2

    first = response.segments[0]
    assert isinstance(first, TranscriptionSegment)
    assert first.text == "Hello world."
    assert first.start == pytest.approx(0.0)
    assert first.end == pytest.approx(1.5)

    second = response.segments[1]
    assert second.text == "This is a test."
    assert second.start == pytest.approx(1.5)
    assert second.end == pytest.approx(3.5)


def test_segments_metadata_extras(audio_file, mock_verbose_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_verbose_clients

    response = model.transcribe(audio_file)

    assert response.segments is not None
    md = response.segments[0].metadata
    assert md is not None
    assert md["confidence"] == pytest.approx(0.97)
    assert md["channel"] == 0
    assert md["id"] == "utt-001"
    assert md["speaker"] == 0


# ---------------------------------------------------------------------------
# Segments absent path
# ---------------------------------------------------------------------------


def test_segments_absent_when_no_utterances_field(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = model.transcribe(audio_file)

    assert response.segments is None


def test_segments_absent_when_utterances_empty(audio_file):
    model = DeepgramSpeechToTextModel(api_key="test-key")

    empty_utterances_response = {
        "metadata": {"duration": 1.0},
        "results": {
            "channels": [{"alternatives": [{"transcript": "hi"}]}],
            "utterances": [],
        },
    }
    mock_resp = _make_sync_response(200, empty_utterances_response)
    model.client = Mock()
    model.client.post.return_value = mock_resp

    response = model.transcribe(audio_file)

    assert response.segments is None


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_value_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Deepgram API key not found"):
            DeepgramSpeechToTextModel(api_key=None)


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


def test_http_error_raises_runtime_error(audio_file):
    model = DeepgramSpeechToTextModel(api_key="test-key")

    error_response = Mock()
    error_response.status_code = 401
    error_response.json.return_value = {"err_msg": "Unauthorized"}

    model.client = Mock()
    model.client.post.return_value = error_response

    with pytest.raises(RuntimeError, match="Deepgram API error: Unauthorized"):
        model.transcribe(audio_file)


def test_http_403_raises_runtime_error(audio_file):
    model = DeepgramSpeechToTextModel(api_key="test-key")

    error_response = Mock()
    error_response.status_code = 403
    error_response.json.return_value = {"err_msg": "Forbidden"}

    model.client = Mock()
    model.client.post.return_value = error_response

    with pytest.raises(RuntimeError, match="Deepgram API error"):
        model.transcribe(audio_file)


# ---------------------------------------------------------------------------
# Async path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_atranscribe_file_path(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = await model.atranscribe(audio_file)

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "This is a test transcription"
    assert response.provider == "deepgram"
    assert response.model == "nova-3"


@pytest.mark.asyncio
async def test_atranscribe_request_shape(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    await model.atranscribe(audio_file)

    model.async_client.post.assert_called_once()
    call_args = model.async_client.post.call_args
    assert call_args[0][0] == "https://api.deepgram.com/v1/listen"
    assert call_args[1]["headers"]["Authorization"] == "Token test-key"
    assert "utterances" in call_args[1]["params"]


@pytest.mark.asyncio
async def test_atranscribe_segments_present(audio_file, mock_verbose_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_verbose_clients

    response = await model.atranscribe(audio_file)

    assert response.segments is not None
    assert len(response.segments) == 2
    assert response.segments[0].text == "Hello world."
    assert response.segments[0].start == pytest.approx(0.0)
    assert response.segments[0].end == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_atranscribe_segments_absent(audio_file, mock_httpx_clients):
    model = DeepgramSpeechToTextModel(api_key="test-key")
    model.client, model.async_client = mock_httpx_clients

    response = await model.atranscribe(audio_file)

    assert response.segments is None


# ---------------------------------------------------------------------------
# Default model and model list
# ---------------------------------------------------------------------------


def test_default_model():
    model = DeepgramSpeechToTextModel(api_key="test-key")
    assert model._get_default_model() == "nova-3"
    assert model.get_model_name() == "nova-3"


def test_get_models_includes_nova():
    model = DeepgramSpeechToTextModel(api_key="test-key")
    models = model._get_models()
    model_ids = [m.id for m in models]
    assert "nova-3" in model_ids
    assert "nova-2" in model_ids


def test_get_models_includes_whisper_variants():
    model = DeepgramSpeechToTextModel(api_key="test-key")
    models = model._get_models()
    model_ids = [m.id for m in models]
    for variant in ("whisper-large", "whisper-medium", "whisper-small", "whisper-base", "whisper-tiny"):
        assert variant in model_ids


def test_provider_name():
    model = DeepgramSpeechToTextModel(api_key="test-key")
    assert model.provider == "deepgram"


# ---------------------------------------------------------------------------
# Top-level package import
# ---------------------------------------------------------------------------


def test_importable_from_package():
    import esperanto

    assert hasattr(esperanto, "DeepgramSpeechToTextModel")
    assert "DeepgramSpeechToTextModel" in esperanto.__all__
