"""Tests for the CAMB AI TTS provider."""

import asyncio
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Mock the camb module before importing the provider
camb_mock = MagicMock()
camb_mock.StreamTtsOutputConfiguration = MagicMock
sys.modules.setdefault("camb", camb_mock)
sys.modules.setdefault("camb.client", MagicMock())

from esperanto.providers.tts.camb import CambTextToSpeechModel, _parse_voice_id


@pytest.fixture(autouse=True)
def set_api_key():
    """Set CAMB_API_KEY for all tests."""
    os.environ["CAMB_API_KEY"] = "test-key"
    yield
    os.environ.pop("CAMB_API_KEY", None)


@pytest.fixture
def tts_model():
    """Create a CAMB TTS model instance."""
    return CambTextToSpeechModel(api_key="test-key", model_name="mars-pro")


def _make_mock_client(chunks=None):
    """Create a mock CAMB client with a streaming TTS method."""
    if chunks is None:
        chunks = [b"audio"]

    captured_kwargs = {}

    async def mock_tts_stream(**kwargs):
        captured_kwargs.update(kwargs)
        for chunk in chunks:
            yield chunk

    mock_client = MagicMock()
    mock_client.text_to_speech.tts = mock_tts_stream
    return mock_client, captured_kwargs


class TestInit:
    """Test initialization."""

    def test_init_defaults(self, tts_model):
        assert tts_model.model_name == "mars-pro"
        assert tts_model.PROVIDER == "camb"
        assert tts_model.api_key == "test-key"
        assert tts_model.language == "en-us"

    def test_init_custom_language(self):
        model = CambTextToSpeechModel(api_key="test-key", language="fr-fr")
        assert model.language == "fr-fr"

    def test_init_default_model(self):
        model = CambTextToSpeechModel(api_key="test-key")
        assert model.model_name == "mars-pro"

    def test_init_missing_api_key(self):
        os.environ.pop("CAMB_API_KEY", None)
        with pytest.raises(ValueError, match="CAMB API key not provided"):
            CambTextToSpeechModel()

    def test_init_from_env(self):
        model = CambTextToSpeechModel()
        assert model.api_key == "test-key"


class TestGenerateSpeech:
    """Test speech generation."""

    @pytest.mark.asyncio
    async def test_agenerate_speech_streams_chunks(self, tts_model):
        """Test that agenerate_speech accumulates streaming chunks correctly."""
        mock_client, _ = _make_mock_client([b"chunk1", b"chunk2", b"chunk3"])
        tts_model._client_instance = mock_client

        response = await tts_model.agenerate_speech(
            text="Hello world", voice="147320"
        )

        assert response.audio_data == b"chunk1chunk2chunk3"
        assert response.content_type == "audio/mp3"
        assert response.model == "mars-pro"
        assert response.voice == "147320"
        assert response.provider == "camb"

    def test_generate_speech_sync(self, tts_model):
        """Test synchronous speech generation wraps async correctly."""
        mock_client, _ = _make_mock_client([b"audio_data"])
        tts_model._client_instance = mock_client

        response = tts_model.generate_speech(text="Hello world", voice="147320")

        assert response.audio_data == b"audio_data"
        assert response.provider == "camb"

    @pytest.mark.asyncio
    async def test_voice_id_converted_to_int(self, tts_model):
        """Test that string voice ID is converted to int for CAMB API."""
        mock_client, captured = _make_mock_client()
        tts_model._client_instance = mock_client

        await tts_model.agenerate_speech(text="Hello", voice="147320")

        assert captured["voice_id"] == 147320
        assert isinstance(captured["voice_id"], int)

    @pytest.mark.asyncio
    async def test_language_propagation(self, tts_model):
        """Test that language parameter is propagated correctly."""
        mock_client, captured = _make_mock_client()
        tts_model._client_instance = mock_client

        await tts_model.agenerate_speech(
            text="Bonjour", voice="147320", language="fr-fr"
        )

        assert captured["language"] == "fr-fr"

    @pytest.mark.asyncio
    async def test_default_language_used(self, tts_model):
        """Test that default language is used when not specified."""
        mock_client, captured = _make_mock_client()
        tts_model._client_instance = mock_client

        await tts_model.agenerate_speech(text="Hello", voice="147320")

        assert captured["language"] == "en-us"

    @pytest.mark.asyncio
    async def test_user_instructions_mars_instruct(self):
        """Test user_instructions only passed for mars-instruct model."""
        model = CambTextToSpeechModel(
            api_key="test-key", model_name="mars-instruct"
        )
        mock_client, captured = _make_mock_client()
        model._client_instance = mock_client

        await model.agenerate_speech(
            text="Hello",
            voice="147320",
            user_instructions="Speak slowly",
        )

        assert captured["user_instructions"] == "Speak slowly"

    @pytest.mark.asyncio
    async def test_user_instructions_ignored_non_instruct(self, tts_model):
        """Test user_instructions is ignored for non-instruct models."""
        mock_client, captured = _make_mock_client()
        tts_model._client_instance = mock_client

        await tts_model.agenerate_speech(
            text="Hello",
            voice="147320",
            user_instructions="Speak slowly",
        )

        assert "user_instructions" not in captured

    @pytest.mark.asyncio
    async def test_output_file_saves(self, tts_model, tmp_path):
        """Test that output file is saved correctly."""
        mock_client, _ = _make_mock_client([b"audio_data"])
        tts_model._client_instance = mock_client

        output_file = tmp_path / "output.mp3"
        response = await tts_model.agenerate_speech(
            text="Hello", voice="147320", output_file=str(output_file)
        )

        assert output_file.read_bytes() == b"audio_data"
        assert response.audio_data == b"audio_data"


class TestTranslatedTTS:
    """Test translated TTS functionality."""

    @pytest.mark.asyncio
    async def test_translated_tts_routes_correctly(self, tts_model):
        """Test that target_language + source_language triggers translation path."""
        mock_client = MagicMock()

        mock_result = MagicMock()
        mock_result.task_id = "task-123"
        mock_client.translated_tts.create_translated_tts = AsyncMock(
            return_value=mock_result
        )

        mock_status = MagicMock()
        mock_status.status = "completed"
        mock_status.run_id = "run-456"
        mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(
            return_value=mock_status
        )

        tts_model._client_instance = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"translated_audio"

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.get = AsyncMock(return_value=mock_resp)

        with patch("esperanto.providers.tts.camb.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_http_client

            response = await tts_model.agenerate_speech(
                text="Hello",
                voice="147320",
                source_language=1,
                target_language=2,
            )

        assert response.audio_data == b"translated_audio"
        mock_client.translated_tts.create_translated_tts.assert_called_once()


class TestAvailableVoices:
    """Test voice listing."""

    def test_available_voices(self, tts_model):
        """Test fetching and mapping voices."""
        mock_voice = MagicMock()
        mock_voice.id = 147320
        mock_voice.voice_name = "Test Voice"
        mock_voice.gender = 2  # female
        mock_voice.language = "en"
        mock_voice.age = 21

        mock_client = MagicMock()
        mock_client.voice_cloning.list_voices = AsyncMock(
            return_value=[mock_voice]
        )

        with patch.object(tts_model, "_create_camb_client", return_value=mock_client):
            voices = tts_model.available_voices

        assert "147320" in voices
        assert voices["147320"].name == "Test Voice"
        assert voices["147320"].id == "147320"
        assert voices["147320"].gender == "FEMALE"
        assert voices["147320"].age == "21"

    def test_voices_cached(self, tts_model):
        """Test that voices are cached after first fetch."""
        mock_voice = MagicMock()
        mock_voice.id = 1
        mock_voice.voice_name = "Voice"
        mock_voice.gender = 1
        mock_voice.language = "en"
        mock_voice.age = None

        mock_client = MagicMock()
        mock_client.voice_cloning.list_voices = AsyncMock(
            return_value=[mock_voice]
        )

        with patch.object(tts_model, "_create_camb_client", return_value=mock_client):
            voices1 = tts_model.available_voices
            voices2 = tts_model.available_voices

        # Should only create client once (cached after first call)
        assert voices1 is voices2

    def test_fetch_voices_api_failure_returns_empty(self, tts_model):
        """Test that API failure returns empty dict instead of crashing."""
        mock_client = MagicMock()
        mock_client.voice_cloning.list_voices = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with patch.object(tts_model, "_create_camb_client", return_value=mock_client):
            voices = tts_model.available_voices

        assert voices == {}

    def test_fetch_voices_dict_format(self, tts_model):
        """Test voice list items returned as dicts."""
        voice_dict = {
            "id": 99999,
            "voice_name": "Dict Voice",
            "gender": 1,
            "language": "fr",
            "age": 30,
        }

        mock_client = MagicMock()
        mock_client.voice_cloning.list_voices = AsyncMock(
            return_value=[voice_dict]
        )

        with patch.object(tts_model, "_create_camb_client", return_value=mock_client):
            voices = tts_model.available_voices

        assert "99999" in voices
        assert voices["99999"].name == "Dict Voice"
        assert voices["99999"].gender == "MALE"
        assert voices["99999"].age == "30"


class TestGetModels:
    """Test model listing."""

    def test_get_models(self, tts_model):
        models = tts_model._get_models()
        model_ids = [m.id for m in models]
        assert "mars-pro" in model_ids
        assert "mars-flash" in model_ids
        assert "mars-instruct" in model_ids


class TestVoiceIdParsing:
    """Test voice ID validation."""

    def test_valid_numeric_string(self):
        assert _parse_voice_id("147320") == 147320

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="CAMB voice ID must be numeric"):
            _parse_voice_id("invalid")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="CAMB voice ID must be numeric"):
            _parse_voice_id("")

    @pytest.mark.asyncio
    async def test_non_numeric_voice_in_agenerate(self, tts_model):
        """Test that non-numeric voice ID raises clear error in agenerate_speech."""
        mock_client, _ = _make_mock_client()
        tts_model._client_instance = mock_client

        with pytest.raises(ValueError, match="CAMB voice ID must be numeric"):
            await tts_model.agenerate_speech(text="Hello", voice="not-a-number")


class TestEmptyAudioValidation:
    """Test empty audio response handling."""

    @pytest.mark.asyncio
    async def test_empty_stream_raises(self, tts_model):
        """Test that empty audio stream raises RuntimeError."""
        mock_client, _ = _make_mock_client(chunks=[])
        tts_model._client_instance = mock_client

        with pytest.raises(RuntimeError, match="No audio data received"):
            await tts_model.agenerate_speech(text="Hello", voice="147320")


class TestValidation:
    """Test input validation."""

    def test_empty_text_raises(self, tts_model):
        with pytest.raises(ValueError, match="Text must be"):
            tts_model.generate_speech(text="", voice="147320")

    def test_empty_voice_raises(self, tts_model):
        with pytest.raises(ValueError, match="Voice must be"):
            tts_model.generate_speech(text="Hello", voice="")
