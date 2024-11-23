"""Tests for Gemini text-to-speech model implementation."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.cloud.texttospeech_v1 import TextToSpeechAsyncClient
from google.cloud.texttospeech_v1.types import (
    AudioConfig,
    SynthesisInput,
    VoiceSelectionParams,
)

from esperanto.providers.text_to_speech.gemini import GeminiTextToSpeechModel


@pytest.fixture
def mock_response():
    """Create a mock Gemini response."""
    response = Mock()
    response.audio_content = b"audio_content"
    return response


@pytest.fixture
def mock_client(mock_response):
    """Create a mock Gemini client."""
    client = AsyncMock(spec=TextToSpeechAsyncClient)
    client.synthesize_speech = AsyncMock(return_value=mock_response)
    client.synthesize_speech.return_value = mock_response
    return client


@pytest.fixture
def model():
    """Create a test model instance."""
    return GeminiTextToSpeechModel()


class TestGeminiTextToSpeechModel:
    """Test suite for Gemini text-to-speech model."""

    async def test_initialization(self):
        """Test model initialization with default config."""
        model = GeminiTextToSpeechModel()
        assert model.config == {}
        assert model._client is None
        assert model.language_code == "en-US"
        assert model.voice_name == "en-US-Standard-A"
        assert model.speaking_rate == 1.0
        assert model.pitch == 0.0
        assert model.volume_gain_db == 0.0
        assert model.audio_encoding == "MP3"
        assert model.sample_rate_hertz == 16000

    async def test_initialization_with_config(self):
        """Test model initialization with custom config."""
        config = {
            "language_code": "es-ES",
            "voice_name": "es-ES-Standard-A",
            "speaking_rate": 1.2,
            "pitch": 2.0,
            "volume_gain_db": 1.0,
            "audio_encoding": "LINEAR16",
            "sample_rate_hertz": 16000,
        }
        model = GeminiTextToSpeechModel(config=config)
        assert model.config == config
        assert model.language_code == "es-ES"
        assert model.voice_name == "es-ES-Standard-A"
        assert model.speaking_rate == 1.2
        assert model.pitch == 2.0
        assert model.volume_gain_db == 1.0
        assert model.audio_encoding == "LINEAR16"
        assert model.sample_rate_hertz == 16000

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "gemini"

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exceptions

    async def test_validate_config_failure_invalid_speaking_rate(self):
        """Test config validation with invalid speaking rate."""
        model = GeminiTextToSpeechModel()
        model.speaking_rate = 0.0  # Must be between 0.25 and 4.0
        with pytest.raises(ValueError, match="speaking_rate must be between 0.25 and 4.0"):
            model.validate_config()

    async def test_validate_config_failure_invalid_pitch(self):
        """Test config validation with invalid pitch."""
        model = GeminiTextToSpeechModel()
        model.pitch = 21.0  # Must be between -20 and 20
        with pytest.raises(ValueError, match="pitch must be between -20.0 and 20.0"):
            model.validate_config()

    async def test_validate_config_failure_invalid_volume_gain(self):
        """Test config validation with invalid volume gain."""
        model = GeminiTextToSpeechModel()
        model.volume_gain_db = -97.0  # Must be between -96 and 16
        with pytest.raises(ValueError, match="volume_gain_db must be between -96.0 and 16.0"):
            model.validate_config()

    async def test_validate_config_failure_invalid_sample_rate(self):
        """Test config validation with invalid sample rate."""
        model = GeminiTextToSpeechModel()
        model.sample_rate_hertz = -1  # Must be between 8000 and 48000
        with pytest.raises(ValueError, match="sample_rate_hertz must be between 8000 and 48000"):
            model.validate_config()

    async def test_client_lazy_initialization(self, model):
        """Test lazy initialization of Gemini client."""
        mock_client = AsyncMock(spec=TextToSpeechAsyncClient)

        with patch("google.cloud.texttospeech_v1.TextToSpeechAsyncClient", return_value=mock_client) as mock_tts_client:
            # First access should initialize the client
            assert model.client is not None
            mock_tts_client.assert_called_once()

            # Second access should use cached client
            assert model.client is model._client
            mock_tts_client.assert_called_once()

    async def test_client_setter(self, model, mock_client):
        """Test client setter."""
        model.client = mock_client
        assert model._client == mock_client

    async def test_client_deleter(self, model, mock_client):
        """Test client deleter."""
        model.client = mock_client
        assert model._client == mock_client

        del model.client
        assert model._client is None

    async def test_synthesize(self, model, mock_client, mock_response, tmp_path):
        """Test text-to-speech synthesis."""
        model.client = mock_client
        text = "Hello, world!"
        output_file = str(tmp_path / "output.mp3")

        result = await model.synthesize(text, output_file)

        # Verify results
        assert result == output_file
        assert os.path.exists(output_file)

        # Verify client calls
        mock_client.synthesize_speech.assert_called_once()
        call_kwargs = mock_client.synthesize_speech.call_args.kwargs

        # Verify input text
        assert isinstance(call_kwargs["input"], SynthesisInput)
        assert call_kwargs["input"].text == text

        # Verify voice selection
        assert isinstance(call_kwargs["voice"], VoiceSelectionParams)
        assert call_kwargs["voice"].language_code == "en-US"
        assert call_kwargs["voice"].name == "en-US-Standard-A"

        # Verify audio config
        assert isinstance(call_kwargs["audio_config"], AudioConfig)
        assert call_kwargs["audio_config"].speaking_rate == 1.0
        assert call_kwargs["audio_config"].pitch == 0.0
        assert call_kwargs["audio_config"].volume_gain_db == 0.0
        assert call_kwargs["audio_config"].sample_rate_hertz == 16000

        # Verify file was written
        with open(output_file, "rb") as f:
            content = f.read()
            assert content == b"audio_content"

    async def test_synthesize_with_config(self, model, mock_client, mock_response, tmp_path):
        """Test text-to-speech synthesis with custom config."""
        config = {
            "language_code": "es-ES",
            "voice_name": "es-ES-Standard-A",
            "speaking_rate": 1.2,
            "pitch": 2.0,
            "volume_gain_db": 1.0,
            "audio_encoding": "LINEAR16",
            "sample_rate_hertz": 16000,
        }
        model = GeminiTextToSpeechModel(config=config)
        model.client = mock_client
        text = "¡Hola, mundo!"
        output_file = str(tmp_path / "output.wav")

        result = await model.synthesize(text, output_file)

        # Verify results
        assert result == output_file
        assert os.path.exists(output_file)

        # Verify client calls
        mock_client.synthesize_speech.assert_called_once()
        call_kwargs = mock_client.synthesize_speech.call_args.kwargs

        # Verify input text
        assert isinstance(call_kwargs["input"], SynthesisInput)
        assert call_kwargs["input"].text == text

        # Verify voice selection
        assert isinstance(call_kwargs["voice"], VoiceSelectionParams)
        assert call_kwargs["voice"].language_code == "es-ES"
        assert call_kwargs["voice"].name == "es-ES-Standard-A"

        # Verify audio config
        assert isinstance(call_kwargs["audio_config"], AudioConfig)
        assert call_kwargs["audio_config"].speaking_rate == 1.2
        assert call_kwargs["audio_config"].pitch == 2.0
        assert call_kwargs["audio_config"].volume_gain_db == 1.0
        assert call_kwargs["audio_config"].sample_rate_hertz == 16000

        # Verify file was written
        with open(output_file, "rb") as f:
            content = f.read()
            assert content == b"audio_content"

    async def test_synthesize_with_kwargs(self, model, mock_client, mock_response, tmp_path):
        """Test text-to-speech synthesis with additional kwargs."""
        model.client = mock_client
        text = "Hello, world!"
        output_file = str(tmp_path / "output.mp3")

        result = await model.synthesize(
            text,
            output_file,
            speaking_rate=1.5,
            pitch=1.0,
            volume_gain_db=2.0,
        )

        # Verify results
        assert result == output_file
        assert os.path.exists(output_file)

        # Verify client calls
        mock_client.synthesize_speech.assert_called_once()
        call_kwargs = mock_client.synthesize_speech.call_args.kwargs

        # Verify input text
        assert isinstance(call_kwargs["input"], SynthesisInput)
        assert call_kwargs["input"].text == text

        # Verify voice selection
        assert isinstance(call_kwargs["voice"], VoiceSelectionParams)
        assert call_kwargs["voice"].language_code == "en-US"
        assert call_kwargs["voice"].name == "en-US-Standard-A"

        # Verify audio config
        assert isinstance(call_kwargs["audio_config"], AudioConfig)
        assert call_kwargs["audio_config"].speaking_rate == 1.5
        assert call_kwargs["audio_config"].pitch == 1.0
        assert call_kwargs["audio_config"].volume_gain_db == 2.0
        assert call_kwargs["audio_config"].sample_rate_hertz == 16000

        # Verify file was written
        with open(output_file, "rb") as f:
            content = f.read()
            assert content == b"audio_content"
