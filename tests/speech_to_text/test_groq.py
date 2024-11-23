"""Tests for Groq speech-to-text model implementation."""

from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest
from groq import AsyncGroq

from esperanto.providers.speech_to_text.groq import GroqSpeechToTextModel


@pytest.fixture
def mock_response():
    """Create a mock Groq response."""
    mock = Mock()
    mock.text = "Hello, world!"
    return mock


@pytest.fixture
def mock_client(mock_response):
    """Create a mock Groq client."""
    mock_transcriptions = Mock()
    mock_transcriptions.create = AsyncMock(return_value=mock_response)

    mock_audio = Mock()
    mock_audio.transcriptions = mock_transcriptions

    client = AsyncMock(spec=AsyncGroq)
    client.audio = mock_audio
    return client


@pytest.fixture
def model():
    """Create a test model instance."""
    return GroqSpeechToTextModel(model_name="whisper-1")


class TestGroqSpeechToTextModel:
    """Test suite for Groq speech-to-text model."""

    async def test_initialization(self):
        """Test model initialization with valid config."""
        model = GroqSpeechToTextModel(model_name="whisper-1")
        assert model.model_name == "whisper-1"
        assert model.config == {}
        assert model._client is None
        assert model.language is None
        assert model.prompt is None
        assert model.response_format == "text"
        assert model.temperature == 0

    async def test_initialization_with_config(self):
        """Test model initialization with config values."""
        config = {
            "language": "en",
            "prompt": "Transcribe this",
            "response_format": "json",
            "temperature": 0.5,
        }
        model = GroqSpeechToTextModel(model_name="whisper-1", config=config)
        assert model.model_name == "whisper-1"
        assert model.config == config
        assert model.language == "en"
        assert model.prompt == "Transcribe this"
        assert model.response_format == "json"
        assert model.temperature == 0.5

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(
            ValueError,
            match="model_name must be specified for Groq speech-to-text model",
        ):
            GroqSpeechToTextModel(model_name="")

    async def test_provider_name(self, model):
        """Test provider name is correct."""
        assert model.provider == "groq"

    async def test_validate_config_success(self, model):
        """Test config validation with valid config."""
        model.validate_config()  # Should not raise any exceptions

    async def test_validate_config_failure_no_model_name(self):
        """Test config validation with no model name."""
        model = GroqSpeechToTextModel(model_name="whisper-1")
        model.model_name = ""
        with pytest.raises(
            ValueError,
            match="model_name must be specified for Groq speech-to-text model",
        ):
            model.validate_config()

    async def test_validate_config_failure_invalid_response_format(self):
        """Test config validation with invalid response format."""
        model = GroqSpeechToTextModel(model_name="whisper-1")
        model.response_format = "invalid"
        with pytest.raises(
            ValueError,
            match="response_format must be one of: json, text",
        ):
            model.validate_config()

    async def test_client_lazy_initialization(self, model):
        """Test lazy initialization of Groq client."""
        mock_client = AsyncMock(spec=AsyncGroq)

        with patch(
            "esperanto.providers.speech_to_text.groq.AsyncGroq",
            return_value=mock_client,
        ) as mock_groq:
            # First access should initialize the client
            assert model.client is not None
            mock_groq.assert_called_once()

            # Second access should use cached client
            assert model.client is model._client
            mock_groq.assert_called_once()

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

    async def test_transcribe(self, model, mock_client, mock_response):
        """Test audio transcription."""
        model.client = mock_client
        audio_file = "test.mp3"

        with patch("builtins.open", mock_open()) as mock_file:
            result = await model.transcribe(audio_file)

            # Verify results
            assert result.text == "Hello, world!"

            # Verify client calls
            mock_file.assert_called_once_with(audio_file, "rb")
            mock_client.audio.transcriptions.create.assert_called_once_with(
                model="whisper-1",
                file=mock_file(),
                language=None,
                prompt=None,
                response_format="text",
                temperature=0,
            )
