"""Tests for ElevenLabs text-to-speech model implementation."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from elevenlabs import VoiceSettings
from elevenlabs.client import AsyncElevenLabs

from esperanto.providers.text_to_speech.elevenlabs import ElevenLabsTextToSpeechModel


@pytest.fixture
def mock_response():
    """Create a mock ElevenLabs response."""
    response = Mock()
    response.content = b"audio_content"
    return response


@pytest.fixture
def mock_voice():
    """Create a mock ElevenLabs voice."""
    voice = Mock()
    voice.voice_id = "test_voice_id"
    voice.settings = VoiceSettings(stability=0.5, similarity_boost=0.5, style=0.5)
    return voice


@pytest.fixture
def mock_client(mock_response, mock_voice):
    """Create a mock ElevenLabs client."""
    with patch(
        "esperanto.providers.text_to_speech.elevenlabs.AsyncElevenLabs"
    ) as mock_client_class:
        mock_client_instance = AsyncMock(spec=AsyncElevenLabs)

        # Create a proper async generator function
        async def mock_generator():
            yield b"chunk1"
            yield b"chunk2"

        mock_client_instance.generate = AsyncMock(return_value=mock_generator())

        mock_client_class.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def model():
    """Create a test model instance."""
    return ElevenLabsTextToSpeechModel(
        model_name="test_model",
        config={
            "api_key": "test_key",
            "voice": "test_voice"
        }
    )


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    if "ELEVENLABS_API_KEY" in os.environ:
        del os.environ["ELEVENLABS_API_KEY"]


class TestElevenLabsTextToSpeechModel:
    """Test suite for ElevenLabs text-to-speech model."""

    def test_initialization(self):
        """Test model initialization with default config."""
        model = ElevenLabsTextToSpeechModel(
            model_name="test_model",
            config={
                "api_key": "test_key",
                "voice": "test_voice"
            }
        )
        assert model.config.get("api_key") == "test_key"
        assert model.config.get("voice") == "test_voice"
        assert model._client is None

    def test_initialization_with_config(self):
        """Test model initialization with custom config."""
        config = {
            "api_key": "test_key",
            "voice": "custom_voice",
            "model_id": "eleven_monolingual_v1"
        }
        model = ElevenLabsTextToSpeechModel(
            model_name="test_model",
            config=config
        )
        assert model.config.get("voice") == "custom_voice"
        assert model.config.get("model_id") == "eleven_monolingual_v1"

    def test_provider_name(self):
        """Test provider name."""
        model = ElevenLabsTextToSpeechModel(
            model_name="test_model",
            config={"api_key": "test_key", "voice": "test_voice"}
        )
        assert model.provider == "elevenlabs"

    def test_client_setter(self, mock_client):
        """Test client setter."""
        model = ElevenLabsTextToSpeechModel(
            model_name="test_model",
            config={"api_key": "test_key", "voice": "test_voice"}
        )
        model.client = mock_client
        assert model._client == mock_client

    def test_client_deleter(self, mock_client):
        """Test client deleter."""
        model = ElevenLabsTextToSpeechModel(
            model_name="test_model",
            config={"api_key": "test_key", "voice": "test_voice"}
        )
        model.client = mock_client
        assert model._client is not None
        del model.client
        assert model._client is None

    @pytest.mark.asyncio
    async def test_synthesize(self, model, mock_client, tmp_path):
        """Test synthesize method."""
        text = "Hello, World!"
        output_file = str(tmp_path / "output.mp3")

        # Set mock client and model attributes
        model.client = mock_client
        model._model = "test_model"  # Set the internal _model attribute
        model._api_key = "test_key"  # Set API key
        model.config["voice"] = "test_voice"  # Set voice in config

        # Synthesize text
        result = await model.synthesize(text, output_file)

        # Verify results
        assert result == output_file
        assert os.path.exists(output_file)
        with open(output_file, "rb") as f:
            content = f.read()
            assert b"chunk1chunk2" in content

        # Verify client call
        mock_client.generate.assert_called_once()
