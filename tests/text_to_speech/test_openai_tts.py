"""Tests for OpenAI text-to-speech model."""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from esperanto.providers.text_to_speech.openai import OpenAITextToSpeechModel


@pytest.fixture
def model_name():
    """Model name fixture."""
    return "tts-1"


@pytest.fixture
def config():
    """Config fixture."""
    return {
        "api_key": "test-api-key",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
    }


@pytest.fixture
def model(model_name, config):
    """Model fixture."""
    return OpenAITextToSpeechModel(model_name=model_name, config=config)


def test_provider(model):
    """Test provider name."""
    assert model.provider == "openai"


def test_validate_config_missing_model_name():
    """Test validate config with missing model name."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        model = OpenAITextToSpeechModel(config={})
        with pytest.raises(ValueError, match="model_name must be specified"):
            model.validate_config()


def test_validate_config_missing_api_key():
    """Test validate config with missing API key."""
    with patch.dict(os.environ, {}, clear=True):
        model = OpenAITextToSpeechModel(model_name="tts-1", config={})
        with pytest.raises(ValueError, match="api_key must be specified"):
            model.validate_config()


def test_validate_config_invalid_voice():
    """Test validate config with invalid voice."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        config = {"voice": "invalid"}
        model = OpenAITextToSpeechModel(model_name="tts-1", config=config)
        with pytest.raises(ValueError, match="Invalid voice"):
            model.validate_config()


def test_validate_config_invalid_response_format():
    """Test validate config with invalid response format."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        config = {
            "voice": "alloy",  # Valid voice
            "response_format": "invalid",
        }
        model = OpenAITextToSpeechModel(model_name="tts-1", config=config)
        with pytest.raises(ValueError, match="Invalid response_format"):
            model.validate_config()


def test_validate_config_invalid_speed():
    """Test validate config with invalid speed."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        config = {
            "voice": "alloy",  # Valid voice
            "response_format": "mp3",  # Valid format
            "speed": 5.0,
        }
        model = OpenAITextToSpeechModel(model_name="tts-1", config=config)
        with pytest.raises(ValueError, match="Speed must be between"):
            model.validate_config()


def test_validate_config_valid():
    """Test validate config with valid config."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        config = {
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.0,
        }
        model = OpenAITextToSpeechModel(model_name="tts-1", config=config)
        model.validate_config()  # Should not raise any exceptions


def test_initialization_with_model_name():
    """Test initialization with model_name parameter."""
    model = OpenAITextToSpeechModel(model_name="tts-1-hd")
    assert model._model == "tts-1-hd"


def test_initialization_with_model_in_config():
    """Test initialization with model in config."""
    model = OpenAITextToSpeechModel(config={"model": "tts-1-hd"})
    assert model._model == "tts-1-hd"


def test_initialization_model_name_overrides_config():
    """Test that model_name parameter overrides config model."""
    model = OpenAITextToSpeechModel(model_name="tts-1-hd", config={"model": "tts-1"})
    assert model._model == "tts-1-hd"


@pytest.mark.asyncio
async def test_synthesize(model, tmp_path):
    """Test synthesize method."""
    text = "Hello, World!"
    output_file = str(tmp_path / "output.mp3")

    # Mock response
    mock_response = MagicMock()
    mock_response.content = b"audio content"
    mock_response.iter_bytes.return_value = [b"audio content"]

    # Mock client with nested attributes
    mock_client = MagicMock()
    mock_client.audio = MagicMock()
    mock_client.audio.speech = MagicMock()
    mock_client.audio.speech.create.return_value = mock_response

    # Set API key and mock client
    model._api_key = SecretStr("test-key")
    model._client = mock_client

    # Synthesize text
    result = await model.synthesize(text, output_file)

    # Verify results
    assert result == output_file
    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        content = f.read()
        assert content == b"audio content"

    # Verify client call
    mock_client.audio.speech.create.assert_called_once_with(
        model=model.model,
        voice=model.voice,
        input=text,
        response_format=model.response_format,
        speed=model.speed,
    )


@pytest.mark.asyncio
async def test_synthesize_with_env_api_key(tmp_path):
    """Test synthesize with API key from environment variable."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        model = OpenAITextToSpeechModel(model_name="tts-1", config={"voice": "alloy"})
        text = "Hello, World!"
        output_file = str(tmp_path / "output.mp3")

        # Mock response
        mock_response = MagicMock()
        mock_response.content = b"audio content"
        mock_response.iter_bytes.return_value = [b"audio content"]

        # Mock client with nested attributes
        mock_client = MagicMock()
        mock_client.audio = MagicMock()
        mock_client.audio.speech = MagicMock()
        mock_client.audio.speech.create.return_value = mock_response

        # Set mock client
        model._client = mock_client

        # Synthesize text
        result = await model.synthesize(text, output_file)

        # Verify results
        assert result == output_file
        assert os.path.exists(output_file)
        with open(output_file, "rb") as f:
            content = f.read()
            assert content == b"audio content"

        # Verify client call
        mock_client.audio.speech.create.assert_called_once_with(
            model=model.model,
            voice=model.voice,
            input=text,
            response_format=model.response_format,
            speed=model.speed,
        )
