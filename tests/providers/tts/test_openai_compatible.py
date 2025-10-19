"""Tests for the OpenAI-compatible TTS provider."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

from esperanto.providers.tts.openai_compatible import OpenAICompatibleTextToSpeechModel


@pytest.fixture
def mock_tts_audio_response():
    """Mock binary audio response data."""
    return b"mock audio data for openai compatible testing"


@pytest.fixture
def mock_openai_compatible_models_response():
    """Mock HTTP response for OpenAI-compatible models API."""
    return {
        "object": "list",
        "data": [
            {
                "id": "piper-tts",
                "object": "model",
                "owned_by": "custom"
            },
            {
                "id": "parler-tts",
                "object": "model",
                "owned_by": "custom"
            }
        ]
    }


@pytest.fixture
def mock_openai_compatible_voices_response():
    """Mock HTTP response for OpenAI-compatible voices API."""
    return {
        "voices": [
            {
                "id": "en/en_US/amy/medium/en_US-amy-medium.onnx",
                "name": "Amy",
                "gender": "FEMALE",
                "language_code": "en-US",
                "description": "Medium quality Amy voice"
            },
            {
                "id": "en/en_US/joe/medium/en_US-joe-medium.onnx",
                "name": "Joe",
                "gender": "MALE",
                "language_code": "en-US",
                "description": "Medium quality Joe voice"
            }
        ]
    }


@pytest.fixture
def mock_httpx_clients(mock_tts_audio_response, mock_openai_compatible_models_response, mock_openai_compatible_voices_response):
    """Mock httpx clients for OpenAI-compatible TTS."""
    client = Mock()
    async_client = AsyncMock()

    # Mock HTTP response objects
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

    # Configure responses based on URL
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_response(200, content=mock_tts_audio_response)
        return make_response(404, json_data={"error": "Not found"})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_response(200, json_data=mock_openai_compatible_models_response)
        elif url.endswith("/audio/voices"):
            return make_response(200, json_data=mock_openai_compatible_voices_response)
        return make_response(404, json_data={"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/audio/speech"):
            return make_async_response(200, content=mock_tts_audio_response)
        return make_async_response(404, json_data={"error": "Not found"})

    async def mock_async_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_async_response(200, json_data=mock_openai_compatible_models_response)
        elif url.endswith("/audio/voices"):
            return make_async_response(200, json_data=mock_openai_compatible_voices_response)
        return make_async_response(404, json_data={"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    async_client.get.side_effect = mock_async_get_side_effect

    return client, async_client


@pytest.fixture
def tts_model(mock_httpx_clients):
    """Create a TTS model instance with mocked HTTP clients."""
    model = OpenAICompatibleTextToSpeechModel(
        model_name="piper-tts",
        api_key="test-key",
        base_url="http://localhost:8000"
    )
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init_with_config():
    """Test model initialization with config."""
    model = OpenAICompatibleTextToSpeechModel(
        model_name="piper-tts",
        api_key="test-key",
        base_url="http://localhost:8000"
    )
    assert model.model_name == "piper-tts"
    assert model.provider == "openai-compatible"
    assert model.base_url == "http://localhost:8000"
    assert model.api_key == "test-key"


def test_init_with_env_vars(monkeypatch):
    """Test model initialization with environment variables."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:9000")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "env-key")

    model = OpenAICompatibleTextToSpeechModel(model_name="test-model")
    assert model.base_url == "http://localhost:9000"
    assert model.api_key == "env-key"


def test_init_missing_base_url(monkeypatch):
    """Test that initialization fails without base URL."""
    # Clear all environment variables
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL_TTS", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OpenAI-compatible base URL is required"):
        OpenAICompatibleTextToSpeechModel(model_name="test-model")


def test_base_url_trailing_slash():
    """Test that trailing slash is stripped from base URL."""
    model = OpenAICompatibleTextToSpeechModel(
        model_name="test-model",
        base_url="http://localhost:8000/"
    )
    assert model.base_url == "http://localhost:8000"


def test_generate_speech(tts_model):
    """Test synchronous speech generation."""
    response = tts_model.generate_speech(
        text="Hello world",
        voice="en/en_US/amy/medium/en_US-amy-medium.onnx"
    )

    # Verify HTTP POST was called
    tts_model.client.post.assert_called_once()
    call_args = tts_model.client.post.call_args

    # Check URL
    assert call_args[0][0] == "http://localhost:8000/audio/speech"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"

    # Check JSON payload (should use OpenAI standard "input" not "text")
    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "piper-tts"
    assert json_payload["voice"] == "en/en_US/amy/medium/en_US-amy-medium.onnx"
    assert json_payload["input"] == "Hello world"

    # Check response
    assert response.audio_data == b"mock audio data for openai compatible testing"
    assert response.content_type == "audio/mp3"
    assert response.model == "piper-tts"
    assert response.voice == "en/en_US/amy/medium/en_US-amy-medium.onnx"
    assert response.provider == "openai-compatible"


@pytest.mark.asyncio
async def test_agenerate_speech(tts_model):
    """Test asynchronous speech generation."""
    response = await tts_model.agenerate_speech(
        text="Hello async world",
        voice="en/en_US/joe/medium/en_US-joe-medium.onnx"
    )

    # Verify async HTTP POST was called
    tts_model.async_client.post.assert_called_once()
    call_args = tts_model.async_client.post.call_args

    # Check URL
    assert call_args[0][0] == "http://localhost:8000/audio/speech"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"

    # Check JSON payload
    json_payload = call_args[1]["json"]
    assert json_payload["model"] == "piper-tts"
    assert json_payload["voice"] == "en/en_US/joe/medium/en_US-joe-medium.onnx"
    assert json_payload["input"] == "Hello async world"

    # Check response
    assert response.audio_data == b"mock audio data for openai compatible testing"
    assert response.content_type == "audio/mp3"
    assert response.model == "piper-tts"
    assert response.voice == "en/en_US/joe/medium/en_US-joe-medium.onnx"
    assert response.provider == "openai-compatible"


def test_available_voices(tts_model):
    """Test getting available voices."""
    voices = tts_model.available_voices

    assert len(voices) == 2

    # Test Amy voice
    amy_voice = voices["en/en_US/amy/medium/en_US-amy-medium.onnx"]
    assert amy_voice.name == "Amy"
    assert amy_voice.id == "en/en_US/amy/medium/en_US-amy-medium.onnx"
    assert amy_voice.gender == "FEMALE"
    assert amy_voice.language_code == "en-US"
    assert amy_voice.description == "Medium quality Amy voice"

    # Test Joe voice
    joe_voice = voices["en/en_US/joe/medium/en_US-joe-medium.onnx"]
    assert joe_voice.name == "Joe"
    assert joe_voice.id == "en/en_US/joe/medium/en_US-joe-medium.onnx"
    assert joe_voice.gender == "MALE"
    assert joe_voice.language_code == "en-US"
    assert joe_voice.description == "Medium quality Joe voice"


def test_available_voices_fallback():
    """Test that available_voices falls back gracefully when endpoint doesn't support it."""
    model = OpenAICompatibleTextToSpeechModel(
        api_key="test-key",
        model_name="test-model",
        base_url="http://localhost:8000"
    )

    # Mock client that returns 404 for voices endpoint
    client = Mock()
    response = Mock()
    response.status_code = 404
    response.json.return_value = {"error": "Not found"}
    client.get.return_value = response
    model.client = client

    voices = model.available_voices

    # Should return default voice
    assert len(voices) == 1
    assert "default" in voices
    assert voices["default"].name == "default"
    assert voices["default"].id == "default"


def test_models(tts_model):
    """Test that the models property works with HTTP."""
    models = tts_model.models

    # Verify HTTP GET was called
    tts_model.client.get.assert_called_with(
        "http://localhost:8000/models",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }
    )

    # Check that models are returned
    assert len(models) == 2
    assert models[0].id == "piper-tts"
    assert models[1].id == "parler-tts"
    assert models[0].type == "text_to_speech"
    assert models[1].type == "text_to_speech"
    assert models[0].owned_by == "custom"
    assert models[1].owned_by == "custom"


def test_models_fallback():
    """Test that models property falls back gracefully when endpoint doesn't support it."""
    model = OpenAICompatibleTextToSpeechModel(
        api_key="test-key",
        model_name="test-model",
        base_url="http://localhost:8000"
    )

    # Mock client that throws exception for models endpoint
    client = Mock()
    client.get.side_effect = Exception("Connection error")
    model.client = client

    models = model.models

    # Should return empty list
    assert models == []


def test_error_handling():
    """Test error handling for HTTP errors."""
    model = OpenAICompatibleTextToSpeechModel(
        api_key="test-key",
        model_name="test-model",
        base_url="http://localhost:8000"
    )

    # Mock client that returns error response
    client = Mock()
    response = Mock()
    response.status_code = 500
    response.text = "Internal server error"
    response.json.side_effect = Exception("Not JSON")
    client.post.return_value = response
    model.client = client

    with pytest.raises(RuntimeError, match="Failed to generate speech: OpenAI-compatible TTS endpoint error"):
        model.generate_speech(text="test", voice="default")


def test_generate_speech_with_default_voice(tts_model):
    """Test speech generation with default voice."""
    response = tts_model.generate_speech(text="Hello world")

    # Check that default voice was used
    call_args = tts_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["voice"] == "default"


def test_generate_speech_with_additional_kwargs(tts_model):
    """Test speech generation with additional parameters."""
    response = tts_model.generate_speech(
        text="Hello world",
        voice="test-voice",
        speed=1.2,
        format="wav"
    )

    # Check that additional kwargs are passed through
    call_args = tts_model.client.post.call_args
    json_payload = call_args[1]["json"]
    assert json_payload["speed"] == 1.2
    assert json_payload["format"] == "wav"


def test_provider_specific_env_var_precedence(monkeypatch):
    """Test that provider-specific env vars take precedence over generic ones."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_TTS", "http://tts-specific:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_TTS", "tts-specific-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-key")

    model = OpenAICompatibleTextToSpeechModel(model_name="test-model")
    assert model.base_url == "http://tts-specific:1234"
    assert model.api_key == "tts-specific-key"


def test_fallback_to_generic_env_var(monkeypatch):
    """Test fallback to generic env vars when provider-specific ones are not set."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-key")

    model = OpenAICompatibleTextToSpeechModel(model_name="test-model")
    assert model.base_url == "http://generic:5678"
    assert model.api_key == "generic-key"


def test_config_overrides_provider_specific_env_vars(monkeypatch):
    """Test that config parameters override provider-specific env vars."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_TTS", "http://tts-env:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_TTS", "tts-env-key")

    model = OpenAICompatibleTextToSpeechModel(
        model_name="test-model",
        base_url="http://config:9090",
        api_key="config-key"
    )
    assert model.base_url == "http://config:9090"
    assert model.api_key == "config-key"


def test_direct_params_override_all_env_vars(monkeypatch):
    """Test that direct parameters override all environment variables."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_TTS", "http://tts-env:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic-env:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_TTS", "tts-env-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-env-key")

    model = OpenAICompatibleTextToSpeechModel(
        model_name="test-model",
        base_url="http://direct:3000",
        api_key="direct-key"
    )
    assert model.base_url == "http://direct:3000"
    assert model.api_key == "direct-key"


def test_error_message_mentions_both_env_vars(monkeypatch):
    """Test that error message mentions both provider-specific and generic env vars."""
    # Clear all environment variables
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL_TTS", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)

    with pytest.raises(ValueError) as exc_info:
        OpenAICompatibleTextToSpeechModel(model_name="test-model", api_key="test-key")

    error_message = str(exc_info.value)
    assert "OPENAI_COMPATIBLE_BASE_URL_TTS" in error_message
    assert "OPENAI_COMPATIBLE_BASE_URL" in error_message