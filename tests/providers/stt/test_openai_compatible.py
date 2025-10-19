"""Tests for the OpenAI-compatible STT provider."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock
from io import BytesIO

from esperanto.providers.stt.openai_compatible import OpenAICompatibleSpeechToTextModel


@pytest.fixture
def mock_stt_response():
    """Mock STT response data."""
    return {
        "text": "Hello, this is a test transcription from OpenAI-compatible endpoint."
    }


@pytest.fixture
def mock_openai_compatible_models_response():
    """Mock HTTP response for OpenAI-compatible models API."""
    return {
        "object": "list",
        "data": [
            {
                "id": "faster-whisper-large-v3",
                "object": "model",
                "owned_by": "custom"
            },
            {
                "id": "whisper-1",
                "object": "model",
                "owned_by": "custom"
            }
        ]
    }


@pytest.fixture
def mock_httpx_clients(mock_stt_response, mock_openai_compatible_models_response):
    """Mock httpx clients for OpenAI-compatible STT."""
    client = Mock()
    async_client = AsyncMock()

    # Mock HTTP response objects
    def make_response(status_code, json_data=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        return response

    def make_async_response(status_code, json_data=None):
        response = AsyncMock()
        response.status_code = status_code
        if json_data is not None:
            response.json = Mock(return_value=json_data)
        return response

    # Configure responses based on URL
    def mock_post_side_effect(url, **kwargs):
        if url.endswith("/audio/transcriptions"):
            return make_response(200, json_data=mock_stt_response)
        return make_response(404, json_data={"error": "Not found"})

    def mock_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_response(200, json_data=mock_openai_compatible_models_response)
        return make_response(404, json_data={"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if url.endswith("/audio/transcriptions"):
            return make_async_response(200, json_data=mock_stt_response)
        return make_async_response(404, json_data={"error": "Not found"})

    async def mock_async_get_side_effect(url, **kwargs):
        if url.endswith("/models"):
            return make_async_response(200, json_data=mock_openai_compatible_models_response)
        return make_async_response(404, json_data={"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect
    async_client.get.side_effect = mock_async_get_side_effect

    return client, async_client


@pytest.fixture
def stt_model(mock_httpx_clients):
    """Create a STT model instance with mocked HTTP clients."""
    model = OpenAICompatibleSpeechToTextModel(
        model_name="faster-whisper-large-v3",
        api_key="test-key",
        base_url="http://localhost:8000"
    )
    model.client, model.async_client = mock_httpx_clients
    return model


def test_init_with_config():
    """Test model initialization with config."""
    model = OpenAICompatibleSpeechToTextModel(
        model_name="faster-whisper-large-v3",
        api_key="test-key",
        base_url="http://localhost:8000"
    )
    assert model.model_name == "faster-whisper-large-v3"
    assert model.provider == "openai-compatible"
    assert model.base_url == "http://localhost:8000"
    assert model.api_key == "test-key"


def test_init_with_env_vars(monkeypatch):
    """Test model initialization with environment variables."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:9000")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "env-key")

    model = OpenAICompatibleSpeechToTextModel(model_name="test-model")
    assert model.base_url == "http://localhost:9000"
    assert model.api_key == "env-key"


def test_init_missing_base_url(monkeypatch):
    """Test that initialization fails without base URL."""
    # Clear all environment variables
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL_STT", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OpenAI-compatible base URL is required"):
        OpenAICompatibleSpeechToTextModel(model_name="test-model")


def test_base_url_trailing_slash():
    """Test that trailing slash is stripped from base URL."""
    model = OpenAICompatibleSpeechToTextModel(
        model_name="test-model",
        base_url="http://localhost:8000/"
    )
    assert model.base_url == "http://localhost:8000"


def test_timeout_configuration():
    """Test timeout configuration."""
    model = OpenAICompatibleSpeechToTextModel(
        model_name="test-model",
        base_url="http://localhost:8000",
        config={"timeout": 600}
    )
    assert model.timeout == 600


def test_transcribe_with_file_path(stt_model, tmp_path):
    """Test synchronous transcription with file path."""
    # Create a temporary audio file
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio data")

    response = stt_model.transcribe(str(audio_file))

    # Verify HTTP POST was called
    stt_model.client.post.assert_called_once()
    call_args = stt_model.client.post.call_args

    # Check URL
    assert call_args[0][0] == "http://localhost:8000/audio/transcriptions"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"

    # Check multipart form data
    files = call_args[1]["files"]
    assert "file" in files

    # Check form data
    data = call_args[1]["data"]
    assert data["model"] == "faster-whisper-large-v3"

    # Check response
    assert response.text == "Hello, this is a test transcription from OpenAI-compatible endpoint."
    assert response.model == "faster-whisper-large-v3"


def test_transcribe_with_file_object(stt_model):
    """Test synchronous transcription with file object."""
    audio_data = BytesIO(b"fake audio data")
    audio_data.name = "test.mp3"

    response = stt_model.transcribe(audio_data)

    # Verify HTTP POST was called
    stt_model.client.post.assert_called_once()
    call_args = stt_model.client.post.call_args

    # Check URL
    assert call_args[0][0] == "http://localhost:8000/audio/transcriptions"

    # Check multipart form data
    files = call_args[1]["files"]
    assert "file" in files

    # Check response
    assert response.text == "Hello, this is a test transcription from OpenAI-compatible endpoint."
    assert response.model == "faster-whisper-large-v3"


@pytest.mark.asyncio
async def test_atranscribe_with_file_path(stt_model, tmp_path):
    """Test asynchronous transcription with file path."""
    # Create a temporary audio file
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio data")

    response = await stt_model.atranscribe(str(audio_file))

    # Verify async HTTP POST was called
    stt_model.async_client.post.assert_called_once()
    call_args = stt_model.async_client.post.call_args

    # Check URL
    assert call_args[0][0] == "http://localhost:8000/audio/transcriptions"

    # Check headers
    headers = call_args[1]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-key"

    # Check response
    assert response.text == "Hello, this is a test transcription from OpenAI-compatible endpoint."
    assert response.model == "faster-whisper-large-v3"


@pytest.mark.asyncio
async def test_atranscribe_with_file_object(stt_model):
    """Test asynchronous transcription with file object."""
    audio_data = BytesIO(b"fake audio data")
    audio_data.name = "test.wav"

    response = await stt_model.atranscribe(audio_data)

    # Verify async HTTP POST was called
    stt_model.async_client.post.assert_called_once()

    # Check response
    assert response.text == "Hello, this is a test transcription from OpenAI-compatible endpoint."


def test_transcribe_with_language_and_prompt(stt_model, tmp_path):
    """Test transcription with language and prompt parameters."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio data")

    response = stt_model.transcribe(
        str(audio_file),
        language="en",
        prompt="This is a technical discussion"
    )

    # Check that language and prompt were passed in form data
    call_args = stt_model.client.post.call_args
    data = call_args[1]["data"]
    assert data["language"] == "en"
    assert data["prompt"] == "This is a technical discussion"


def test_models(stt_model):
    """Test that the models property works with HTTP."""
    models = stt_model.models

    # Verify HTTP GET was called
    stt_model.client.get.assert_called_with(
        "http://localhost:8000/models",
        headers={"Authorization": "Bearer test-key"}
    )

    # Check that models are returned
    assert len(models) == 2
    assert models[0].id == "faster-whisper-large-v3"
    assert models[1].id == "whisper-1"
    assert models[0].type == "speech_to_text"
    assert models[1].type == "speech_to_text"
    assert models[0].owned_by == "custom"
    assert models[1].owned_by == "custom"


def test_models_fallback():
    """Test that models property falls back gracefully when endpoint doesn't support it."""
    model = OpenAICompatibleSpeechToTextModel(
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
    model = OpenAICompatibleSpeechToTextModel(
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

    audio_data = BytesIO(b"fake audio data")
    with pytest.raises(RuntimeError, match="OpenAI-compatible STT endpoint error"):
        model.transcribe(audio_data)


def test_get_default_model():
    """Test default model name."""
    model = OpenAICompatibleSpeechToTextModel(
        base_url="http://localhost:8000"
    )
    assert model._get_default_model() == "whisper-1"


def test_provider_name(stt_model):
    """Test provider name."""
    assert stt_model.provider == "openai-compatible"


def test_get_model_name(stt_model):
    """Test getting model name."""
    assert stt_model.get_model_name() == "faster-whisper-large-v3"


def test_get_headers(stt_model):
    """Test header generation."""
    headers = stt_model._get_headers()
    assert headers["Authorization"] == "Bearer test-key"


def test_get_api_kwargs(stt_model):
    """Test API kwargs generation."""
    kwargs = stt_model._get_api_kwargs("en", "technical discussion")
    assert kwargs["model"] == "faster-whisper-large-v3"
    assert kwargs["language"] == "en"
    assert kwargs["prompt"] == "technical discussion"


def test_provider_specific_env_var_precedence(monkeypatch):
    """Test that provider-specific env vars take precedence over generic ones."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_STT", "http://stt-specific:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_STT", "stt-specific-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-key")

    model = OpenAICompatibleSpeechToTextModel(model_name="test-model")
    assert model.base_url == "http://stt-specific:1234"
    assert model.api_key == "stt-specific-key"


def test_fallback_to_generic_env_var(monkeypatch):
    """Test fallback to generic env vars when provider-specific ones are not set."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-key")

    model = OpenAICompatibleSpeechToTextModel(model_name="test-model")
    assert model.base_url == "http://generic:5678"
    assert model.api_key == "generic-key"


def test_config_overrides_provider_specific_env_vars(monkeypatch):
    """Test that config parameters override provider-specific env vars."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_STT", "http://stt-env:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_STT", "stt-env-key")

    model = OpenAICompatibleSpeechToTextModel(
        model_name="test-model",
        base_url="http://config:9090",
        api_key="config-key"
    )
    assert model.base_url == "http://config:9090"
    assert model.api_key == "config-key"


def test_direct_params_override_all_env_vars(monkeypatch):
    """Test that direct parameters override all environment variables."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_STT", "http://stt-env:1234")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://generic-env:5678")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY_STT", "stt-env-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-env-key")

    model = OpenAICompatibleSpeechToTextModel(
        model_name="test-model",
        base_url="http://direct:3000",
        api_key="direct-key"
    )
    assert model.base_url == "http://direct:3000"
    assert model.api_key == "direct-key"


def test_error_message_mentions_both_env_vars(monkeypatch):
    """Test that error message mentions both provider-specific and generic env vars."""
    # Clear all environment variables
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL_STT", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)

    with pytest.raises(ValueError) as exc_info:
        OpenAICompatibleSpeechToTextModel(model_name="test-model", api_key="test-key")

    error_message = str(exc_info.value)
    assert "OPENAI_COMPATIBLE_BASE_URL_STT" in error_message
    assert "OPENAI_COMPATIBLE_BASE_URL" in error_message