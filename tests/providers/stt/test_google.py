"""Tests for Google speech-to-text provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import TranscriptionResponse
from esperanto.factory import AIFactory
from esperanto.providers.stt.google import GoogleSpeechToTextModel


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"mock audio content")
    return str(audio_file)


@pytest.fixture
def mock_gemini_transcription_response():
    """Mock HTTP response for Gemini transcription API."""
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "This is a test transcription from Gemini"
                }]
            }
        }]
    }


@pytest.fixture
def mock_gemini_error_response():
    """Mock HTTP error response."""
    return {
        "error": {
            "code": 400,
            "message": "Invalid request",
            "status": "INVALID_ARGUMENT"
        }
    }


@pytest.fixture
def mock_httpx_clients(mock_gemini_transcription_response):
    """Mock httpx clients for Google STT."""
    client = Mock()
    async_client = AsyncMock()

    # Mock HTTP response objects
    def make_response(status_code, data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data
        return response

    def make_async_response(status_code, data):
        response = AsyncMock()
        response.status_code = status_code
        response.json = Mock(return_value=data)
        return response

    # Configure responses
    def mock_post_side_effect(url, **kwargs):
        if "generateContent" in url:
            return make_response(200, mock_gemini_transcription_response)
        return make_response(404, {"error": "Not found"})

    async def mock_async_post_side_effect(url, **kwargs):
        if "generateContent" in url:
            return make_async_response(200, mock_gemini_transcription_response)
        return make_async_response(404, {"error": "Not found"})

    # Mock synchronous HTTP client
    client.post.side_effect = mock_post_side_effect

    # Mock async HTTP client
    async_client.post.side_effect = mock_async_post_side_effect

    return client, async_client


class TestGoogleSpeechToTextModel:
    """Tests for GoogleSpeechToTextModel."""

    def test_init_with_google_api_key(self):
        """Test initialization with GOOGLE_API_KEY."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-123"}):
            model = GoogleSpeechToTextModel()
            assert model.api_key == "test-key-123"
            assert model.provider == "google"

    def test_init_with_gemini_api_key(self):
        """Test initialization fallback to GEMINI_API_KEY."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key-456"}, clear=True):
            model = GoogleSpeechToTextModel()
            assert model.api_key == "gemini-key-456"

    def test_init_without_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key not found"):
                GoogleSpeechToTextModel()

    def test_get_default_model(self):
        """Test default model name."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            assert model._get_default_model() == "gemini-2.5-flash"

    def test_provider_property(self):
        """Test provider property returns 'google'."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            assert model.provider == "google"

    def test_get_models(self):
        """Test get_models returns hardcoded list."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            models = model._get_models()
            assert len(models) >= 2
            assert any(m.id == "gemini-2.5-flash" for m in models)
            assert any(m.id == "gemini-2.0-flash" for m in models)
            assert all(m.owned_by == "Google" for m in models)

    def test_get_mime_type_mp3(self):
        """Test MIME type detection for MP3."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            assert model._get_mime_type("test.mp3") == "audio/mp3"

    def test_get_mime_type_wav(self):
        """Test MIME type detection for WAV."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            assert model._get_mime_type("test.wav") == "audio/wav"

    def test_get_mime_type_all_formats(self):
        """Test MIME type detection for all supported formats."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            formats = {
                "test.mp3": "audio/mp3",
                "test.wav": "audio/wav",
                "test.aiff": "audio/aiff",
                "test.aac": "audio/aac",
                "test.ogg": "audio/ogg",
                "test.flac": "audio/flac",
            }
            for file, expected_mime in formats.items():
                assert model._get_mime_type(file) == expected_mime

    def test_get_mime_type_unsupported(self):
        """Test MIME type detection raises error for unsupported format."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            with pytest.raises(ValueError, match="Unsupported audio format"):
                model._get_mime_type("test.txt")

    def test_build_prompt_basic(self):
        """Test basic prompt construction."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            prompt = model._build_prompt()
            assert "Generate a transcript" in prompt
            assert "audio file" in prompt

    def test_build_prompt_with_language(self):
        """Test prompt construction with language."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            prompt = model._build_prompt(language="en")
            assert "en language" in prompt

    def test_build_prompt_with_user_prompt(self):
        """Test prompt construction with user prompt."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            prompt = model._build_prompt(prompt="Focus on technical terms")
            assert "Focus on technical terms" in prompt

    def test_build_prompt_with_both(self):
        """Test prompt construction with language and user prompt."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            prompt = model._build_prompt(language="es", prompt="Medical terminology")
            assert "es language" in prompt
            assert "Medical terminology" in prompt

    def test_parse_response_success(self, mock_gemini_transcription_response):
        """Test successful response parsing."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            text = model._parse_response(mock_gemini_transcription_response)
            assert text == "This is a test transcription from Gemini"

    def test_parse_response_invalid_structure(self):
        """Test response parsing with invalid structure."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            with pytest.raises(RuntimeError, match="Failed to parse transcription"):
                model._parse_response({"invalid": "structure"})

    def test_transcribe_with_file_path(self, audio_file, mock_httpx_clients):
        """Test synchronous transcription with file path."""
        client, async_client = mock_httpx_clients

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            model.client = client
            model.async_client = async_client

            response = model.transcribe(audio_file)

            assert isinstance(response, TranscriptionResponse)
            assert response.text == "This is a test transcription from Gemini"
            assert response.model == "gemini-2.5-flash"

            # Verify HTTP call was made
            assert client.post.called
            call_args = client.post.call_args
            assert "generateContent" in call_args[0][0]
            assert "key=test-key" in call_args[0][0]

    def test_transcribe_with_binary_io(self, audio_file, mock_httpx_clients):
        """Test synchronous transcription with BinaryIO."""
        client, async_client = mock_httpx_clients

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            model.client = client
            model.async_client = async_client

            with open(audio_file, "rb") as f:
                response = model.transcribe(f)

            assert isinstance(response, TranscriptionResponse)
            assert response.text == "This is a test transcription from Gemini"

    def test_transcribe_with_language(self, audio_file, mock_httpx_clients):
        """Test transcription with language parameter."""
        client, async_client = mock_httpx_clients

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            model.client = client
            model.async_client = async_client

            response = model.transcribe(audio_file, language="pt")

            assert response.language == "pt"
            # Verify language was included in request
            call_args = client.post.call_args
            request_json = call_args[1]["json"]
            prompt_text = request_json["contents"][0]["parts"][0]["text"]
            assert "pt language" in prompt_text

    def test_transcribe_with_prompt(self, audio_file, mock_httpx_clients):
        """Test transcription with custom prompt."""
        client, async_client = mock_httpx_clients

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            model.client = client
            model.async_client = async_client

            response = model.transcribe(audio_file, prompt="Focus on names")

            # Verify custom prompt was included in request
            call_args = client.post.call_args
            request_json = call_args[1]["json"]
            prompt_text = request_json["contents"][0]["parts"][0]["text"]
            assert "Focus on names" in prompt_text

    def test_transcribe_error_handling(self, audio_file, mock_gemini_error_response):
        """Test error handling in transcription."""
        client = Mock()

        # Mock error response
        error_response = Mock()
        error_response.status_code = 400
        error_response.json.return_value = mock_gemini_error_response
        client.post.return_value = error_response

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            model.client = client

            with pytest.raises(RuntimeError, match="Google API error"):
                model.transcribe(audio_file)

    @pytest.mark.asyncio
    async def test_atranscribe_success(self, audio_file, mock_httpx_clients):
        """Test asynchronous transcription."""
        client, async_client = mock_httpx_clients

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            model.client = client
            model.async_client = async_client

            response = await model.atranscribe(audio_file)

            assert isinstance(response, TranscriptionResponse)
            assert response.text == "This is a test transcription from Gemini"
            assert response.model == "gemini-2.5-flash"

            # Verify async HTTP call was made
            assert async_client.post.called

    @pytest.mark.asyncio
    async def test_atranscribe_with_params(self, audio_file, mock_httpx_clients):
        """Test async transcription with language and prompt."""
        client, async_client = mock_httpx_clients

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            model.client = client
            model.async_client = async_client

            response = await model.atranscribe(
                audio_file,
                language="fr",
                prompt="Technical terms"
            )

            assert response.language == "fr"
            # Verify parameters were included in request
            call_args = async_client.post.call_args
            request_json = call_args[1]["json"]
            prompt_text = request_json["contents"][0]["parts"][0]["text"]
            assert "fr language" in prompt_text
            assert "Technical terms" in prompt_text

    @pytest.mark.asyncio
    async def test_atranscribe_error_handling(self, audio_file, mock_gemini_error_response):
        """Test error handling in async transcription."""
        async_client = AsyncMock()

        # Mock error response
        error_response = AsyncMock()
        error_response.status_code = 400
        error_response.json = Mock(return_value=mock_gemini_error_response)
        async_client.post.return_value = error_response

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = GoogleSpeechToTextModel()
            model.async_client = async_client

            with pytest.raises(RuntimeError, match="Google API error"):
                await model.atranscribe(audio_file)

    def test_factory_integration(self):
        """Test creation via AIFactory."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            model = AIFactory.create_speech_to_text("google", "gemini-2.5-flash")
            assert isinstance(model, GoogleSpeechToTextModel)
            assert model.provider == "google"
            assert model.get_model_name() == "gemini-2.5-flash"

    def test_factory_available_providers(self):
        """Test Google appears in available providers."""
        providers = AIFactory.get_available_providers()
        assert "google" in providers["speech_to_text"]
