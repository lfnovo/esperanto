"""Common test fixtures and configuration."""

import os
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from google.cloud import speech, texttospeech
from google.cloud.speech_v1.types import RecognizeResponse
from google.cloud.texttospeech_v1.types import SynthesizeSpeechResponse
from openai import AsyncOpenAI

from esperanto.base import (
    Message,
    ChatCompletion,
    Embedding,
    AudioTranscription,
    AudioSynthesis,
)
from esperanto.providers.speech_to_text.google import GoogleSpeechToTextModel
from esperanto.providers.text_to_speech.gemini import GeminiTextToSpeechModel
from esperanto.providers.text_to_speech.openai import OpenAITextToSpeechModel


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GOOGLE_APPLICATION_CREDENTIALS": "test-google-creds.json",
        "GOOGLE_PROJECT": "test-project",
        "ELEVENLABS_API_KEY": "test-elevenlabs-key",
        "GROQ_API_KEY": "test-groq-key",
        "OPENROUTER_API_KEY": "test-openrouter-key",
        "XAI_API_KEY": "test-xai-key",
        "LITELLM_API_KEY": "test-litellm-key",
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def mock_google_credentials():
    """Mock Google credentials."""
    with patch("google.auth.default", return_value=(Mock(), "test-project")):
        yield


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Mock OpenAI client."""
    return AsyncMock()


@pytest.fixture
def mock_anthropic_client() -> AsyncMock:
    """Mock Anthropic client."""
    return AsyncMock()


@pytest.fixture
def mock_vertex_client() -> MagicMock:
    """Mock Vertex AI client."""
    return MagicMock()


@pytest.fixture
def mock_gemini_client() -> AsyncMock:
    """Mock Gemini client."""
    return AsyncMock()


@pytest.fixture
def mock_elevenlabs_client() -> AsyncMock:
    """Mock ElevenLabs client."""
    return AsyncMock()


@pytest.fixture
def mock_groq_client() -> AsyncMock:
    """Mock Groq client."""
    return AsyncMock()


@pytest.fixture
def mock_openrouter_client() -> AsyncMock:
    """Mock OpenRouter client."""
    return AsyncMock()


@pytest.fixture
def mock_xai_client() -> AsyncMock:
    """Mock XAI client."""
    return AsyncMock()


@pytest.fixture
def mock_litellm_client() -> AsyncMock:
    """Mock LiteLLM client."""
    return AsyncMock()


@pytest.fixture
def model():
    """Create a model instance for testing."""
    return GoogleSpeechToTextModel()


@pytest.fixture
def mock_client():
    """Create a mock client."""
    return AsyncMock(spec=speech.SpeechClient)


@pytest.fixture
def mock_response():
    """Create a mock response."""
    response = Mock(spec=RecognizeResponse)
    response.results = [Mock(alternatives=[Mock(transcript="Hello, world!")])]
    return response


@pytest.fixture
def tts_model():
    """Create a text-to-speech model instance for testing."""
    return GeminiTextToSpeechModel()


@pytest.fixture
def tts_mock_client():
    """Create a mock text-to-speech client."""
    return AsyncMock(spec=texttospeech.TextToSpeechClient)


@pytest.fixture
def tts_mock_response():
    """Create a mock text-to-speech response."""
    response = Mock(spec=SynthesizeSpeechResponse)
    response.audio_content = b"audio_content"
    return response


@pytest.fixture
def openai_model():
    """Create an OpenAI text-to-speech model instance for testing."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        return OpenAITextToSpeechModel(model_name="tts-1")


@pytest.fixture
def openai_mock_client():
    """Create a mock OpenAI client."""
    return AsyncMock(spec=AsyncOpenAI)
