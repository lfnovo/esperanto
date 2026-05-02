"""Real integration tests for text-to-speech - these call actual APIs.

These tests verify that speech generation works correctly with real API calls.
They require API keys to be configured in the environment.

Run with: uv run pytest tests/integration/test_tts_real.py -v -s -m release
"""

import asyncio
import os

import pytest

from esperanto import AIFactory

# =============================================================================
# Test Configuration
# =============================================================================

TEST_TEXT = "Hello, Esperanto."


# =============================================================================
# OpenAI Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not configured",
)
class TestOpenAITTS:
    """Real integration tests for OpenAI text-to-speech."""

    def test_sync_generate_speech(self):
        """Test sync speech generation."""
        model = AIFactory.create_text_to_speech("openai", "tts-1")
        response = model.generate_speech(TEST_TEXT, voice="alloy")
        assert response.audio_data
        assert response.content_type.startswith("audio/")

    def test_async_agenerate_speech(self):
        """Test async speech generation."""
        model = AIFactory.create_text_to_speech("openai", "tts-1")

        async def _run() -> object:
            return await model.agenerate_speech(TEST_TEXT, voice="alloy")

        response = asyncio.run(_run())
        assert response.audio_data
        assert response.content_type.startswith("audio/")


# =============================================================================
# ElevenLabs Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("ELEVENLABS_API_KEY"),
    reason="ELEVENLABS_API_KEY not configured",
)
class TestElevenLabsTTS:
    """Real integration tests for ElevenLabs text-to-speech."""

    def test_sync_generate_speech(self):
        """Test sync speech generation."""
        model = AIFactory.create_text_to_speech("elevenlabs")
        voice = list(model.available_voices.keys())[0]
        response = model.generate_speech(TEST_TEXT, voice=voice)
        assert response.audio_data
        assert response.content_type.startswith("audio/")

    def test_async_agenerate_speech(self):
        """Test async speech generation."""
        model = AIFactory.create_text_to_speech("elevenlabs")
        voice = list(model.available_voices.keys())[0]

        async def _run() -> object:
            return await model.agenerate_speech(TEST_TEXT, voice=voice)

        response = asyncio.run(_run())
        assert response.audio_data
        assert response.content_type.startswith("audio/")


# =============================================================================
# Google Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
class TestGoogleTTS:
    """Real integration tests for Google text-to-speech."""

    def test_sync_generate_speech(self):
        """Test sync speech generation."""
        model = AIFactory.create_text_to_speech(
            "google",
            config={"api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")},
        )
        response = model.generate_speech(TEST_TEXT, voice="achernar")
        assert response.audio_data
        assert response.content_type.startswith("audio/")

    def test_async_agenerate_speech(self):
        """Test async speech generation."""
        model = AIFactory.create_text_to_speech(
            "google",
            config={"api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")},
        )

        async def _run() -> object:
            return await model.agenerate_speech(TEST_TEXT, voice="achernar")

        response = asyncio.run(_run())
        assert response.audio_data
        assert response.content_type.startswith("audio/")


# =============================================================================
# Vertex Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        and (os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT"))
    ),
    reason="Vertex TTS requires GOOGLE_APPLICATION_CREDENTIALS and a project env var (VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT)",
)
class TestVertexTTS:
    """Real integration tests for Vertex AI text-to-speech."""

    def test_sync_generate_speech(self):
        """Test sync speech generation."""
        model = AIFactory.create_text_to_speech("vertex")
        response = model.generate_speech(TEST_TEXT, voice="en-US-Standard-A")
        assert response.audio_data
        assert response.content_type.startswith("audio/")

    def test_async_agenerate_speech(self):
        """Test async speech generation."""
        model = AIFactory.create_text_to_speech("vertex")

        async def _run() -> object:
            return await model.agenerate_speech(TEST_TEXT, voice="en-US-Standard-A")

        response = asyncio.run(_run())
        assert response.audio_data
        assert response.content_type.startswith("audio/")


# =============================================================================
# Azure Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (
        (os.getenv("AZURE_OPENAI_API_KEY_TTS") or os.getenv("AZURE_OPENAI_API_KEY"))
        and (os.getenv("AZURE_OPENAI_ENDPOINT_TTS") or os.getenv("AZURE_OPENAI_ENDPOINT"))
        and (os.getenv("AZURE_OPENAI_API_VERSION_TTS") or os.getenv("AZURE_OPENAI_API_VERSION"))
    ),
    reason="Azure TTS requires API key, endpoint, and API version (AZURE_OPENAI_API_KEY[_TTS] + AZURE_OPENAI_ENDPOINT[_TTS] + AZURE_OPENAI_API_VERSION[_TTS])",
)
class TestAzureTTS:
    """Real integration tests for Azure text-to-speech."""

    def test_sync_generate_speech(self):
        """Test sync speech generation."""
        model = AIFactory.create_text_to_speech(
            "azure",
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_TTS"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_TTS"),
            },
        )
        response = model.generate_speech(TEST_TEXT, voice="alloy")
        assert response.audio_data
        assert response.content_type.startswith("audio/")

    def test_async_agenerate_speech(self):
        """Test async speech generation."""
        model = AIFactory.create_text_to_speech(
            "azure",
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_TTS"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_TTS"),
            },
        )

        async def _run() -> object:
            return await model.agenerate_speech(TEST_TEXT, voice="alloy")

        response = asyncio.run(_run())
        assert response.audio_data
        assert response.content_type.startswith("audio/")


# =============================================================================
# xAI Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("XAI_API_KEY"),
    reason="XAI_API_KEY not configured",
)
class TestXAITTS:
    """Real integration tests for xAI text-to-speech."""

    def test_sync_generate_speech(self):
        """Test sync speech generation."""
        model = AIFactory.create_text_to_speech("xai")
        response = model.generate_speech(TEST_TEXT, voice="eve")
        assert response.audio_data
        assert response.content_type.startswith("audio/")

    def test_async_agenerate_speech(self):
        """Test async speech generation."""
        model = AIFactory.create_text_to_speech("xai")

        async def _run() -> object:
            return await model.agenerate_speech(TEST_TEXT, voice="eve")

        response = asyncio.run(_run())
        assert response.audio_data
        assert response.content_type.startswith("audio/")


# =============================================================================
# Mistral Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not configured",
)
class TestMistralTTS:
    """Real integration tests for Mistral text-to-speech."""

    def test_sync_generate_speech(self):
        """Test sync speech generation."""
        model = AIFactory.create_text_to_speech("mistral")
        voice = list(model.available_voices.keys())[0]
        response = model.generate_speech(TEST_TEXT, voice=voice)
        assert response.audio_data
        assert response.content_type.startswith("audio/")

    def test_async_agenerate_speech(self):
        """Test async speech generation."""
        model = AIFactory.create_text_to_speech("mistral")
        voice = list(model.available_voices.keys())[0]

        async def _run() -> object:
            return await model.agenerate_speech(TEST_TEXT, voice=voice)

        response = asyncio.run(_run())
        assert response.audio_data
        assert response.content_type.startswith("audio/")


# =============================================================================
# OpenAI-Compatible Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("OPENAI_COMPATIBLE_BASE_URL_TTS") or os.getenv("OPENAI_COMPATIBLE_BASE_URL")),
    reason="OpenAI-compatible TTS requires OPENAI_COMPATIBLE_BASE_URL_TTS or OPENAI_COMPATIBLE_BASE_URL",
)
class TestOpenAICompatibleTTS:
    """Real integration tests for OpenAI-compatible text-to-speech."""

    def test_sync_generate_speech(self):
        """Test sync speech generation.

        Uses the provider's default voice rather than hardcoding 'alloy', so
        non-OpenAI-mimicking endpoints (which don't have an 'alloy' voice)
        still work. Override via OPENAI_COMPATIBLE_TTS_VOICE if the default
        voice is not appropriate for the configured backend.
        """
        voice_override = os.getenv("OPENAI_COMPATIBLE_TTS_VOICE")
        kwargs = {"voice": voice_override} if voice_override else {}
        model = AIFactory.create_text_to_speech(
            "openai-compatible",
            config={"base_url": os.getenv("OPENAI_COMPATIBLE_BASE_URL_TTS") or os.getenv("OPENAI_COMPATIBLE_BASE_URL")},
        )
        response = model.generate_speech(TEST_TEXT, **kwargs)
        assert response.audio_data
        assert response.content_type.startswith("audio/")

    def test_async_agenerate_speech(self):
        """Test async speech generation. See sync variant for voice handling notes."""
        voice_override = os.getenv("OPENAI_COMPATIBLE_TTS_VOICE")
        kwargs = {"voice": voice_override} if voice_override else {}
        model = AIFactory.create_text_to_speech(
            "openai-compatible",
            config={"base_url": os.getenv("OPENAI_COMPATIBLE_BASE_URL_TTS") or os.getenv("OPENAI_COMPATIBLE_BASE_URL")},
        )

        async def _run() -> object:
            return await model.agenerate_speech(TEST_TEXT, **kwargs)

        response = asyncio.run(_run())
        assert response.audio_data
        assert response.content_type.startswith("audio/")
