"""Real integration tests for speech-to-text - these call actual APIs.

These tests verify that transcription works correctly with real API calls.
They require API keys to be configured in the environment.

Run with: uv run pytest tests/integration/test_stt_real.py -v -s -m release
"""

import asyncio
import os
from pathlib import Path

import pytest

from esperanto import AIFactory

# =============================================================================
# Test Configuration
# =============================================================================

EXPECTED_TRANSCRIPT_FRAGMENT = "Supernova"
SAMPLE_AUDIO = Path(__file__).parent.parent / "fixtures" / "sample.mp3"


# =============================================================================
# OpenAI Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not configured",
)
class TestOpenAISTT:
    """Real integration tests for OpenAI speech-to-text."""

    def test_sync_transcribe(self):
        """Test sync transcription with file-path input."""
        model = AIFactory.create_speech_to_text("openai", "whisper-1")
        result = model.transcribe(str(SAMPLE_AUDIO))
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_sync_transcribe_binary_io(self):
        """Test sync transcription with BinaryIO input."""
        model = AIFactory.create_speech_to_text("openai", "whisper-1")
        with open(SAMPLE_AUDIO, "rb") as f:
            result = model.transcribe(f)
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_async_atranscribe(self):
        """Test async transcription with file-path input."""
        model = AIFactory.create_speech_to_text("openai", "whisper-1")

        async def _run() -> object:
            return await model.atranscribe(str(SAMPLE_AUDIO))

        result = asyncio.run(_run())
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_async_atranscribe_binary_io(self):
        """Test async transcription with BinaryIO input."""
        model = AIFactory.create_speech_to_text("openai", "whisper-1")

        async def _run() -> object:
            with open(SAMPLE_AUDIO, "rb") as f:
                return await model.atranscribe(f)

        result = asyncio.run(_run())
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()


# =============================================================================
# Groq Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not configured",
)
class TestGroqSTT:
    """Real integration tests for Groq speech-to-text."""

    def test_sync_transcribe(self):
        """Test sync transcription."""
        model = AIFactory.create_speech_to_text("groq", "whisper-large-v3")
        result = model.transcribe(str(SAMPLE_AUDIO))
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_async_atranscribe(self):
        """Test async transcription."""
        model = AIFactory.create_speech_to_text("groq", "whisper-large-v3")

        async def _run() -> object:
            return await model.atranscribe(str(SAMPLE_AUDIO))

        result = asyncio.run(_run())
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()


# =============================================================================
# Google Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
class TestGoogleSTT:
    """Real integration tests for Google speech-to-text."""

    def test_sync_transcribe(self):
        """Test sync transcription."""
        model = AIFactory.create_speech_to_text(
            "google",
            "gemini-2.5-flash",
            config={"api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")},
        )
        result = model.transcribe(str(SAMPLE_AUDIO))
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_async_atranscribe(self):
        """Test async transcription."""
        model = AIFactory.create_speech_to_text(
            "google",
            "gemini-2.5-flash",
            config={"api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")},
        )

        async def _run() -> object:
            return await model.atranscribe(str(SAMPLE_AUDIO))

        result = asyncio.run(_run())
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()


# =============================================================================
# Azure Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY_STT"),
    reason="AZURE_OPENAI_API_KEY_STT not configured",
)
class TestAzureSTT:
    """Real integration tests for Azure speech-to-text."""

    def test_sync_transcribe(self):
        """Test sync transcription."""
        model = AIFactory.create_speech_to_text(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_STT", "whisper-1"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_STT"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_STT"),
            },
        )
        result = model.transcribe(str(SAMPLE_AUDIO))
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_async_atranscribe(self):
        """Test async transcription."""
        model = AIFactory.create_speech_to_text(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_STT", "whisper-1"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_STT"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_STT"),
            },
        )

        async def _run() -> object:
            return await model.atranscribe(str(SAMPLE_AUDIO))

        result = asyncio.run(_run())
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()


# =============================================================================
# Mistral Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not configured",
)
class TestMistralSTT:
    """Real integration tests for Mistral speech-to-text."""

    def test_sync_transcribe(self):
        """Test sync transcription."""
        model = AIFactory.create_speech_to_text("mistral", "voxtral-mini-latest")
        result = model.transcribe(str(SAMPLE_AUDIO))
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_async_atranscribe(self):
        """Test async transcription."""
        model = AIFactory.create_speech_to_text("mistral", "voxtral-mini-latest")

        async def _run() -> object:
            return await model.atranscribe(str(SAMPLE_AUDIO))

        result = asyncio.run(_run())
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()


# =============================================================================
# ElevenLabs Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("ELEVENLABS_API_KEY"),
    reason="ELEVENLABS_API_KEY not configured",
)
class TestElevenLabsSTT:
    """Real integration tests for ElevenLabs speech-to-text."""

    def test_sync_transcribe(self):
        """Test sync transcription."""
        model = AIFactory.create_speech_to_text("elevenlabs")
        result = model.transcribe(str(SAMPLE_AUDIO))
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_async_atranscribe(self):
        """Test async transcription."""
        model = AIFactory.create_speech_to_text("elevenlabs")

        async def _run() -> object:
            return await model.atranscribe(str(SAMPLE_AUDIO))

        result = asyncio.run(_run())
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()


# =============================================================================
# OpenAI-Compatible Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENAI_COMPATIBLE_STT_BASE_URL"),
    reason="OPENAI_COMPATIBLE_STT_BASE_URL not configured",
)
class TestOpenAICompatibleSTT:
    """Real integration tests for OpenAI-compatible speech-to-text."""

    def test_sync_transcribe(self):
        """Test sync transcription."""
        model = AIFactory.create_speech_to_text(
            "openai-compatible",
            "whisper-1",
            config={"base_url": os.getenv("OPENAI_COMPATIBLE_STT_BASE_URL")},
        )
        result = model.transcribe(str(SAMPLE_AUDIO))
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()

    def test_async_atranscribe(self):
        """Test async transcription."""
        model = AIFactory.create_speech_to_text(
            "openai-compatible",
            "whisper-1",
            config={"base_url": os.getenv("OPENAI_COMPATIBLE_STT_BASE_URL")},
        )

        async def _run() -> object:
            return await model.atranscribe(str(SAMPLE_AUDIO))

        result = asyncio.run(_run())
        assert EXPECTED_TRANSCRIPT_FRAGMENT.lower() in result.text.lower()
