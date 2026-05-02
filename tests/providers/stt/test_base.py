"""Tests for base speech-to-text model."""

import mimetypes
from dataclasses import dataclass
from typing import BinaryIO, Optional, Union

import pytest

from esperanto.common_types import TranscriptionResponse
from esperanto.providers.stt.base import SpeechToTextModel, _guess_audio_content_type


class TestGuessAudioContentType:
    def test_wav_returns_audio_type(self):
        result = _guess_audio_content_type("audio.wav")
        assert result.startswith("audio/")

    def test_flac_returns_audio_type(self):
        result = _guess_audio_content_type("audio.flac")
        assert result.startswith("audio/")

    def test_m4a_returns_audio_type(self):
        result = _guess_audio_content_type("audio.m4a")
        assert result.startswith("audio/")

    def test_ogg_returns_audio_ogg(self):
        result = _guess_audio_content_type("audio.ogg")
        assert result == "audio/ogg"

    def test_webm_returns_audio_type_or_fallback(self):
        result = _guess_audio_content_type("audio.webm")
        mime, _ = mimetypes.guess_type("audio.webm")
        if mime and mime.startswith("audio/"):
            assert result == mime
        else:
            assert result == "audio/mpeg"

    def test_unknown_extension_falls_back(self):
        assert _guess_audio_content_type("audio.xyz") == "audio/mpeg"

    def test_stream_without_name_falls_back(self):
        # getattr(stream, 'name', 'audio.mp3') gives 'audio.mp3' → audio/mpeg
        assert _guess_audio_content_type("audio.mp3") == "audio/mpeg"

    def test_non_audio_mime_falls_back(self):
        # .html is text/html — not audio
        assert _guess_audio_content_type("page.html") == "audio/mpeg"


def test_cannot_instantiate_abstract_base():
    """Test that SpeechToTextModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SpeechToTextModel()


def test_concrete_implementation_must_implement_all_abstract_methods():
    """Test that concrete implementations must implement all abstract methods."""

    @dataclass
    class IncompleteModel(SpeechToTextModel):
        """Test implementation missing required methods."""

        pass

    with pytest.raises(TypeError):
        IncompleteModel()


def test_concrete_implementation_works():
    """Test that a complete concrete implementation works."""

    @dataclass
    class TestModel(SpeechToTextModel):
        """Complete test implementation."""

        def transcribe(
            self,
            audio_file: Union[str, BinaryIO],
            language: Optional[str] = None,
            prompt: Optional[str] = None,
        ) -> TranscriptionResponse:
            return TranscriptionResponse(text="test")

        async def atranscribe(
            self,
            audio_file: Union[str, BinaryIO],
            language: Optional[str] = None,
            prompt: Optional[str] = None,
        ) -> TranscriptionResponse:
            return TranscriptionResponse(text="test")

        @property
        def provider(self) -> str:
            """Get the provider name."""
            return "test"

        @property
        def _get_models(self):
            """List all available models for this provider."""
            return []

        def _get_default_model(self) -> str:
            return "test-model"

    model = TestModel()
    assert model.provider == "test"
    assert model.get_model_name() == "test-model"


def test_model_name_from_config():
    """Test that model name can be set via config."""

    @dataclass
    class TestModel(SpeechToTextModel):
        """Test implementation with config handling."""

        def transcribe(
            self,
            audio_file: Union[str, BinaryIO],
            language: Optional[str] = None,
            prompt: Optional[str] = None,
        ) -> TranscriptionResponse:
            return TranscriptionResponse(text="test")

        async def atranscribe(
            self,
            audio_file: Union[str, BinaryIO],
            language: Optional[str] = None,
            prompt: Optional[str] = None,
        ) -> TranscriptionResponse:
            return TranscriptionResponse(text="test")

        @property
        def provider(self) -> str:
            return "test"

        @property
        def _get_models(self):
            """List all available models for this provider."""
            return []

        def _get_default_model(self) -> str:
            return "default-model"

    # Test with model name in config
    model = TestModel(config={"model_name": "config-model"})
    assert model.get_model_name() == "config-model"

    # Test with direct model name
    model = TestModel(model_name="direct-model")
    assert model.get_model_name() == "direct-model"

    # Test fallback to default
    model = TestModel()
    assert model.get_model_name() == "default-model"
