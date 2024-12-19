"""Tests for base speech-to-text model."""
import pytest
from dataclasses import dataclass
from typing import Optional, Union, BinaryIO

from esperanto.providers.stt.base import SpeechToTextModel
from esperanto.types import TranscriptionResponse


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
            return "test"
        
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