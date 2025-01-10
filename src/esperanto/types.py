"""Common type definitions for Esperanto."""
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class Usage:
    """Usage information for API calls."""
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_characters: Optional[int] = None
    audio_duration: Optional[float] = None  # Duration in seconds


@dataclass
class Model:
    """Model information from providers."""
    id: str
    owned_by: str
    context_window: Optional[int] = None
    type: Literal["language", "embedding", "text_to_speech", "speech_to_text"] = "language"


@dataclass
class TranscriptionResponse:
    """Response from speech-to-text transcription."""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None  # Duration in seconds
    usage: Optional[Usage] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AudioResponse:
    """Response from text-to-speech generation."""
    audio_data: bytes
    duration: Optional[float] = None  # Duration in seconds
    content_type: str = "audio/mp3"  # Default to MP3
    usage: Optional[Usage] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
