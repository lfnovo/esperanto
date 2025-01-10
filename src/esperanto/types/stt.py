"""Types for speech-to-text functionality."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResponse:
    """Response from a speech-to-text transcription.
    
    Attributes:
        text: The transcribed text.
        language: Detected or specified language code.
        duration: Duration of the audio in seconds.
        model: Name of the model used.
        provider: Name of the provider.
    """
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
