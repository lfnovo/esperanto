"""Types for text-to-speech functionality."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AudioResponse:
    """Response from a text-to-speech generation.
    
    Attributes:
        audio_data: Raw audio data in bytes.
        duration: Duration of the audio in seconds.
        content_type: MIME type of the audio (e.g., 'audio/mp3').
        model: Name of the model used.
        voice: Voice ID or name used.
        provider: Name of the provider.
        metadata: Additional provider-specific metadata.
    """
    audio_data: bytes
    duration: Optional[float] = None
    content_type: str = "audio/mp3"
    model: Optional[str] = None
    voice: Optional[str] = None
    provider: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Voice:
    """Common voice representation across all TTS providers."""

    name: str
    """The name of the voice."""

    id: str
    """The unique identifier for the voice."""

    gender: str
    """The gender of the voice (MALE, FEMALE, NEUTRAL)."""

    language_code: str
    """The language code of the voice (e.g., 'en-US', 'pt-BR')."""

    description: Optional[str] = None
    """Optional description of the voice."""

    accent: Optional[str] = None
    """Optional accent of the voice (e.g., 'british', 'american')."""

    age: Optional[str] = None
    """Optional age category of the voice (e.g., 'young', 'middle-aged', 'old')."""

    use_case: Optional[str] = None
    """Optional primary use case of the voice (e.g., 'narration', 'conversational')."""

    preview_url: Optional[str] = None
    """Optional URL to preview the voice."""

    @staticmethod
    def normalize_gender(gender: str) -> str:
        """Normalize gender values across providers."""
        gender = gender.upper()
        if gender in ["MALE", "M"]:
            return "MALE"
        elif gender in ["FEMALE", "F"]:
            return "FEMALE"
        else:
            return "NEUTRAL"

    @staticmethod
    def normalize_language_code(language_code: str) -> str:
        """Normalize language codes across providers."""
        # Add language code normalization as needed
        return language_code

    @staticmethod
    def normalize_age(age: Optional[str]) -> Optional[str]:
        """Normalize age categories across providers."""
        if not age:
            return None
        age = age.lower()
        if "young" in age:
            return "young"
        elif "old" in age:
            return "old"
        else:
            return "middle-aged"
