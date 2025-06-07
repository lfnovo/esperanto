"""Text-to-speech type definitions for Esperanto."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from esperanto.common_types.response import Usage


class Voice(BaseModel):
    """Voice information for text-to-speech."""

    name: str = Field(description="Display name of the voice")
    id: str = Field(description="Unique identifier for the voice")
    gender: str = Field(description="Gender of the voice (e.g., 'FEMALE', 'MALE')")
    language_code: Optional[str] = Field(
        default=None, description="Language code (e.g., 'en-US')"
    )
    description: Optional[str] = Field(
        default=None, description="Description of the voice"
    )
    accent: Optional[str] = Field(
        default=None, description="Accent of the voice (e.g., 'American')"
    )
    age: Optional[str] = Field(
        default=None, description="Age category of the voice (e.g., 'young')"
    )
    use_case: Optional[str] = Field(
        default=None, description="Recommended use case for the voice"
    )
    preview_url: Optional[str] = Field(
        default=None, description="URL to a preview audio sample"
    )

    model_config = ConfigDict(frozen=True)


class AudioResponse(BaseModel):
    """Response from text-to-speech generation."""

    audio_data: bytes = Field(description="The generated audio data")
    duration: Optional[float] = Field(
        default=None, description="Duration of the audio in seconds"
    )
    content_type: str = Field(
        default="audio/mp3", description="MIME type of the audio data"
    )
    usage: Optional[Usage] = Field(
        default=None, description="Usage statistics for this generation"
    )
    model: Optional[str] = Field(
        default=None, description="The model used for generation"
    )
    voice: Optional[str] = Field(
        default=None, description="The voice used for generation"
    )
    provider: Optional[str] = Field(
        default=None, description="The provider that generated this audio"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata from the provider"
    )

    model_config = ConfigDict(frozen=True)
