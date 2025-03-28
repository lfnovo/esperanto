"""Speech-to-text type definitions for Esperanto."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from esperanto.common_types.response import Usage


class TranscriptionResponse(BaseModel):
    """Response from speech-to-text transcription."""

    text: str = Field(description="The transcribed text")
    language: Optional[str] = Field(
        default=None, description="The detected or specified language of the audio"
    )
    duration: Optional[float] = Field(
        default=None, description="Duration of the audio in seconds"
    )
    usage: Optional[Usage] = Field(
        default=None, description="Usage statistics for this transcription"
    )
    model: Optional[str] = Field(
        default=None, description="The model used for transcription"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata from the provider"
    )

    model_config = ConfigDict(frozen=True)
