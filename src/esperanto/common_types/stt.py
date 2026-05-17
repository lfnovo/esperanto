"""Speech-to-text type definitions for Esperanto."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TranscriptionSegment(BaseModel):
    """A single timestamped segment of a transcription."""

    text: str = Field(description="Segment text")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Provider-specific extras for this segment "
            "(e.g. avg_logprob, confidence, tokens). See provider docs for the keys."
        ),
    )

    model_config = ConfigDict(frozen=True)


class TranscriptionUsage(BaseModel):
    """Usage statistics for a transcription request."""

    input_seconds: Optional[float] = Field(
        default=None,
        description="Audio duration sent to the provider, in seconds (if reported).",
    )
    input_tokens: Optional[int] = Field(
        default=None, description="Input/prompt tokens consumed (if reported)."
    )
    output_tokens: Optional[int] = Field(
        default=None, description="Output/completion tokens produced (if reported)."
    )
    total_tokens: Optional[int] = Field(
        default=None, description="Total tokens for the request (if reported)."
    )

    model_config = ConfigDict(frozen=True)


class TranscriptionResponse(BaseModel):
    """Response from speech-to-text transcription."""

    text: str = Field(description="The transcribed text")
    language: Optional[str] = Field(
        default=None, description="The detected or specified language of the audio"
    )
    duration: Optional[float] = Field(
        default=None, description="Duration of the audio in seconds"
    )
    usage: Optional[TranscriptionUsage] = Field(
        default=None, description="Usage statistics for this transcription"
    )
    model: Optional[str] = Field(
        default=None, description="The model used for transcription"
    )
    provider: Optional[str] = Field(
        default=None, description="The provider that produced this transcription"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata from the provider"
    )
    segments: Optional[List[TranscriptionSegment]] = Field(
        default=None,
        description=(
            "Timestamped segments of the transcription, when the provider supports it. "
            "Providers that don't return segment-level data leave this as None."
        ),
    )

    model_config = ConfigDict(frozen=True)
