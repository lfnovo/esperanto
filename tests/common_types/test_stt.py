"""Tests for the speech-to-text common types."""

import pytest
from pydantic import ValidationError

from esperanto.common_types import (
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionUsage,
)


def test_transcription_segment_required_fields():
    """TranscriptionSegment requires text, start, end."""
    segment = TranscriptionSegment(text="hello", start=0.0, end=1.5)
    assert segment.text == "hello"
    assert segment.start == 0.0
    assert segment.end == 1.5
    assert segment.metadata is None


def test_transcription_segment_with_metadata():
    """TranscriptionSegment accepts arbitrary provider-specific metadata."""
    segment = TranscriptionSegment(
        text="hello",
        start=0.0,
        end=1.5,
        metadata={"avg_logprob": -0.25, "tokens": [50364, 50464]},
    )
    assert segment.metadata == {"avg_logprob": -0.25, "tokens": [50364, 50464]}


def test_transcription_segment_is_frozen():
    """TranscriptionSegment is immutable (Pydantic frozen=True)."""
    segment = TranscriptionSegment(text="hello", start=0.0, end=1.5)
    with pytest.raises(ValidationError):
        segment.text = "modified"  # type: ignore[misc]


def test_transcription_segment_missing_required_field():
    """TranscriptionSegment raises on missing required fields."""
    with pytest.raises(ValidationError):
        TranscriptionSegment(text="hello", start=0.0)  # type: ignore[call-arg]


def test_transcription_usage_all_optional():
    """TranscriptionUsage has all-optional fields, defaulting to None."""
    usage = TranscriptionUsage()
    assert usage.input_seconds is None
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


def test_transcription_usage_partial_population():
    """TranscriptionUsage accepts any subset of fields (e.g. audio-only providers)."""
    usage = TranscriptionUsage(input_seconds=12.5)
    assert usage.input_seconds == 12.5
    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_transcription_usage_is_frozen():
    """TranscriptionUsage is immutable (Pydantic frozen=True)."""
    usage = TranscriptionUsage(input_seconds=10.0)
    with pytest.raises(ValidationError):
        usage.input_seconds = 20.0  # type: ignore[misc]


def test_transcription_response_segments_default_none():
    """TranscriptionResponse.segments defaults to None for backward compatibility."""
    response = TranscriptionResponse(text="hello world")
    assert response.segments is None
    assert response.usage is None
    assert response.duration is None


def test_transcription_response_accepts_segments():
    """TranscriptionResponse stores a list of TranscriptionSegment."""
    segments = [
        TranscriptionSegment(text="hello", start=0.0, end=0.5),
        TranscriptionSegment(text="world", start=0.5, end=1.0),
    ]
    response = TranscriptionResponse(text="hello world", segments=segments)
    assert response.segments is not None
    assert len(response.segments) == 2
    assert response.segments[0].text == "hello"
    assert response.segments[1].start == 0.5


def test_transcription_response_accepts_transcription_usage():
    """TranscriptionResponse.usage accepts the new TranscriptionUsage type."""
    usage = TranscriptionUsage(input_seconds=12.5, total_tokens=42)
    response = TranscriptionResponse(text="hello", usage=usage)
    assert response.usage is not None
    assert response.usage.input_seconds == 12.5
    assert response.usage.total_tokens == 42
