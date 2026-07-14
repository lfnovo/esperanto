"""Embedding validation utilities."""

from typing import Any, List


def validate_and_decode_embedding(idx: int, raw: Any) -> List[float]:
    """Validate and decode a raw embedding value from a provider response.

    Raises RuntimeError if the embedding is null, empty, or contains null values.
    """
    if raw is None or len(raw) == 0 or any(v is None for v in raw):
        raise RuntimeError(
            f"Embedding at index {idx} is null, empty, or contains null values. "
            "This typically happens when the input is too short or contains only special tokens. "
            "Consider filtering very short inputs before embedding."
        )
    return [float(v) for v in raw]
