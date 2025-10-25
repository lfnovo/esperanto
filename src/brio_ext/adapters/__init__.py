"""Adapter interfaces for BrioDocs prompt rendering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Union

RenderedPrompt = Dict[str, Union[str, List[str], List[Dict[str, str]]]]


class ChatAdapter(ABC):
    """Interface for adapting Esperanto-style messages to provider-specific prompts."""

    @abstractmethod
    def can_handle(self, model_id: str) -> bool:
        """Return True if the adapter should handle the given model identifier."""
        raise NotImplementedError

    @abstractmethod
    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        """Render Esperanto messages into provider-specific payloads."""
        raise NotImplementedError


__all__ = ["ChatAdapter", "RenderedPrompt"]
