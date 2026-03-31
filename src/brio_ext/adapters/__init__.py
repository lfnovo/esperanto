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
    def render(self, messages: List[Dict[str, str]], no_think: bool = False) -> RenderedPrompt:
        """Render Esperanto messages into provider-specific payloads.

        Args:
            messages: Conversation messages with 'role' and 'content' keys.
            no_think: When True, suppress the model's internal reasoning phase.
                Only meaningful for models that support a thinking/reasoning mode
                (e.g. Qwen3/Qwen3.5). Adapters for other model families should
                accept the parameter and ignore it.
        """
        raise NotImplementedError

    def clean_response(self, text: str) -> str:
        """
        Clean model-specific format markers from response text.

        Each adapter should override this to strip its model's format markers
        (e.g., [/INST] for Llama, <|im_end|> for Qwen/Phi, etc.).

        Default implementation returns text unchanged.
        """
        return text


__all__ = ["ChatAdapter", "RenderedPrompt"]
