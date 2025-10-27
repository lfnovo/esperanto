"""Phi family prompt adapter."""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class PhiAdapter(ChatAdapter):
    """Adapter that keeps Phi models on vanilla chat payloads."""

    def can_handle(self, model_id: str) -> bool:
        mid = (model_id or "").lower()
        return "phi-4-mini" in mid or mid.startswith("phi")

    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        # Phi models follow the OpenAI ChatML style in most runtimes.
        # Let the provider handle stop tokens, brio_ext will fence the response.
        return {"messages": messages, "stop": []}
