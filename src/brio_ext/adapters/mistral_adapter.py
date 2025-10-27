"""Mistral family prompt adapter."""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class MistralAdapter(ChatAdapter):
    """Render Esperanto messages into Mistral-style [INST] prompts."""

    def can_handle(self, model_id: str) -> bool:
        return "mistral" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]

        system_text = "\n".join(system_messages) if system_messages else ""
        user_text = "\n".join(user_messages)

        # Mistral format - model generates content, brio_ext will fence it
        prompt = f"[INST] {system_text}\n{user_text} [/INST]"
        return {"prompt": prompt, "stop": []}
