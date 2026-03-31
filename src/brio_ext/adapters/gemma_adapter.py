"""Gemma family prompt adapter."""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class GemmaAdapter(ChatAdapter):
    """Render Esperanto messages into Gemma turn-based prompts."""

    def can_handle(self, model_id: str) -> bool:
        return "gemma" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]], no_think: bool = False) -> RenderedPrompt:
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]

        system_text = "\n".join(system_messages) if system_messages else ""
        user_text = "\n".join(user_messages)

        # Gemma format - model generates content, brio_ext will fence it
        prompt = (
            f"<start_of_turn>system\n{system_text}\n<end_of_turn>\n"
            f"<start_of_turn>user\n{user_text}\n<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        return {"prompt": prompt, "stop": ["<end_of_turn>"]}

    def clean_response(self, text: str) -> str:
        """Remove Gemma format markers from response."""
        # Strip any Gemma turn markers that leaked through
        cleaned = text
        for marker in ["<start_of_turn>", "<end_of_turn>"]:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()
