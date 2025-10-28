"""Llama family prompt adapter."""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class LlamaAdapter(ChatAdapter):
    """Render Esperanto messages into Llama-compatible prompts."""

    def can_handle(self, model_id: str) -> bool:
        return "llama" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]

        system_text = "\n".join(system_messages) if system_messages else ""
        user_text = "\n".join(user_messages)

        # Llama format - model generates content, brio_ext will fence it
        prompt = f"[INST] <<SYS>>\n{system_text}\n<</SYS>>\n{user_text} [/INST]"
        # Stop token prevents model from starting a new instruction turn
        # Note: We can't use [/INST] as stop token since it's in the prompt!
        return {"prompt": prompt, "stop": ["[INST]"]}

    def clean_response(self, text: str) -> str:
        """Remove Llama format markers from response."""
        # Strip [INST] and [/INST] markers that may appear in response
        cleaned = text
        for marker in ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()
