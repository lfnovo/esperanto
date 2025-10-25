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

        prompt = f"[INST] <<SYS>>\n{system_text}\n<</SYS>>\n{user_text} [/INST]\n<out>"
        return {"prompt": prompt, "stop": ["</out>"]}
