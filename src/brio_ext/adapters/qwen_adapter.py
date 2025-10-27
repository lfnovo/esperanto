"""Qwen family prompt adapter."""

from __future__ import annotations

import os
from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class QwenAdapter(ChatAdapter):
    """Render Esperanto messages into ChatML prompts for Qwen models."""

    def can_handle(self, model_id: str) -> bool:
        return "qwen" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        if os.getenv("BRIO_DEBUG"):
            print(f"[QwenAdapter] Rendering {len(messages)} messages")

        def block(role: str, content: str) -> str:
            return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        user_and_tool = [m for m in messages if m["role"] != "system"]

        system_text = "\n".join(system_messages) if system_messages else ""
        conversation = "".join(block(m["role"], m["content"]) for m in user_and_tool)

        # Start assistant turn - model will generate content, brio_ext will fence it
        prompt = f"{block('system', system_text)}{conversation}<|im_start|>assistant\n"

        if os.getenv("BRIO_DEBUG"):
            print(f"[QwenAdapter] Generated prompt: {len(prompt)} chars")
            print(f"[QwenAdapter] Stop tokens: ['<|im_end|>']")

        return {"prompt": prompt, "stop": ["<|im_end|>"]}

    def clean_response(self, text: str) -> str:
        """Remove ChatML format markers from response."""
        # Strip any ChatML markers that leaked through
        cleaned = text
        for marker in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()
