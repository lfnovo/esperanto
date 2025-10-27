"""Phi family prompt adapter."""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class PhiAdapter(ChatAdapter):
    """Render Esperanto messages into ChatML prompts for Phi models."""

    def can_handle(self, model_id: str) -> bool:
        mid = (model_id or "").lower()
        return "phi-4-mini" in mid or mid.startswith("phi")

    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        # Phi models use ChatML format like Qwen
        def block(role: str, content: str) -> str:
            return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

        # Build conversation with all messages
        prompt_parts = []
        for msg in messages:
            prompt_parts.append(block(msg["role"], msg["content"]))

        # Start assistant turn
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "".join(prompt_parts)

        # Stop at <|im_end|> to prevent model from continuing conversation
        return {"prompt": prompt, "stop": ["<|im_end|>"]}
