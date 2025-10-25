"""Qwen family prompt adapter."""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class QwenAdapter(ChatAdapter):
    """Render Esperanto messages into ChatML prompts for Qwen models."""

    def can_handle(self, model_id: str) -> bool:
        return "qwen" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        def block(role: str, content: str) -> str:
            return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        user_and_tool = [m for m in messages if m["role"] != "system"]

        system_text = "\n".join(system_messages) if system_messages else ""
        conversation = "".join(block(m["role"], m["content"]) for m in user_and_tool)

        # Add trailing newline after <out> so the model starts generating content
        prompt = f"{block('system', system_text)}{conversation}<|im_start|>assistant\n<out>\n"
        return {"prompt": prompt, "stop": ["</out>", "<|im_end|>"]}
