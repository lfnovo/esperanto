"""Qwen family prompt adapter."""

from __future__ import annotations

import os
from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt

# Sentinel prepended by BrioLangChainWrapper._convert_messages when no_think=True.
# We detect and handle it here because the /completion endpoint bypasses the
# Qwen3 native Jinja template (which normally processes this token).  Instead
# we prefill the assistant turn with an empty <think></think> block, which is
# the standard technique for telling Qwen3/Qwen3.5 to skip the reasoning phase
# when using raw prompt completion.
_NO_THINK_PREFIX = "/no_think\n"


class QwenAdapter(ChatAdapter):
    """Render Esperanto messages into ChatML prompts for Qwen models."""

    def can_handle(self, model_id: str) -> bool:
        return "qwen" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        if os.getenv("BRIO_DEBUG"):
            print(f"[QwenAdapter] Rendering {len(messages)} messages")

        def block(role: str, content: str) -> str:
            return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

        # Detect /no_think flag injected by BrioLangChainWrapper._convert_messages.
        # Strip the sentinel and remember to disable the thinking prefix.
        no_think = False
        cleaned_messages = []
        for m in messages:
            if (
                not no_think
                and m["role"] == "user"
                and m["content"].startswith(_NO_THINK_PREFIX)
            ):
                no_think = True
                cleaned_messages.append(
                    {**m, "content": m["content"][len(_NO_THINK_PREFIX):]}
                )
            else:
                cleaned_messages.append(m)

        system_messages = [m["content"] for m in cleaned_messages if m["role"] == "system"]
        user_and_tool = [m for m in cleaned_messages if m["role"] != "system"]

        system_text = "\n".join(system_messages) if system_messages else ""
        conversation = "".join(block(m["role"], m["content"]) for m in user_and_tool)

        if no_think:
            # Prefill assistant turn with an empty <think></think> block.
            # This signals to Qwen3/Qwen3.5 that the reasoning phase is complete
            # and the model should produce the answer directly, without generating
            # a new <think> block.  This works via raw /completion because the
            # model is conditioned on having "finished" thinking.
            assistant_prefix = "<|im_start|>assistant\n<think>\n\n</think>\n"
        else:
            # Normal: model will generate <think>...</think> then the answer
            assistant_prefix = "<|im_start|>assistant\n"

        prompt = f"{block('system', system_text)}{conversation}{assistant_prefix}"

        if os.getenv("BRIO_DEBUG"):
            print(f"[QwenAdapter] Generated prompt: {len(prompt)} chars")
            print(f"[QwenAdapter] no_think={no_think}")
            print(f"[QwenAdapter] Stop tokens: ['<|im_end|>']")

        return {"prompt": prompt, "stop": ["<|im_end|>"]}

    def clean_response(self, text: str) -> str:
        """Remove ChatML format markers from response."""
        # Strip any ChatML markers that leaked through
        cleaned = text
        for marker in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()
