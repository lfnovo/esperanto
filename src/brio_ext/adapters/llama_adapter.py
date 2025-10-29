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

        # Stop tokens for Llama 3.1:
        # - "[INST]" prevents model from starting a new instruction turn
        # - "<|eot_id|>" is Llama 3.1's end-of-turn token
        # - "<|end_of_text|>" is the EOS token
        # Note: We can't use [/INST] as stop token since it appears in the prompt!
        return {"prompt": prompt, "stop": ["[INST]", "<|eot_id|>", "<|end_of_text|>"]}

    def clean_response(self, text: str) -> str:
        """Remove Llama format markers from response."""
        import re
        import sys
        print(f"[BRIO_DEBUG] LlamaAdapter.clean_response called", file=sys.stderr)
        print(f"[BRIO_DEBUG] Input text ends with: {text[-100:]}", file=sys.stderr)
        cleaned = text

        # Strip Llama instruction format markers
        # Note: [/SYS] is not standard Llama format but models sometimes hallucinate it
        for marker in ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "[/SYS]", "[SYS]"]:
            if marker in cleaned:
                print(f"[BRIO_DEBUG] Found and removing marker: {marker}", file=sys.stderr)
            cleaned = cleaned.replace(marker, "")

        # Strip Llama 3.1 special tokens
        for marker in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]:
            if marker in cleaned:
                print(f"[BRIO_DEBUG] Found and removing token: {marker}", file=sys.stderr)
            cleaned = cleaned.replace(marker, "")

        # Aggressive cleanup: Remove repeated whitespace
        # This catches cases where model generates "[/SYS]  [/SYS]  [/SYS]..." after JSON
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()

        # Note: Incomplete tokens (e.g., "[/", "<|e") are handled by
        # _strip_trailing_incomplete_tokens() in factory.py for ALL adapters

        print(f"[BRIO_DEBUG] Output text ends with: {cleaned[-100:]}", file=sys.stderr)
        return cleaned
