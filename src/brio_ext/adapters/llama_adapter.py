"""Llama family prompt adapter."""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class LlamaAdapter(ChatAdapter):
    """Render Esperanto messages into Llama-compatible prompts."""

    def can_handle(self, model_id: str) -> bool:
        return "llama" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]]) -> RenderedPrompt:
        """
        Render messages for Llama models.

        For llamacpp provider with native chat template support (Llama 3.1+),
        we return messages directly and let llamacpp apply the correct template.
        This avoids format mismatches when the server is configured with
        --chat-template llama-3 or similar.

        For other providers or Llama 2, we'd need to manually render prompts,
        but for now we rely on llamacpp's native template handling.
        """
        # Return messages in OpenAI format for llamacpp to apply native chat template
        # This works with llamacpp's /v1/chat/completions endpoint which handles:
        # - Llama 3.1: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        # - Llama 2: [INST] <<SYS>>...<</SYS>>... [/INST]
        # - Other Llama variants automatically
        return {
            "messages": messages,
            "stop": ["<|eot_id|>", "<|end_of_text|>"]  # Llama 3.1+ stop tokens
        }

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
