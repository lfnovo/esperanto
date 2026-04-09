"""Mistral family prompt adapter.

Handles Mistral 7B Instruct v0.3, Ministral 3B/8B, and related models.

The official Mistral chat template places the system message appended to the
LAST user turn. This design likely assumes the system message is short
(guardrails, persona) — not a large document context. When used for document
QA with multi-turn conversation, the model sees the entire dialogue history
without any context, then gets the system prompt + question crammed into the
final turn. In practice this causes the model to largely ignore the system
content.

To work around this, we inject the system message as a synthetic first
user/assistant exchange so the model sees the context before any dialogue
turns. This produces significantly better results for document-grounded chat.

Rendered format:

    <s>[INST] {system}[/INST] OK, I will follow these instructions.</s>
    [INST] {user1}[/INST] {assistant1}</s>[INST] {user2}[/INST]

See: https://docs.mistral.ai/cookbooks/concept-deep-dive-tokenization-chat_templates
"""

from __future__ import annotations

from typing import Dict, List

from loguru import logger

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class MistralAdapter(ChatAdapter):
    """Render Esperanto messages using the Mistral Instruct template."""

    def can_handle(self, model_id: str) -> bool:
        return "mistral" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]], no_think: bool = False) -> RenderedPrompt:
        logger.info(f"[MistralAdapter] Rendering {len(messages)} messages")

        # Extract optional leading system message and convert to a user
        # message at the start of the conversation so the model sees
        # the context before any dialogue turns.
        system_text = ""
        loop_messages = list(messages)
        if loop_messages and loop_messages[0]["role"] == "system":
            system_text = loop_messages.pop(0)["content"]
            logger.info(f"[MistralAdapter] System message: {len(system_text)} chars (injected as first user turn)")

        logger.info(f"[MistralAdapter] Conversation messages: {len(loop_messages)}")
        for i, msg in enumerate(loop_messages):
            logger.info(f"[MistralAdapter]   [{i}] role={msg['role']}, {len(msg['content'])} chars")

        parts: list[str] = []

        # Inject system as a fake first user/assistant exchange
        if system_text:
            parts.append(f"[INST] {system_text}[/INST]")
            parts.append(f" OK, I will follow these instructions.</s>")

        for msg in loop_messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                parts.append(f"[INST] {content}[/INST]")
            elif role == "assistant":
                parts.append(f" {content}</s>")

        prompt = "<s>" + "".join(parts)
        return {"prompt": prompt, "stop": ["</s>"]}

    def clean_response(self, text: str) -> str:
        """Remove Mistral format markers from response."""
        cleaned = text
        for marker in ["[INST]", "[/INST]", "</s>", "<s>"]:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()
