"""Gemma 4 family prompt adapter.

Mirrors the chat template shipped with `google/gemma-4-*` models, which uses
turn markers `<|turn>...<turn|>` and a thinking channel `<|channel>...<channel|>`.
Tools and multimodal content are out of scope (BrioDocs does not use them with
this adapter), so this implementation handles only system / user / assistant
text turns.
"""

from __future__ import annotations

import re
from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt

_TURN_OPEN = "<|turn>"
_TURN_CLOSE = "<turn|>"
_CHANNEL_OPEN = "<|channel>"
_CHANNEL_CLOSE = "<channel|>"
_THINK = "<|think|>"

_THINKING_BLOCK_RE = re.compile(
    re.escape(_CHANNEL_OPEN) + r".*?" + re.escape(_CHANNEL_CLOSE),
    re.DOTALL,
)
_TURN_HEADER_RE = re.compile(
    re.escape(_TURN_OPEN) + r"(?:user|model|system|developer)\n?"
)


class Gemma4Adapter(ChatAdapter):
    """Render Esperanto messages into Gemma 4 turn-based prompts."""

    def can_handle(self, model_id: str) -> bool:
        lower = (model_id or "").lower()
        # Match only Gemma 4 — earlier and future Gemma generations use
        # different chat templates and need their own adapters.
        return "gemma-4" in lower or "gemma4" in lower

    def render(self, messages: List[Dict[str, str]], no_think: bool = False) -> RenderedPrompt:
        thinking_enabled = not no_think
        parts: List[str] = []

        first_is_system = bool(messages) and messages[0].get("role") in ("system", "developer")

        # Template opens a system turn whenever thinking is enabled or the first
        # message is system/developer (chat_template.jinja line 179).
        if thinking_enabled or first_is_system:
            parts.append(f"{_TURN_OPEN}system\n")
            if thinking_enabled:
                parts.append(f"{_THINK}\n")
            if first_is_system:
                parts.append((messages[0].get("content") or "").strip())
            parts.append(f"{_TURN_CLOSE}\n")

        loop_messages = messages[1:] if first_is_system else messages
        for msg in loop_messages:
            role = msg.get("role")
            if role == "tool":
                continue
            rendered_role = "model" if role == "assistant" else role
            content = msg.get("content") or ""
            if rendered_role == "model":
                content = self._strip_thinking_block(content)
            content = content.strip()

            parts.append(f"{_TURN_OPEN}{rendered_role}\n")
            parts.append(content)
            parts.append(f"{_TURN_CLOSE}\n")

        parts.append(f"{_TURN_OPEN}model\n")
        if not thinking_enabled:
            # Empty thought channel signals "skip reasoning, answer directly"
            # (chat_template.jinja line 343).
            parts.append(f"{_CHANNEL_OPEN}thought\n{_CHANNEL_CLOSE}")

        return {"prompt": "".join(parts), "stop": [_TURN_CLOSE]}

    def clean_response(self, text: str) -> str:
        cleaned = _THINKING_BLOCK_RE.sub("", text)
        if _CHANNEL_OPEN in cleaned:
            cleaned = cleaned.split(_CHANNEL_OPEN)[0]
        cleaned = _TURN_HEADER_RE.sub("", cleaned)
        for marker in (_TURN_CLOSE, _CHANNEL_CLOSE, _THINK):
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()

    @staticmethod
    def _strip_thinking_block(text: str) -> str:
        result = _THINKING_BLOCK_RE.sub("", text)
        if _CHANNEL_OPEN in result:
            result = result.split(_CHANNEL_OPEN)[0]
        return result
