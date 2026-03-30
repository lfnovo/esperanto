"""Adapter registry for BrioDocs model families."""

from __future__ import annotations

from typing import Optional

from brio_ext.adapters import ChatAdapter
from brio_ext.adapters.gemma_adapter import GemmaAdapter
from brio_ext.adapters.llama_adapter import LlamaAdapter
from brio_ext.adapters.mistral_adapter import MistralAdapter
from brio_ext.adapters.phi_adapter import PhiAdapter
from brio_ext.adapters.qwen_adapter import QwenAdapter

ADAPTERS: tuple[ChatAdapter, ...] = (
    QwenAdapter(),
    LlamaAdapter(),
    MistralAdapter(),
    GemmaAdapter(),
    PhiAdapter(),
)


def get_adapter(model_id: str, chat_format: Optional[str] = None) -> Optional[ChatAdapter]:
    """
    Return the first adapter that can handle the supplied model identifier.

    Args:
        model_id: The model identifier (e.g., "qwen2.5-7b-instruct", "phi-4-mini-reasoning")
        chat_format: Optional chat format hint (e.g., "chatml", "llama", "mistral-instruct")
                    If provided, this takes precedence over model_id pattern matching

    Returns:
        The matching ChatAdapter, or None if no adapter matches
    """
    if not model_id:
        return None

    # Model-id matching takes priority — a phi model should use PhiAdapter even
    # if chat_format=chatml, not the more generic QwenAdapter.
    for adapter in ADAPTERS:
        if adapter.can_handle(model_id):
            return adapter

    # Fall back to format-based matching for models whose id doesn't match any
    # known pattern (e.g. a custom chatml model with an unfamiliar name).
    if chat_format:
        adapter = _get_adapter_by_format(chat_format)
        if adapter:
            return adapter

    return None


def _get_adapter_by_format(chat_format: str) -> Optional[ChatAdapter]:
    """
    Map chat format strings to adapters.

    This allows external systems (like BrioDocs) to specify the chat format
    explicitly when the model name doesn't follow standard patterns.
    """
    format_key = (chat_format or "").lower()

    # ChatML format -> Qwen or Phi adapter (Qwen is more general)
    if format_key in ("chatml", "chat-ml"):
        return QwenAdapter()

    # Llama format
    if format_key in ("llama", "llama3", "llama-3"):
        return LlamaAdapter()

    # Mistral format
    if format_key in ("mistral", "mistral-instruct"):
        return MistralAdapter()

    # Gemma format
    if format_key in ("gemma",):
        return GemmaAdapter()

    return None
