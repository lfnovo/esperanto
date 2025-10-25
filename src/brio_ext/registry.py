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


def get_adapter(model_id: str) -> Optional[ChatAdapter]:
    """Return the first adapter that can handle the supplied model identifier."""
    if not model_id:
        return None

    for adapter in ADAPTERS:
        if adapter.can_handle(model_id):
            return adapter
    return None
