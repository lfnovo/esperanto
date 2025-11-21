"""Central prompt rendering logic for BrioDocs."""

from __future__ import annotations

import os
from typing import Dict, List

from brio_ext.adapters import RenderedPrompt
from brio_ext.registry import get_adapter

# No default stop tokens - adapters provide model-specific stops, brio_ext handles fencing
DEFAULT_STOP: List[str] = []
TEMPLATE_PROVIDERS = {"openai", "anthropic", "grok", "ollama"}
PROMPT_PROVIDERS = {"llamacpp", "hf_local"}


def render_for_model(
    model_id: str,
    messages: List[Dict[str, str]],
    provider: str,
    chat_format: str = None,
) -> RenderedPrompt:
    """
    Render Esperanto messages into provider/model specific payloads.

    Args:
        model_id: The model identifier (e.g., "qwen2.5-7b-instruct")
        messages: List of message dictionaries with 'role' and 'content' keys
        provider: The provider name (e.g., "llamacpp", "openai")
        chat_format: Optional chat format hint (e.g., "chatml", "llama", "mistral-instruct")

    Returns:
        RenderedPrompt dictionary with 'prompt' or 'messages' key and 'stop' tokens
    """
    provider_key = (provider or "").lower()
    adapter = get_adapter(model_id, chat_format=chat_format)

    if os.getenv("BRIO_DEBUG"):
        print(f"[RENDERER] model_id={model_id}, provider={provider_key}")
        print(f"[RENDERER] adapter={adapter.__class__.__name__ if adapter else None}")

    if adapter:
        if provider_key in PROMPT_PROVIDERS:
            payload = dict(adapter.render(messages))
            payload["stop"] = _merge_stops(payload.get("stop"), DEFAULT_STOP)
            if os.getenv("BRIO_DEBUG"):
                print(f"[RENDERER] mode=PROMPT (completions)")
                print(f"[RENDERER] prompt_length={len(payload.get('prompt', ''))} chars")
                print(f"[RENDERER] stops={payload['stop']}")
                print(f"[RENDERER] Full prompt sent to model:")
                print("─" * 80)
                print(payload.get('prompt', ''))
                print("─" * 80)
            return payload

        rendered = adapter.render(messages)
        if "messages" in rendered:
            payload = {
                "messages": rendered["messages"],
                "stop": _merge_stops(rendered.get("stop"), DEFAULT_STOP),
            }
            if os.getenv("BRIO_DEBUG"):
                print(f"[RENDERER] mode=MESSAGES (chat)")
                print(f"[RENDERER] stops={payload['stop']}")
            return payload

        # If adapter produced a prompt but provider is not prompt-capable, fall through
        # to the default message behavior.

    if provider_key in TEMPLATE_PROVIDERS or adapter is None:
        if os.getenv("BRIO_DEBUG"):
            print(f"[RENDERER] mode=PASSTHROUGH (template provider or no adapter)")
            print(f"[RENDERER] stops={DEFAULT_STOP}")
        return {"messages": messages, "stop": DEFAULT_STOP.copy()}

    if adapter:
        payload = dict(adapter.render(messages))
        payload["stop"] = _merge_stops(payload.get("stop"), DEFAULT_STOP)
        if os.getenv("BRIO_DEBUG"):
            print(f"[RENDERER] mode=FALLBACK (adapter without prompt provider)")
            print(f"[RENDERER] stops={payload['stop']}")
        return payload

    return {"messages": messages, "stop": DEFAULT_STOP.copy()}


def _merge_stops(*stop_lists) -> List[str]:
    """Merge stop lists (dedupe, preserve order, guarantee default)."""
    merged: List[str] = []
    for stops in stop_lists:
        if not stops:
            continue
        for token in stops:
            if not token:
                continue
            if token not in merged:
                merged.append(token)

    if not merged:
        merged.extend(DEFAULT_STOP)
    return merged
