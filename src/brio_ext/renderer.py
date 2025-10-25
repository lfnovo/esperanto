"""Central prompt rendering logic for BrioDocs."""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import RenderedPrompt
from brio_ext.registry import get_adapter

DEFAULT_STOP = ["</out>"]
TEMPLATE_PROVIDERS = {"openai", "anthropic", "grok", "ollama"}
PROMPT_PROVIDERS = {"llamacpp", "hf_local"}


def render_for_model(
    model_id: str,
    messages: List[Dict[str, str]],
    provider: str,
) -> RenderedPrompt:
    """Render Esperanto messages into provider/model specific payloads."""
    provider_key = (provider or "").lower()
    adapter = get_adapter(model_id)

    if adapter:
        if provider_key in PROMPT_PROVIDERS:
            payload = dict(adapter.render(messages))
            payload["stop"] = _merge_stops(payload.get("stop"), DEFAULT_STOP)
            return payload

        rendered = adapter.render(messages)
        if "messages" in rendered:
            payload = {
                "messages": rendered["messages"],
                "stop": _merge_stops(rendered.get("stop"), DEFAULT_STOP),
            }
            return payload

        # If adapter produced a prompt but provider is not prompt-capable, fall through
        # to the default message behavior.

    if provider_key in TEMPLATE_PROVIDERS or adapter is None:
        return {"messages": messages, "stop": DEFAULT_STOP.copy()}

    if adapter:
        payload = dict(adapter.render(messages))
        payload["stop"] = _merge_stops(payload.get("stop"), DEFAULT_STOP)
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
