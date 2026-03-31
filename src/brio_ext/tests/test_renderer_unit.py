"""Unit tests for render_for_model helper."""

import copy

import pytest

from brio_ext.renderer import DEFAULT_STOP, render_for_model

MESSAGES = [
    {"role": "system", "content": "Follow rules."},
    {"role": "user", "content": "Edit the paragraph."},
]


def test_template_provider_without_adapter_passes_through_messages():
    rendered = render_for_model("gpt-4o-mini", copy.deepcopy(MESSAGES), "openai")
    assert rendered["messages"] == MESSAGES
    assert rendered["stop"] == DEFAULT_STOP
    assert "prompt" not in rendered


def test_qwen_with_llamacpp_returns_prompt():
    rendered = render_for_model("qwen2.5-7b-instruct", copy.deepcopy(MESSAGES), "llamacpp")
    prompt = rendered["prompt"]

    assert prompt.startswith("<|im_start|>system")
    # Ends with open assistant turn (no <out> fencing — handled at factory level)
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "<out>" not in prompt
    # ChatML stop token; </out> is NOT a stop token (fencing at factory level)
    assert "<|im_end|>" in rendered["stop"]
    assert "</out>" not in rendered["stop"]


def test_unknown_model_with_llamacpp_falls_back_to_messages():
    rendered = render_for_model("custom-model", copy.deepcopy(MESSAGES), "llamacpp")
    assert rendered["messages"] == MESSAGES
    assert rendered["stop"] == DEFAULT_STOP
    assert "prompt" not in rendered


def test_qwen_with_template_provider_prefers_messages():
    rendered = render_for_model("qwen2.5-7b-instruct", copy.deepcopy(MESSAGES), "openai")
    assert rendered["messages"] == MESSAGES
    assert rendered["stop"] == DEFAULT_STOP
    assert "prompt" not in rendered


def test_phi_with_template_provider_passes_through_messages():
    """With a template provider (openai/anthropic), PhiAdapter prompt is ignored and messages pass through."""
    rendered = render_for_model("phi-4-mini", copy.deepcopy(MESSAGES), "openai")
    assert rendered["messages"] == MESSAGES
    # No fencing stop tokens — factory-level fencing, not renderer-level
    assert "</out>" not in rendered["stop"]


def test_phi_with_local_provider_returns_chatml_prompt():
    """With hf_local, PhiAdapter builds a ChatML prompt for raw completion."""
    rendered = render_for_model("phi-4-mini", copy.deepcopy(MESSAGES), "hf_local")
    assert "prompt" in rendered
    assert rendered["prompt"].startswith("<|im_start|>")
    assert rendered["prompt"].endswith("<|im_start|>assistant\n")
    # ChatML stop tokens; </out> is NOT a stop token (fencing at factory level)
    assert "<|im_end|>" in rendered["stop"]
    assert "</out>" not in rendered["stop"]
