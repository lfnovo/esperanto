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
    assert prompt.endswith("<out>\n")
    assert "</out>" in rendered["stop"]
    assert "<|im_end|>" in rendered["stop"]


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


@pytest.mark.parametrize(
    "model_id,provider",
    [
        ("phi-4-mini", "openai"),
        ("phi-4-mini", "hf_local"),
    ],
)
def test_phi_adapter_keeps_messages(model_id, provider):
    rendered = render_for_model(model_id, copy.deepcopy(MESSAGES), provider)
    assert rendered["messages"] == MESSAGES
    assert "</out>" in rendered["stop"]
