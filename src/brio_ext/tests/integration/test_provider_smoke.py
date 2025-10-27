"""Optional smoke tests that hit real providers through BrioAIFactory.

Set BRIO_TEST_<PROVIDER>_MODEL environment variables (and optional overrides
such as BRIO_TEST_<PROVIDER>_CONFIG) before running:

    BRIO_TEST_OPENAI_MODEL=gpt-4o-mini \
    BRIO_TEST_GROQ_MODEL=groq/llama3-8b-8192-tool-use-preview \
    pytest src/brio_ext/tests/integration/test_provider_smoke.py -q -m integration

llama.cpp remains opt-in via BRIO_TEST_LLAMACPP_MODEL and BRIO_TEST_LLAMACPP_BASE_URL.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Optional

import pytest

from brio_ext.factory import BrioAIFactory

DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a senior legal editor. "
            "Return ONLY the rewritten text wrapped in <out>...</out>. "
            "Do not include explanations."
        ),
    },
    {
        "role": "user",
        "content": (
            "TASK: Fix grammar while preserving meaning and defined terms.\n"
            "TEXT:\n<<<\nThe party hereby agree that the term is two year.\n>>>"
        ),
    },
]


def _call_chat(provider: str, model_id: str, config: Optional[Dict] = None):
    model = BrioAIFactory.create_language(provider=provider, model_name=model_id, config=config or {})
    return model.chat_complete(DEFAULT_MESSAGES)


def _assert_fenced_content(response):
    assert response.choices, "No choices returned from provider"
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    assert content.startswith("<out>"), f"response missing <out> fence: {content!r}"
    assert content.endswith("</out>"), f"response missing </out> close: {content!r}"

    body = content[len("<out>") : -len("</out>")].strip()
    assert body, "response body inside <out>...</out> was empty"
    return body, choice


def _build_provider_config(prefix: str, defaults: Optional[Dict] = None) -> Dict:
    config: Dict = dict(defaults or {})

    max_tokens = os.getenv(f"BRIO_TEST_{prefix}_MAX_TOKENS", os.getenv("BRIO_TEST_DEFAULT_MAX_TOKENS", "512"))
    temperature = os.getenv(f"BRIO_TEST_{prefix}_TEMPERATURE", os.getenv("BRIO_TEST_DEFAULT_TEMPERATURE", "0.25"))
    top_p = os.getenv(f"BRIO_TEST_{prefix}_TOP_P", os.getenv("BRIO_TEST_DEFAULT_TOP_P", "0.8"))

    if max_tokens:
        config.setdefault("max_tokens", int(max_tokens))
    if temperature:
        config.setdefault("temperature", float(temperature))
    if top_p:
        config.setdefault("top_p", float(top_p))

    base_url = os.getenv(f"BRIO_TEST_{prefix}_BASE_URL")
    if base_url:
        config.setdefault("base_url", base_url)

    config_override = os.getenv(f"BRIO_TEST_{prefix}_CONFIG")
    if config_override:
        config.update(json.loads(config_override))

    return config


CLOUD_PROVIDERS = [
    {"prefix": "OPENAI", "default_provider": "openai"},
    {"prefix": "ANTHROPIC", "default_provider": "anthropic"},
    {"prefix": "GROK", "default_provider": "grok"},
    {"prefix": "GROQ", "default_provider": "groq"},
    {"prefix": "XAI", "default_provider": "xai"},
    {"prefix": "PERPLEXITY", "default_provider": "perplexity"},
    {"prefix": "MISTRAL", "default_provider": "mistral"},
    {"prefix": "DEEPSEEK", "default_provider": "deepseek"},
    {"prefix": "GOOGLE", "default_provider": "google"},
    {"prefix": "VERTEX", "default_provider": "vertex"},
    {"prefix": "AZURE", "default_provider": "azure"},
    {"prefix": "OPENROUTER", "default_provider": "openrouter"},
    {"prefix": "OPENAI_COMPATIBLE", "default_provider": "openai-compatible"},
]


@pytest.mark.integration
@pytest.mark.parametrize("entry", CLOUD_PROVIDERS, ids=lambda e: e["prefix"].lower())
def test_cloud_provider_smoke(entry):
    prefix = entry["prefix"]
    model_env = f"BRIO_TEST_{prefix}_MODEL"
    model_id = os.getenv(model_env)
    if not model_id:
        pytest.skip(f"{model_env} not set")

    provider = os.getenv(f"BRIO_TEST_{prefix}_PROVIDER", entry["default_provider"])
    config = _build_provider_config(prefix)

    response = _call_chat(provider, model_id, config=config)
    body, choice = _assert_fenced_content(response)

    assert len(body.split()) >= 3, "response body too short to be useful"
    assert choice.finish_reason in {"stop", "length", None}


@pytest.mark.integration
def test_llamacpp_chat_smoke():
    model_id = os.getenv("BRIO_TEST_LLAMACPP_MODEL")
    if not model_id:
        pytest.skip("BRIO_TEST_LLAMACPP_MODEL not set")

    base_url = os.getenv("BRIO_TEST_LLAMACPP_BASE_URL", "http://127.0.0.1:8765")
    config = {
        "base_url": base_url,
        "max_tokens": int(os.getenv("BRIO_TEST_LLAMACPP_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("BRIO_TEST_LLAMACPP_TEMPERATURE", "0.25")),
        "top_p": float(os.getenv("BRIO_TEST_LLAMACPP_TOP_P", "0.8")),
    }

    response = _call_chat("llamacpp", model_id, config=config)
    body, choice = _assert_fenced_content(response)

    assert len(body.split()) >= 3, "response body too short to be useful"
    assert choice.finish_reason in {"stop", "length", None}
