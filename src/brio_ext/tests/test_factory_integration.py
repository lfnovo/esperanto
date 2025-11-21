"""Integration tests for Brio factory wrapping helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from brio_ext.factory import BrioAIFactory, DEFAULT_STOP, _wrap_language_model
from esperanto.common_types import ChatCompletion, Choice, Message
from esperanto.providers.llm.base import LanguageModel

MESSAGES = [
    {"role": "system", "content": "System guidance."},
    {"role": "user", "content": "Please revise the clause."},
]


class DummyLanguageModel(LanguageModel):
    """Minimal LanguageModel implementation for testing."""

    def __post_init__(self):
        super().__post_init__()
        self.chat_calls: List[Dict[str, Any]] = []
        self.prompt_calls: List[Dict[str, Any]] = []

    @property
    def models(self):
        return []

    def chat_complete(self, messages: List[Dict[str, str]], stream: Optional[bool] = None):
        self.chat_calls.append(
            {
                "messages": messages,
                "stream": stream,
                "stop_snapshot": list(self._config.get("stop") or []),
            }
        )
        return {"messages": messages}

    async def achat_complete(self, messages, stream: Optional[bool] = None):
        return self.chat_complete(messages, stream=stream)

    def prompt_complete(self, prompt: str, stop=None, stream: Optional[bool] = None):
        self.prompt_calls.append(
            {
                "prompt": prompt,
                "stop": list(stop or []),
                "stream": stream,
                "stop_snapshot": list(self._config.get("stop") or []),
            }
        )
        response = ChatCompletion(
            id="dummy",
            model=self.get_model_name(),
            provider=self.provider,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Polished clause."),
                    finish_reason="stop",
                )
            ],
        )
        return response

    async def aprompt_complete(self, prompt: str, stop=None, stream: Optional[bool] = None):
        return self.prompt_complete(prompt, stop=stop, stream=stream)

    def _get_default_model(self) -> str:
        return "dummy"

    @property
    def provider(self) -> str:
        return "dummy"

    def to_langchain(self) -> Any:
        raise NotImplementedError


def test_wrap_language_model_preserves_existing_stop():
    model = DummyLanguageModel(model_name="dummy", config={"stop": ["<EOF>"]})
    wrapped = _wrap_language_model(model, model_id="gpt-4o-mini", provider="openai")

    assert wrapped is model
    assert getattr(model, "_brio_wrapped")

    result = wrapped.chat_complete(MESSAGES)
    assert result == {"messages": MESSAGES}
    assert len(model.chat_calls) == 1
    call = model.chat_calls[0]

    assert call["messages"] == MESSAGES
    assert call["stop_snapshot"] == DEFAULT_STOP
    assert model._config["stop"] == ["<EOF>"]


def test_wrap_language_model_uses_prompt_handler_for_local_engines():
    model = DummyLanguageModel(model_name="dummy", config={"stop": ["<EOF>"]})
    wrapped = _wrap_language_model(model, model_id="qwen2.5-7b-instruct", provider="llamacpp")

    result = wrapped.chat_complete(MESSAGES)
    assert not model.chat_calls
    assert len(model.prompt_calls) == 1

    call = model.prompt_calls[0]
    assert call["prompt"].startswith("<|im_start|>system")
    assert call["prompt"].endswith("<|im_start|>assistant\n")
    assert "<|im_end|>" in call["stop"]
    assert call["stop_snapshot"] == ["<|im_end|>"]
    assert model._config["stop"] == ["<EOF>"]
    assert result.content == "<out>\nPolished clause.\n</out>"


def test_wrap_language_model_is_idempotent():
    model = DummyLanguageModel(model_name="dummy", config={})
    first = _wrap_language_model(model, model_id="mistral-7b", provider="llamacpp")
    second = _wrap_language_model(model, model_id="mistral-7b", provider="llamacpp")

    assert first is second
    assert getattr(model, "_brio_wrapped")


def test_wrap_language_model_with_chat_format_hint():
    """Test that chat_format parameter correctly selects adapter for unknown model names."""
    model = DummyLanguageModel(model_name="dummy", config={})
    # Use a custom model name that wouldn't normally match any adapter
    wrapped = _wrap_language_model(
        model,
        model_id="phi-4-mini-reasoning",  # Custom name with no standard pattern
        provider="llamacpp",
        chat_format="chatml"  # Hint to use ChatML format (Qwen/Phi adapter)
    )

    result = wrapped.chat_complete(MESSAGES)

    # Should use prompt_complete (not chat_complete) for llamacpp provider
    assert not model.chat_calls
    assert len(model.prompt_calls) == 1

    call = model.prompt_calls[0]
    # ChatML format should be used (from QwenAdapter which handles chatml)
    assert call["prompt"].startswith("<|im_start|>system")
    assert call["prompt"].endswith("<|im_start|>assistant\n")
    assert "<|im_end|>" in call["stop"]


def test_wrap_language_model_chat_format_llama():
    """Test that chat_format='llama' correctly selects Llama adapter."""
    model = DummyLanguageModel(model_name="dummy", config={})
    wrapped = _wrap_language_model(
        model,
        model_id="custom-llama-model",  # Custom name
        provider="llamacpp",
        chat_format="llama"  # Hint to use Llama format
    )

    result = wrapped.chat_complete(MESSAGES)

    # Llama adapter returns messages (not prompt), so it uses chat_complete path
    # with native chat template from llamacpp server
    assert len(model.chat_calls) == 1
    assert not model.prompt_calls

    call = model.chat_calls[0]
    # Messages should be passed through for native chat template rendering
    assert call["messages"] == MESSAGES
    assert "<|eot_id|>" in call["stop_snapshot"] or "<|end_of_text|>" in call["stop_snapshot"]


def test_wrap_language_model_chat_format_mistral():
    """Test that chat_format='mistral' correctly selects Mistral adapter."""
    model = DummyLanguageModel(model_name="dummy", config={})
    wrapped = _wrap_language_model(
        model,
        model_id="custom-mistral-model",  # Custom name
        provider="llamacpp",
        chat_format="mistral-instruct"  # Hint to use Mistral format
    )

    result = wrapped.chat_complete(MESSAGES)

    assert not model.chat_calls
    assert len(model.prompt_calls) == 1

    call = model.prompt_calls[0]
    # Mistral format should be used
    assert call["prompt"].startswith("[INST]")
    assert call["prompt"].endswith("[/INST]")


def test_brio_factory_extracts_chat_format_from_config():
    """Test that BrioAIFactory.create_language() extracts chat_format from config."""
    from unittest.mock import Mock, patch

    # Mock the parent factory's create_language to return our dummy model
    with patch.object(
        BrioAIFactory.__bases__[0],  # AIFactory
        'create_language',
        return_value=DummyLanguageModel(model_name="test-model", config={})
    ):
        # Create a model with chat_format in config
        model = BrioAIFactory.create_language(
            provider="llamacpp",
            model_name="phi-4-mini-reasoning",
            config={
                "base_url": "http://127.0.0.1:8765/v1",
                "api_key": "not-needed",
                "chat_format": "chatml"
            }
        )

        # Verify that chat_format was stored on the wrapped model
        assert hasattr(model, "_brio_wrapped")
        assert getattr(model, "_brio_chat_format") == "chatml"

        # Verify that the model uses ChatML format when rendering
        result = model.chat_complete(MESSAGES)

        # Should use prompt_complete (not chat_complete) for llamacpp provider
        assert not model.chat_calls
        assert len(model.prompt_calls) == 1

        call = model.prompt_calls[0]
        # ChatML format should be used
        assert call["prompt"].startswith("<|im_start|>system")
        assert "<|im_end|>" in call["stop"]
