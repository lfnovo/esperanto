"""Integration tests for Brio factory wrapping helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from brio_ext.factory import DEFAULT_STOP, _wrap_language_model
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
    assert call["prompt"].endswith("<out>\n")
    assert "</out>" in call["stop"]
    assert "<|im_end|>" in call["stop"]
    assert call["stop_snapshot"] == ["</out>", "<|im_end|>"]
    assert model._config["stop"] == ["<EOF>"]
    assert result.content == "<out>\nPolished clause.\n</out>"


def test_wrap_language_model_is_idempotent():
    model = DummyLanguageModel(model_name="dummy", config={})
    first = _wrap_language_model(model, model_id="mistral-7b", provider="llamacpp")
    second = _wrap_language_model(model, model_id="mistral-7b", provider="llamacpp")

    assert first is second
    assert getattr(model, "_brio_wrapped")
