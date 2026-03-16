"""Test that BrioBaseChatModel and BrioLangChainWrapper produce equivalent results."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from esperanto.common_types import (
    ChatCompletion,
    Choice,
    Message,
    Usage,
)
from brio_ext.langchain_wrapper import BrioBaseChatModel, BrioLangChainWrapper


def _make_fake_response(content: str = "Hello, world!") -> ChatCompletion:
    """Create a fake ChatCompletion response."""
    return ChatCompletion(
        id="test-123",
        object="chat.completion",
        created=1700000000,
        model="test-model",
        provider="test",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_brio_model(fake_response: ChatCompletion) -> MagicMock:
    """Create a mock brio_model with _brio_wrapped=True."""
    model = MagicMock()
    model._brio_wrapped = True
    model._brio_provider = "test"
    model._brio_model_id = "test-model"
    model.model_name = "test-model"
    model.temperature = 0.7
    model.max_tokens = 512
    model.chat_complete = MagicMock(return_value=fake_response)
    model.achat_complete = AsyncMock(return_value=fake_response)
    return model


class TestInvokeParity:
    """Both wrappers should return the same content for the same input."""

    @pytest.mark.parametrize(
        "content",
        [
            "Hello, world!",
            "<out>extracted content</out>",
            "<output>extracted output content</output>",
            "<think>thinking here</think>actual content",
            "<think>only thinking</think>",
            '<think>{"key": "value"}</think>',
            "plain text no tags",
        ],
        ids=[
            "plain",
            "out_fenced",
            "output_fenced",
            "think_with_content",
            "think_only",
            "think_json",
            "no_tags",
        ],
    )
    def test_invoke_content_matches(self, content):
        response = _make_fake_response(content)
        model = _make_brio_model(response)

        legacy = BrioLangChainWrapper(model)
        new = BrioBaseChatModel(brio_model=model)

        legacy_result = legacy.invoke("test prompt")
        new_result = new.invoke("test prompt")

        assert legacy_result.content == new_result.content, (
            f"Content mismatch for input {content!r}:\n"
            f"  legacy: {legacy_result.content!r}\n"
            f"  new:    {new_result.content!r}"
        )

    @pytest.mark.parametrize(
        "content",
        [
            "Hello, world!",
            "<out>extracted content</out>",
            "<output>extracted output content</output>",
            "<think>thinking here</think>actual content",
            "<think>only thinking</think>",
            '<think>{"key": "value"}</think>',
            "plain text no tags",
        ],
        ids=[
            "plain",
            "out_fenced",
            "output_fenced",
            "think_with_content",
            "think_only",
            "think_json",
            "no_tags",
        ],
    )
    @pytest.mark.asyncio
    async def test_ainvoke_content_matches(self, content):
        response = _make_fake_response(content)
        model = _make_brio_model(response)

        legacy = BrioLangChainWrapper(model)
        new = BrioBaseChatModel(brio_model=model)

        legacy_result = await legacy.ainvoke("test prompt")
        new_result = await new.ainvoke("test prompt")

        assert legacy_result.content == new_result.content, (
            f"Content mismatch for input {content!r}:\n"
            f"  legacy: {legacy_result.content!r}\n"
            f"  new:    {new_result.content!r}"
        )
