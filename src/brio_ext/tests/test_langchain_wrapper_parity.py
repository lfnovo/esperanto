"""Test that BrioBaseChatModel and BrioLangChainWrapper produce equivalent results."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    Message,
    StreamChoice,
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


class TestNewWrapperIsBaseChatModel:
    """Verify the new wrapper integrates with LangChain properly."""

    def test_is_base_chat_model(self):
        from langchain_core.language_models.chat_models import BaseChatModel

        response = _make_fake_response()
        model = _make_brio_model(response)
        wrapper = BrioBaseChatModel(brio_model=model)
        assert isinstance(wrapper, BaseChatModel)

    def test_llm_type(self):
        response = _make_fake_response()
        model = _make_brio_model(response)
        wrapper = BrioBaseChatModel(brio_model=model)
        assert wrapper._llm_type == "brio_langchain_wrapper"

    def test_returns_langchain_ai_message(self):
        from langchain_core.messages import AIMessage

        response = _make_fake_response("test content")
        model = _make_brio_model(response)
        wrapper = BrioBaseChatModel(brio_model=model)
        result = wrapper.invoke("hello")
        assert isinstance(result, AIMessage)

    def test_response_metadata_preserved(self):
        response = _make_fake_response("test content")
        model = _make_brio_model(response)
        wrapper = BrioBaseChatModel(brio_model=model)
        result = wrapper.invoke("hello")
        assert result.response_metadata["model"] == "test-model"
        assert result.response_metadata["finish_reason"] == "stop"
        assert result.response_metadata["usage"]["total_tokens"] == 15

    def test_rejects_unwrapped_model(self):
        model = MagicMock()
        model._brio_wrapped = False
        with pytest.raises(ValueError, match="BrioAIFactory"):
            BrioBaseChatModel(brio_model=model)

    def test_message_conversion_from_langchain_types(self):
        from langchain_core.messages import HumanMessage, SystemMessage

        response = _make_fake_response("reply")
        model = _make_brio_model(response)
        wrapper = BrioBaseChatModel(brio_model=model)

        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hi"),
        ]
        wrapper.invoke(messages)

        call_args = model.chat_complete.call_args[0][0]
        assert call_args[0] == {"role": "system", "content": "You are helpful."}
        assert call_args[1] == {"role": "user", "content": "Hi"}


def _make_stream_chunks(tokens: list[str]) -> list[ChatCompletionChunk]:
    """Create fake ChatCompletionChunk objects for streaming tests."""
    return [
        ChatCompletionChunk(
            id=f"chunk-{i}",
            object="chat.completion.chunk",
            created=1700000000,
            model="test-model",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=token),
                    finish_reason=None if i < len(tokens) - 1 else "stop",
                )
            ],
        )
        for i, token in enumerate(tokens)
    ]


class TestStreaming:
    """Test _stream and _astream on BrioBaseChatModel."""

    def test_stream_yields_all_tokens(self):
        tokens = ["Hello", ", ", "world", "!"]
        chunks = _make_stream_chunks(tokens)

        model = MagicMock()
        model._brio_wrapped = True
        model._brio_provider = "test"
        model._brio_model_id = "test-model"
        model.model_name = "test-model"
        model.temperature = 0.7
        model.max_tokens = 512
        model.chat_complete = MagicMock(return_value=iter(chunks))

        wrapper = BrioBaseChatModel(brio_model=model)
        # BaseChatModel.stream() yields AIMessageChunk objects, not ChatGenerationChunk
        result_tokens = [msg.content for msg in wrapper.stream("hello")]

        assert result_tokens == tokens
        model.chat_complete.assert_called_once()
        # Verify stream=True was passed
        assert model.chat_complete.call_args[1]["stream"] is True

    def test_stream_concatenated_matches_invoke(self):
        """Streaming tokens concatenated should equal invoke result (for plain content)."""
        content = "Hello, world!"
        tokens = ["Hello", ", ", "world", "!"]
        chunks = _make_stream_chunks(tokens)

        response = _make_fake_response(content)
        model = _make_brio_model(response)
        original_chat = model.chat_complete

        def smart_chat_complete(messages, stream=None):
            if stream:
                return iter(chunks)
            return original_chat(messages, stream=stream)

        model.chat_complete = MagicMock(side_effect=smart_chat_complete)

        wrapper = BrioBaseChatModel(brio_model=model)

        invoke_result = wrapper.invoke("hello")
        stream_result = "".join(msg.content for msg in wrapper.stream("hello"))

        assert stream_result == invoke_result.content

    @pytest.mark.asyncio
    async def test_astream_yields_all_tokens(self):
        tokens = ["Hello", ", ", "world", "!"]
        chunks = _make_stream_chunks(tokens)

        async def async_chunk_iter():
            for chunk in chunks:
                yield chunk

        model = MagicMock()
        model._brio_wrapped = True
        model._brio_provider = "test"
        model._brio_model_id = "test-model"
        model.model_name = "test-model"
        model.temperature = 0.7
        model.max_tokens = 512
        model.achat_complete = AsyncMock(return_value=async_chunk_iter())

        wrapper = BrioBaseChatModel(brio_model=model)
        result_tokens = [msg.content async for msg in wrapper.astream("hello")]

        assert result_tokens == tokens
