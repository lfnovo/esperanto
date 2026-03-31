"""Tests for BrioBaseChatModel: LangChain integration, streaming, and think-tag stripping."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    Message,
    StreamChoice,
    Usage,
)
from brio_ext.langchain_wrapper import BrioBaseChatModel


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


def _make_streaming_model(tokens: list[str]) -> MagicMock:
    """Create a mock model that returns streaming chunks."""
    chunks = _make_stream_chunks(tokens)
    model = MagicMock()
    model._brio_wrapped = True
    model._brio_provider = "test"
    model._brio_model_id = "test-model"
    model.model_name = "test-model"
    model.temperature = 0.7
    model.max_tokens = 512
    model.chat_complete = MagicMock(return_value=iter(chunks))
    return model


def _make_async_streaming_model(tokens: list[str]) -> MagicMock:
    """Create a mock model that returns async streaming chunks."""
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
    return model


class TestLangChainIntegration:
    """Verify BrioBaseChatModel integrates with LangChain properly."""

    def test_is_base_chat_model(self):
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


class TestStreaming:
    """Test _stream and _astream on BrioBaseChatModel."""

    def test_stream_yields_all_tokens(self):
        tokens = ["Hello", ", ", "world", "!"]
        model = _make_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        # Filter empty-content chunks: BaseChatModel.stream() may append a final
        # aggregated chunk with empty content and only response_metadata.
        result_tokens = [msg.content for msg in wrapper.stream("hello") if msg.content]

        assert result_tokens == tokens
        model.chat_complete.assert_called_once()
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
        model = _make_async_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        result_tokens = [msg.content async for msg in wrapper.astream("hello")]

        assert result_tokens == tokens


class TestStreamingThinkTagStripping:
    """Test that think tags are stripped during streaming."""

    def test_think_tags_stripped_from_stream(self):
        tokens = ["<think>", "reasoning here", "</think>", "The answer is 4"]
        model = _make_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        result = "".join(msg.content for msg in wrapper.stream("hello"))
        assert result == "The answer is 4"

    def test_think_tags_split_across_tokens(self):
        tokens = ["<thi", "nk>", "hidden", "</thi", "nk>", "visible"]
        model = _make_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        result = "".join(msg.content for msg in wrapper.stream("hello"))
        assert result == "visible"

    def test_multiple_think_blocks_stripped(self):
        tokens = [
            "<think>", "thought 1", "</think>",
            "answer 1 ",
            "<think>", "thought 2", "</think>",
            "answer 2",
        ]
        model = _make_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        result = "".join(msg.content for msg in wrapper.stream("hello"))
        assert result == "answer 1 answer 2"

    def test_no_think_tags_passes_through(self):
        tokens = ["Hello", ", ", "world", "!"]
        model = _make_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        result = "".join(msg.content for msg in wrapper.stream("hello"))
        assert result == "Hello, world!"

    def test_only_think_content_yields_empty(self):
        """When stream is only think content, no meaningful tokens are yielded."""
        tokens = ["<think>", "all thinking", "</think>"]
        model = _make_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        chunks = list(wrapper._stream([HumanMessage(content="hello")]))
        result = "".join(c.message.content for c in chunks)
        assert result == ""

    @pytest.mark.asyncio
    async def test_astream_think_tags_stripped(self):
        tokens = ["<think>", "reasoning", "</think>", "the answer"]
        model = _make_async_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        result_parts = []
        async for msg in wrapper.astream("hello"):
            result_parts.append(msg.content)
        assert "".join(result_parts) == "the answer"

    @pytest.mark.asyncio
    async def test_astream_split_think_tags_stripped(self):
        tokens = ["<thi", "nk>", "hidden", "</thi", "nk>", "visible"]
        model = _make_async_streaming_model(tokens)
        wrapper = BrioBaseChatModel(brio_model=model)
        result_parts = []
        async for msg in wrapper.astream("hello"):
            result_parts.append(msg.content)
        assert "".join(result_parts) == "visible"
