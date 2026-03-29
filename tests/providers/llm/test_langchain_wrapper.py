"""
Unit tests for BrioLangChainWrapper.

Covers two fixes:

1. _parse_fenced_content — unclosed <think> handling
   Thinking models (Qwen3/Qwen3.5, DeepSeek-R1) generate a <think>...</think>
   block before answering.  When the completion budget is exhausted mid-reasoning
   the closing </think> tag is never emitted.  The wrapper must strip the partial
   thinking block rather than returning it to the user.

2. no_think flag — /no_think injection
   When no_think=True the wrapper prepends "/no_think" to the first user message
   so that Qwen3/Qwen3.5 skips the reasoning block entirely.  For models without
   thinking mode the token is a harmless no-op.

Run with: pytest tests/providers/llm/test_langchain_wrapper.py -v
"""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wrapper(no_think: bool = False):
    """Return a BrioLangChainWrapper with a minimal mock model."""
    from brio_ext.langchain_wrapper import BrioLangChainWrapper

    mock_model = MagicMock()
    mock_model._brio_wrapped = True
    mock_model.model_name = "test-model"
    mock_model.temperature = 0.7
    mock_model.max_tokens = 512
    return BrioLangChainWrapper(mock_model, no_think=no_think)


# ---------------------------------------------------------------------------
# _parse_fenced_content — thinking content stripping
# ---------------------------------------------------------------------------

class TestParseFencedContentThinking:
    """_parse_fenced_content strips <think> blocks in all cases."""

    def test_closed_think_block_stripped(self):
        """Complete <think>...</think> block removed; real answer returned."""
        wrapper = _make_wrapper()
        raw = "<think>Let me reason about this...</think>Here is the answer."
        assert wrapper._parse_fenced_content(raw) == "Here is the answer."

    def test_unclosed_think_block_returns_empty(self):
        """Unclosed <think> (token limit hit mid-reasoning) returns empty string."""
        wrapper = _make_wrapper()
        raw = "<think>Reasoning step one...\nReasoning step two, still going..."
        result = wrapper._parse_fenced_content(raw)
        assert result == "", f"Expected empty string, got: {result!r}"

    def test_unclosed_think_does_not_leak_content(self):
        """<think> content is never returned to the user when unclosed."""
        wrapper = _make_wrapper()
        raw = "<think>SYSTEM PROMPT DETAILS: You are BRIO, a private assistant..."
        result = wrapper._parse_fenced_content(raw)
        assert "SYSTEM PROMPT DETAILS" not in result
        assert "<think>" not in result

    def test_content_before_unclosed_think_preserved(self):
        """Any real content that appears before an unclosed <think> is kept."""
        wrapper = _make_wrapper()
        raw = "Here is part of the answer.\n<think>Then I started reasoning..."
        result = wrapper._parse_fenced_content(raw)
        assert "Here is part of the answer." in result
        assert "<think>" not in result

    def test_fenced_with_closed_think_inside(self):
        """<think> inside <out> fencing is stripped; answer extracted cleanly."""
        wrapper = _make_wrapper()
        raw = "<out>\n<think>internal reasoning</think>\nActual answer here.\n</out>"
        assert wrapper._parse_fenced_content(raw) == "Actual answer here."

    def test_fenced_with_unclosed_think_returns_empty(self):
        """<out> content that is only an unclosed <think> block returns empty."""
        wrapper = _make_wrapper()
        raw = "<out>\n<think>Reasoning that never ends...</out>"
        result = wrapper._parse_fenced_content(raw)
        assert result == ""

    def test_no_think_tags_passes_through(self):
        """Content without any <think> tags is returned unchanged."""
        wrapper = _make_wrapper()
        raw = "Plain answer with no thinking tags."
        assert wrapper._parse_fenced_content(raw) == raw

    def test_multiple_closed_think_blocks_all_stripped(self):
        """Multiple complete <think> blocks are all removed."""
        wrapper = _make_wrapper()
        raw = "<think>first thought</think>Answer start <think>second thought</think> answer end."
        result = wrapper._parse_fenced_content(raw)
        assert "<think>" not in result
        assert "Answer start" in result
        assert "answer end." in result


# ---------------------------------------------------------------------------
# no_think flag — /no_think injection in _convert_messages
# ---------------------------------------------------------------------------

class TestNoThinkInjection:
    """no_think=True prepends /no_think to the first user message only."""

    def _make_langchain_msg(self, role: str, content: str):
        msg = MagicMock()
        msg.type = role
        msg.content = content
        return msg

    def test_no_think_prepended_to_first_user_message(self):
        wrapper = _make_wrapper(no_think=True)
        msgs = [self._make_langchain_msg("human", "What is a patent?")]
        converted = wrapper._convert_messages(msgs)
        assert converted[0]["content"].startswith("/no_think\n")

    def test_no_think_only_on_first_user_message(self):
        """Second user message must NOT get the /no_think prefix."""
        wrapper = _make_wrapper(no_think=True)
        msgs = [
            self._make_langchain_msg("human", "First question"),
            self._make_langchain_msg("ai", "Some answer"),
            self._make_langchain_msg("human", "Follow-up question"),
        ]
        converted = wrapper._convert_messages(msgs)
        assert converted[0]["content"].startswith("/no_think\n")
        assert not converted[2]["content"].startswith("/no_think")

    def test_no_think_false_does_not_modify_messages(self):
        """no_think=False leaves messages untouched."""
        wrapper = _make_wrapper(no_think=False)
        msgs = [self._make_langchain_msg("human", "What is a patent?")]
        converted = wrapper._convert_messages(msgs)
        assert not converted[0]["content"].startswith("/no_think")
        assert converted[0]["content"] == "What is a patent?"

    def test_no_think_skips_system_message(self):
        """System message is not prefixed; only the first user message gets it."""
        wrapper = _make_wrapper(no_think=True)
        msgs = [
            self._make_langchain_msg("system", "You are a helpful assistant."),
            self._make_langchain_msg("human", "Tell me about patents."),
        ]
        converted = wrapper._convert_messages(msgs)
        assert not converted[0]["content"].startswith("/no_think")
        assert converted[1]["content"].startswith("/no_think\n")

    def test_no_think_original_content_preserved(self):
        """The original user message content follows the /no_think prefix."""
        wrapper = _make_wrapper(no_think=True)
        question = "Do they hear patent cases?"
        msgs = [self._make_langchain_msg("human", question)]
        converted = wrapper._convert_messages(msgs)
        assert converted[0]["content"] == f"/no_think\n{question}"
