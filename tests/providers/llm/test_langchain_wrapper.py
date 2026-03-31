"""
Unit tests for BrioLangChainWrapper.

Covers three areas:

1. _parse_fenced_content (standalone) — fencing extraction + <think> stripping
   Callable directly as ``from brio_ext.langchain_wrapper import _parse_fenced_content``.
   Tests verify it handles <out>/<output> fencing and complete/unclosed <think> blocks.

2. no_think flag — /no_think injection
   When no_think=True the wrapper prepends "/no_think" to the first user message
   so that Qwen3/Qwen3.5 skips the reasoning block entirely.

3. StreamingFenceFilter — streaming-path <out>/<output> extraction
   Verifies that fencing in a stream is correctly stripped token-by-token.

Run with: pytest tests/providers/llm/test_langchain_wrapper.py -v
"""

from unittest.mock import MagicMock

import pytest

from brio_ext.langchain_wrapper import _parse_fenced_content
from esperanto.utils.streaming import StreamingFenceFilter


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


def _run_fence_filter(tokens):
    """Helper: feed tokens through StreamingFenceFilter, return collected output."""
    f = StreamingFenceFilter()
    parts = []
    for t in tokens:
        out = f.process(t)
        if out:
            parts.append(out)
    remaining = f.flush()
    if remaining:
        parts.append(remaining)
    return "".join(parts)


# ---------------------------------------------------------------------------
# _parse_fenced_content — standalone function tests
# ---------------------------------------------------------------------------

class TestParseFencedContent:
    """_parse_fenced_content is a module-level function; test it directly."""

    def test_plain_text_passes_through(self):
        raw = "Plain answer with no thinking tags."
        assert _parse_fenced_content(raw) == raw

    def test_out_fencing_extracted(self):
        raw = "<out>The actual answer.</out>"
        assert _parse_fenced_content(raw) == "The actual answer."

    def test_output_fencing_extracted(self):
        raw = "<output>The actual answer.</output>"
        assert _parse_fenced_content(raw) == "The actual answer."

    def test_out_fencing_with_leading_whitespace(self):
        raw = "<out>\n  Indented answer.\n</out>"
        assert _parse_fenced_content(raw) == "Indented answer."

    def test_no_fencing_when_only_open_tag(self):
        """Only an open tag with no close — treated as plain text."""
        raw = "<out>Some content without a close tag"
        result = _parse_fenced_content(raw)
        # Should not crash; returns something reasonable (content or empty)
        assert isinstance(result, str)

    def test_closed_think_block_stripped(self):
        raw = "<think>Let me reason about this...</think>Here is the answer."
        assert _parse_fenced_content(raw) == "Here is the answer."

    def test_unclosed_think_block_returns_empty(self):
        raw = "<think>Reasoning step one...\nReasoning step two, still going..."
        result = _parse_fenced_content(raw)
        assert result == "", f"Expected empty string, got: {result!r}"

    def test_unclosed_think_does_not_leak_content(self):
        raw = "<think>SYSTEM PROMPT DETAILS: You are BRIO, a private assistant..."
        result = _parse_fenced_content(raw)
        assert "SYSTEM PROMPT DETAILS" not in result
        assert "<think>" not in result

    def test_content_before_unclosed_think_preserved(self):
        raw = "Here is part of the answer.\n<think>Then I started reasoning..."
        result = _parse_fenced_content(raw)
        assert "Here is part of the answer." in result
        assert "<think>" not in result

    def test_fenced_with_closed_think_inside(self):
        raw = "<out>\n<think>internal reasoning</think>\nActual answer here.\n</out>"
        assert _parse_fenced_content(raw) == "Actual answer here."

    def test_fenced_with_unclosed_think_returns_empty(self):
        raw = "<out>\n<think>Reasoning that never ends...</out>"
        result = _parse_fenced_content(raw)
        assert result == ""

    def test_multiple_closed_think_blocks_all_stripped(self):
        raw = "<think>first thought</think>Answer start <think>second thought</think> answer end."
        result = _parse_fenced_content(raw)
        assert "<think>" not in result
        assert "Answer start" in result
        assert "answer end." in result

    def test_stray_closing_tags_stripped(self):
        """Orphan </think>, </assistant>, <|im_end|> etc. are stripped."""
        raw = "Good answer.</think>"
        result = _parse_fenced_content(raw)
        assert "</think>" not in result
        assert "Good answer." in result


# ---------------------------------------------------------------------------
# StreamingFenceFilter — streaming-path fencing extraction
# ---------------------------------------------------------------------------

class TestStreamingFenceFilter:
    """StreamingFenceFilter extracts <out>/<output> fenced content from a token stream."""

    def test_plain_text_passes_through(self):
        tokens = ["Hello ", "world"]
        assert _run_fence_filter(tokens) == "Hello world"

    def test_out_fencing_extracted(self):
        tokens = ["<out>", "The actual answer.", "</out>"]
        assert _run_fence_filter(tokens) == "The actual answer."

    def test_output_fencing_extracted(self):
        tokens = ["<output>", "The actual answer.", "</output>"]
        assert _run_fence_filter(tokens) == "The actual answer."

    def test_content_after_close_tag_discarded(self):
        """Nothing after </out> should appear in output."""
        tokens = ["<out>Answer</out>", "extra content"]
        result = _run_fence_filter(tokens)
        assert "extra content" not in result
        assert "Answer" in result

    def test_fencing_split_across_tokens(self):
        """Open/close tags split across multiple tokens are still detected."""
        tokens = ["<", "out>", "Inner content.", "</", "out>"]
        result = _run_fence_filter(tokens)
        assert "Inner content." in result
        assert "<out>" not in result
        assert "</out>" not in result

    def test_no_fencing_passthrough(self):
        """When no <out> tag appears, all content passes through unchanged."""
        tokens = ["No", " fencing", " here."]
        assert _run_fence_filter(tokens) == "No fencing here."

    def test_empty_tokens_handled(self):
        tokens = ["<out>", "", "content", "", "</out>"]
        result = _run_fence_filter(tokens)
        assert "content" in result

    def test_output_tag_multipart(self):
        tokens = ["<output>Result</output>"]
        assert _run_fence_filter(tokens) == "Result"


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
