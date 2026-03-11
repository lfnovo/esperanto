"""Tests for StreamingThinkTagFilter."""

import pytest

from esperanto.utils.streaming import StreamingThinkTagFilter


def _run_filter(tokens, tag_names=None):
    """Helper: feed tokens through a filter and return collected output."""
    f = StreamingThinkTagFilter(tag_names=tag_names)
    parts = []
    for t in tokens:
        out = f.process(t)
        if out:
            parts.append(out)
    remaining = f.flush()
    if remaining:
        parts.append(remaining)
    return "".join(parts)


class TestPassthrough:
    def test_plain_text(self):
        assert _run_filter(["Hello ", "world"]) == "Hello world"

    def test_empty_tokens(self):
        assert _run_filter(["", "", "hi", ""]) == "hi"

    def test_single_token(self):
        assert _run_filter(["complete sentence"]) == "complete sentence"


class TestBasicThinkRemoval:
    def test_think_tag_whole_tokens(self):
        tokens = ["<think>", "reasoning here", "</think>", "The answer is 4"]
        assert _run_filter(tokens) == "The answer is 4"

    def test_think_tag_in_single_token(self):
        tokens = ["<think>internal thought</think>The answer is 4"]
        assert _run_filter(tokens) == "The answer is 4"

    def test_only_think_content(self):
        tokens = ["<think>", "all thinking", "</think>"]
        assert _run_filter(tokens) == ""

    def test_think_at_start_then_content(self):
        tokens = ["<think>", "hmm let me see", "</think>", "42"]
        assert _run_filter(tokens) == "42"


class TestPartialTagBuffering:
    def test_tag_split_across_tokens(self):
        tokens = ["<", "think", ">", "hidden", "</", "think", ">", "visible"]
        assert _run_filter(tokens) == "visible"

    def test_tag_split_mid_word(self):
        tokens = ["<thi", "nk>", "suppressed", "</thi", "nk>", "output"]
        assert _run_filter(tokens) == "output"

    def test_partial_match_then_not_a_tag(self):
        """<this is not a think tag, should pass through."""
        tokens = ["<this", " is content"]
        assert _run_filter(tokens) == "<this is content"

    def test_angle_bracket_in_normal_content(self):
        tokens = ["x < y and y > z"]
        assert _run_filter(tokens) == "x < y and y > z"

    def test_lone_angle_bracket_flushed(self):
        tokens = ["hello <"]
        assert _run_filter(tokens) == "hello <"


class TestMultipleThinkBlocks:
    def test_two_think_blocks(self):
        tokens = [
            "<think>", "first thought", "</think>",
            "answer 1 ",
            "<think>", "second thought", "</think>",
            "answer 2",
        ]
        assert _run_filter(tokens) == "answer 1 answer 2"

    def test_think_between_content(self):
        tokens = ["start ", "<think>", "hidden", "</think>", " end"]
        assert _run_filter(tokens) == "start  end"


class TestCustomTagNames:
    def test_reasoning_tag(self):
        tokens = ["<reasoning>", "step by step", "</reasoning>", "42"]
        assert _run_filter(tokens, tag_names=["reasoning"]) == "42"

    def test_multiple_tag_names(self):
        tokens = [
            "<think>", "thought", "</think>",
            "A ",
            "<reasoning>", "logic", "</reasoning>",
            "B",
        ]
        assert _run_filter(tokens, tag_names=["think", "reasoning"]) == "A B"

    def test_unregistered_tag_passes_through(self):
        tokens = ["<reflection>", "content", "</reflection>"]
        result = _run_filter(tokens, tag_names=["think"])
        assert "<reflection>" in result
        assert "content" in result


class TestStateProperty:
    def test_inside_think_tracks_state(self):
        f = StreamingThinkTagFilter()
        assert not f.inside_think
        f.process("<think>")
        assert f.inside_think
        f.process("hidden")
        assert f.inside_think
        f.process("</think>")
        assert not f.inside_think

    def test_reset(self):
        f = StreamingThinkTagFilter()
        f.process("<think>")
        assert f.inside_think
        f.reset()
        assert not f.inside_think


class TestEdgeCases:
    def test_nested_angle_brackets_not_tag(self):
        tokens = ["<not_a_tag>", "content"]
        result = _run_filter(tokens)
        assert "content" in result
        assert "<not_a_tag>" in result

    def test_empty_think_block(self):
        tokens = ["<think>", "</think>", "visible"]
        assert _run_filter(tokens) == "visible"

    def test_unclosed_think_tag_suppresses_rest(self):
        tokens = ["<think>", "never closed"]
        assert _run_filter(tokens) == ""

    def test_close_tag_without_open(self):
        """Close tag without open should not crash, content passes through."""
        tokens = ["hello ", "</think>", " world"]
        result = _run_filter(tokens)
        assert "hello" in result
        assert "world" in result

    def test_char_by_char_streaming(self):
        text = "<think>hidden</think>visible"
        tokens = list(text)  # one char at a time
        assert _run_filter(tokens) == "visible"
