"""Tests for Message.thinking and Message.cleaned_content properties."""

import pytest

from esperanto.common_types import Message


class TestMessageThinkingProperty:
    """Tests for the thinking property."""

    def test_thinking_extracts_single_block(self):
        """Test extracting content from a single think block."""
        content = "<think>Let me think about this.</think>\n\nThe answer is 42."
        msg = Message(content=content, role="assistant")
        assert msg.thinking == "Let me think about this."

    def test_thinking_extracts_multiline_block(self):
        """Test extracting multiline content from think block."""
        content = """<think>
First, I need to consider X.
Then, I should look at Y.
Finally, Z is important.
</think>

The answer is 42."""
        msg = Message(content=content, role="assistant")
        assert "First, I need to consider X." in msg.thinking
        assert "Then, I should look at Y." in msg.thinking
        assert "Finally, Z is important." in msg.thinking

    def test_thinking_concatenates_multiple_blocks(self):
        """Test that multiple think blocks are concatenated."""
        content = """<think>First thought</think>

Some response

<think>Second thought</think>

More response"""
        msg = Message(content=content, role="assistant")
        assert msg.thinking == "First thought\n\nSecond thought"

    def test_thinking_returns_none_without_tags(self):
        """Test that thinking returns None when no think tags present."""
        content = '{"name": "John", "age": 30}'
        msg = Message(content=content, role="assistant")
        assert msg.thinking is None

    def test_thinking_returns_none_for_empty_content(self):
        """Test that thinking returns None for None content."""
        msg = Message(content=None, role="assistant")
        assert msg.thinking is None

    def test_thinking_returns_none_for_empty_think_tags(self):
        """Test that thinking returns None when think tags are empty."""
        content = "<think>\n\n</think>\n\n{\"result\": 42}"
        msg = Message(content=content, role="assistant")
        assert msg.thinking is None

    def test_thinking_strips_whitespace(self):
        """Test that thinking content is stripped of leading/trailing whitespace."""
        content = "<think>   \n  Padded content  \n   </think>"
        msg = Message(content=content, role="assistant")
        assert msg.thinking == "Padded content"


class TestMessageCleanedContentProperty:
    """Tests for the cleaned_content property."""

    def test_cleaned_content_removes_think_block(self):
        """Test that cleaned_content removes think blocks."""
        content = "<think>Let me think.</think>\n\n{\"answer\": 42}"
        msg = Message(content=content, role="assistant")
        assert msg.cleaned_content == '{"answer": 42}'

    def test_cleaned_content_removes_multiple_blocks(self):
        """Test that cleaned_content removes multiple think blocks."""
        content = """<think>First thought</think>

Some response

<think>Second thought</think>

More response"""
        msg = Message(content=content, role="assistant")
        cleaned = msg.cleaned_content
        assert "<think>" not in cleaned
        assert "</think>" not in cleaned
        assert "First thought" not in cleaned
        assert "Second thought" not in cleaned
        assert "Some response" in cleaned
        assert "More response" in cleaned

    def test_cleaned_content_returns_full_content_without_tags(self):
        """Test that cleaned_content returns full content when no think tags."""
        content = '{"name": "John", "age": 30}'
        msg = Message(content=content, role="assistant")
        assert msg.cleaned_content == content

    def test_cleaned_content_returns_empty_string_for_none(self):
        """Test that cleaned_content returns empty string for None content."""
        msg = Message(content=None, role="assistant")
        assert msg.cleaned_content == ""

    def test_cleaned_content_handles_empty_think_tags(self):
        """Test that cleaned_content handles empty think tags correctly."""
        content = "<think>\n\n</think>\n\n{\"result\": 42}"
        msg = Message(content=content, role="assistant")
        assert msg.cleaned_content == '{"result": 42}'

    def test_cleaned_content_cleans_excessive_newlines(self):
        """Test that cleaned_content normalizes excessive newlines."""
        content = "<think>Thinking...</think>\n\n\n\n\nThe result"
        msg = Message(content=content, role="assistant")
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in msg.cleaned_content
        assert "The result" in msg.cleaned_content


class TestMessageThinkingIntegration:
    """Integration tests for thinking/cleaned_content with real-world examples."""

    def test_qwen_style_response(self):
        """Test parsing Qwen3-style response with think tags."""
        content = """<think>
Okay, the user wants a JSON with name and age for a fictional person.
Let me create something reasonable.
Name: Elena Voss
Age: 34
</think>

{"name": "Elena Voss", "age": 34}"""
        msg = Message(content=content, role="assistant")

        assert msg.thinking is not None
        assert "Elena Voss" in msg.thinking
        assert "Age: 34" in msg.thinking

        assert msg.cleaned_content == '{"name": "Elena Voss", "age": 34}'
        assert "<think>" not in msg.cleaned_content

    def test_empty_string_content(self):
        """Test with empty string content."""
        msg = Message(content="", role="assistant")
        assert msg.thinking is None
        assert msg.cleaned_content == ""

    def test_original_content_unchanged(self):
        """Test that original content property is unchanged."""
        content = "<think>Reasoning</think>\n\nResult"
        msg = Message(content=content, role="assistant")

        # Original content should be preserved
        assert msg.content == content
        # Properties should parse it differently
        assert msg.thinking == "Reasoning"
        assert msg.cleaned_content == "Result"
