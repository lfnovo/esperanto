"""Streaming utilities for filtering model output tokens."""

from typing import List, Optional


class StreamingThinkTagFilter:
    """State-machine filter that strips think-style tags from a token stream.

    Buffers partial tag matches and suppresses content inside think tags.
    Content outside think tags passes through immediately; the only latency
    is a few characters of buffering when a ``<`` is encountered.

    Usage::

        tag_filter = StreamingThinkTagFilter(tag_names=["think", "reasoning"])
        for token in stream:
            output = tag_filter.process(token)
            if output:
                yield output
        remaining = tag_filter.flush()
        if remaining:
            yield remaining
    """

    def __init__(self, tag_names: Optional[List[str]] = None):
        self.tag_names = tag_names or ["think"]
        self._open_tags = [f"<{name}>" for name in self.tag_names]
        self._close_tags = [f"</{name}>" for name in self.tag_names]
        self._all_tags = self._open_tags + self._close_tags
        self._buffer = ""
        self._inside_think = False

    @property
    def inside_think(self) -> bool:
        """Whether the filter is currently inside a think tag."""
        return self._inside_think

    def process(self, token: str) -> str:
        """Process a token and return content to yield (may be empty string)."""
        output = ""
        for char in token:
            self._buffer += char

            if self._buffer[0] == "<":
                # Check for complete open tag match
                matched = False
                for tag in self._open_tags:
                    if self._buffer == tag:
                        self._inside_think = True
                        self._buffer = ""
                        matched = True
                        break
                if not matched:
                    for tag in self._close_tags:
                        if self._buffer == tag:
                            self._inside_think = False
                            self._buffer = ""
                            matched = True
                            break
                if not matched:
                    # Still a valid prefix of some tag? Keep buffering.
                    is_prefix = any(
                        tag.startswith(self._buffer) for tag in self._all_tags
                    )
                    if not is_prefix:
                        # Not a tag — flush buffer
                        if not self._inside_think:
                            output += self._buffer
                        self._buffer = ""
            else:
                # Regular character — no tag possible
                if not self._inside_think:
                    output += char
                self._buffer = ""

        return output

    def flush(self) -> str:
        """Flush remaining buffer. Call once at end of stream."""
        if self._buffer and not self._inside_think:
            result = self._buffer
            self._buffer = ""
            return result
        self._buffer = ""
        return ""

    def reset(self) -> None:
        """Reset filter state for reuse."""
        self._buffer = ""
        self._inside_think = False
