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


class StreamingFenceFilter:
    """State-machine filter that extracts content from <out>/<output> fencing.

    When a response is wrapped in ``<out>...</out>`` or ``<output>...</output>``,
    only the inner content is yielded.  If no opening fence tag is found within
    the first ``MAX_SEARCH_CHARS`` characters, all content passes through unchanged.

    Combine with :class:`StreamingThinkTagFilter` by applying this filter first::

        fence_filter = StreamingFenceFilter()
        think_filter = StreamingThinkTagFilter()
        for token in stream:
            defenced = fence_filter.process(token)
            if defenced:
                output = think_filter.process(defenced)
                if output:
                    yield output
        remaining = think_filter.process(fence_filter.flush())
        remaining += think_filter.flush()
        if remaining:
            yield remaining
    """

    # Beyond this many characters without an open tag, assume no fencing.
    MAX_SEARCH_CHARS = 30

    OPEN_TAGS = ("<out>", "<output>")
    _CLOSE_MAP = {"<out>": "</out>", "<output>": "</output>"}

    def __init__(self) -> None:
        self._search_buf = ""   # Buffering while looking for open tag
        self._searching = True  # Still scanning for an open tag
        self._fenced = False    # Found an open tag; extracting inner content
        self._done = False      # Found close tag; discard the rest
        self._close_tag = ""    # Which close tag we're looking for
        self._inner_buf = ""    # Tail buffer to catch split close tags
        self._max_close_len = max(len(t) for t in self._CLOSE_MAP.values())

    def process(self, token: str) -> str:
        """Process a token and return content to yield (may be empty string)."""
        if self._done:
            return ""

        if not self._searching:
            if self._fenced:
                return self._consume_inner(token)
            # passthrough mode (no fencing detected)
            return token

        # Still scanning for an opening fence tag
        self._search_buf += token

        for tag in self.OPEN_TAGS:
            if self._search_buf.startswith(tag):
                self._searching = False
                self._fenced = True
                self._close_tag = self._CLOSE_MAP[tag]
                remainder = self._search_buf[len(tag):]
                self._search_buf = ""
                return self._consume_inner(remainder)

        # Is the accumulated buffer still a valid prefix of any open tag?
        buf = self._search_buf
        is_possible_prefix = any(tag.startswith(buf) for tag in self.OPEN_TAGS)

        if not is_possible_prefix or len(buf) >= self.MAX_SEARCH_CHARS:
            # Definitively not fenced — switch to passthrough
            self._searching = False
            self._fenced = False
            result = self._search_buf
            self._search_buf = ""
            return result

        return ""  # Still buffering

    def _consume_inner(self, text: str) -> str:
        """Process text that is inside the fence."""
        if not text:
            return ""
        self._inner_buf += text

        idx = self._inner_buf.find(self._close_tag)
        if idx != -1:
            output = self._inner_buf[:idx]
            self._inner_buf = ""
            self._done = True
            return output

        # Flush a safe prefix; keep the last max_close_len chars buffered so
        # we can detect a close tag that arrives split across multiple tokens.
        safe_len = max(0, len(self._inner_buf) - self._max_close_len)
        if safe_len > 0:
            output = self._inner_buf[:safe_len]
            self._inner_buf = self._inner_buf[safe_len:]
            return output
        return ""

    def flush(self) -> str:
        """Flush remaining buffer. Call once at end of stream."""
        if self._done:
            self._inner_buf = ""
            return ""
        if self._searching or not self._fenced:
            # Either still searching (no fence found) or passthrough
            result = self._search_buf + self._inner_buf
            self._search_buf = ""
            self._inner_buf = ""
            return result
        # Inside a fence but the stream ended before the close tag
        result = self._inner_buf
        self._inner_buf = ""
        return result

    def reset(self) -> None:
        """Reset filter state for reuse."""
        self._search_buf = ""
        self._searching = True
        self._fenced = False
        self._done = False
        self._close_tag = ""
        self._inner_buf = ""
