"""Unit tests for llama.cpp provider configuration."""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import MagicMock

import pytest

from brio_ext.providers.llamacpp_provider import LlamaCppLanguageModel


def test_llamacpp_max_tokens_maps_to_both_fields():
    model = LlamaCppLanguageModel(
        model_name="qwen2.5-7b-instruct",
        config={"max_tokens": 1200, "temperature": 0.25, "top_p": 0.8},
    )

    kwargs = model._get_api_kwargs()
    assert kwargs["max_tokens"] == 1200
    assert kwargs["n_predict"] == 1200
    assert kwargs["temperature"] == 0.25
    assert kwargs["top_p"] == 0.8

    # Ensure overrides persist on subsequent calls
    model._config["max_tokens"] = 64
    kwargs = model._get_api_kwargs()
    assert kwargs["max_tokens"] == 64
    assert kwargs["n_predict"] == 64


def test_parse_stream_handles_sse_format():
    """_parse_stream must strip 'data: ' prefix from SSE lines."""
    model = LlamaCppLanguageModel(model_name="test-model")

    sse_lines = [
        'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"id":"2","choices":[{"delta":{"content":" world"}}]}',
        "",
        "data: [DONE]",
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(sse_lines)

    chunks = list(model._parse_stream(mock_response))

    assert len(chunks) == 2, f"Expected 2 chunks but got {len(chunks)}: {chunks}"
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " world"


def test_parse_stream_handles_raw_json():
    """_parse_stream should also work with raw JSON lines (no SSE prefix)."""
    model = LlamaCppLanguageModel(model_name="test-model")

    raw_lines = [
        '{"id":"1","choices":[{"delta":{"content":"Hi"}}]}',
        '{"id":"2","choices":[{"delta":{"content":"!"}}]}',
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(raw_lines)

    chunks = list(model._parse_stream(mock_response))

    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hi"


# ---------------------------------------------------------------------------
# Integration test: real HTTP streaming with deliberate delays
# ---------------------------------------------------------------------------

TOKENS = ["Hello", ", ", "world", "!"]
CHUNK_DELAY_S = 0.05  # 50ms between each SSE chunk


def _make_sse_chunk(token: str, index: int) -> str:
    """Build one SSE data line for a chat completion chunk."""
    payload = {
        "id": f"chunk-{index}",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": token},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


class _SSEHandler(BaseHTTPRequestHandler):
    """Tiny handler that streams SSE chunks with deliberate delays."""

    def do_POST(self):
        # Read request body (required to avoid broken pipe)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        for i, token in enumerate(TOKENS):
            self.wfile.write(_make_sse_chunk(token, i).encode())
            self.wfile.flush()
            time.sleep(CHUNK_DELAY_S)

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        pass  # silence request logs during tests


@pytest.fixture()
def sse_server():
    """Start a local HTTP server that streams SSE with delays."""
    server = HTTPServer(("127.0.0.1", 0), _SSEHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_chat_complete_streams_incrementally(sse_server):
    """
    Tokens must arrive incrementally, not all at once.

    The server sends 4 tokens with 50ms gaps (total ~200ms).
    If httpx buffers the full response, all tokens arrive after ~200ms.
    If httpx streams properly, the first token arrives after ~50ms.
    """
    model = LlamaCppLanguageModel(
        model_name="test-model",
        config={"base_url": sse_server},
    )
    model.base_url = sse_server

    stream = model.chat_complete(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )

    arrival_times = []
    collected_tokens = []
    t0 = time.perf_counter()

    for chunk in stream:
        arrival_times.append(time.perf_counter() - t0)
        content = chunk.choices[0].delta.content
        if content:
            collected_tokens.append(content)

    # All tokens arrived
    assert collected_tokens == TOKENS

    # Key assertion: first token arrives well before last token.
    # With 4 chunks at 50ms each, total server time is ~200ms.
    # If truly streaming: first token at ~50ms, last at ~200ms → spread ~150ms.
    # If buffered: all tokens at ~200ms → spread ~0ms.
    spread = arrival_times[-1] - arrival_times[0]
    assert spread > 0.05, (
        f"Tokens arrived too close together (spread={spread:.3f}s). "
        f"HTTP response is likely buffered, not streamed. "
        f"Arrival times: {[f'{t:.3f}' for t in arrival_times]}"
    )
