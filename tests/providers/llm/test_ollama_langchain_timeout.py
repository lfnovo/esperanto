"""Timeout parity through the real LangChain/Ollama HTTP clients."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import httpx
import pytest
import pytest_asyncio

from esperanto import AIFactory


@pytest_asyncio.fixture
async def make_chat(monkeypatch):
    # Keep loopback tests independent of host proxy, TLS and tracing settings.
    for name in (
        "ESPERANTO_LLM_TIMEOUT",
        "OLLAMA_API_KEY",
        "ESPERANTO_SSL_VERIFY",
        "ESPERANTO_SSL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    models = []

    def create(base_url="http://127.0.0.1:9", **config):
        model = AIFactory.create_language(
            "ollama", "test-model", config={"base_url": base_url, **config}
        )
        chat = model.to_langchain()
        models.append((model, chat))
        return model, chat

    yield create
    for model, chat in models:
        model.close()
        await model.aclose()
        chat._client._client.close()
        await chat._async_client._client.aclose()


@pytest.mark.parametrize("ssl_verify", [True, False])
@pytest.mark.parametrize(
    ("env_timeout", "config", "expected"),
    [(None, {}, 60.0), ("90", {}, 90.0), ("90", {"timeout": 0.25}, 0.25)],
)
async def test_langchain_timeout_precedence(
    make_chat, monkeypatch, ssl_verify, env_timeout, config, expected
):
    if env_timeout is not None:
        monkeypatch.setenv("ESPERANTO_LLM_TIMEOUT", env_timeout)
    model, chat = make_chat(verify_ssl=ssl_verify, **config)
    assert model.client.timeout == httpx.Timeout(expected)
    assert chat._client._client.timeout == model.client.timeout
    assert chat._async_client._client.timeout == model.async_client.timeout
    assert chat.client_kwargs.get("verify", True) is ssl_verify


@pytest.fixture
def ollama_endpoint():
    release = Event()
    received = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            payload = (
                json.dumps(
                    {
                        "model": "test-model",
                        "message": {"role": "assistant", "content": "loopback reply"},
                        "done": True,
                    }
                ).encode()
                + b"\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            received.set()
            if self.path == "/stall/api/chat":
                # An unfixed client gets a response eventually, not an infinite hang.
                release.wait(timeout=3)
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the timed-out client has already closed.

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # server_close joins request handlers after the teardown releases them.
    server.daemon_threads = False
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_langchain_times_out_on_stalled_response(
    make_chat, ollama_endpoint, use_async, stream
):
    url, received = ollama_endpoint
    _, chat = make_chat(base_url=url + "/stall", timeout=0.25)
    with pytest.raises(httpx.ReadTimeout):
        if use_async:
            await chat.ainvoke("hello", stream=stream)
        else:
            chat.invoke("hello", stream=stream)
    assert received.is_set()


@pytest.mark.parametrize("use_async", [False, True])
async def test_langchain_responsive_endpoint_still_works(
    make_chat, ollama_endpoint, use_async
):
    url, received = ollama_endpoint
    _, chat = make_chat(base_url=url, timeout=2)
    if use_async:
        response = await chat.ainvoke("hello")
    else:
        response = chat.invoke("hello")
    assert received.is_set()
    assert response.content == "loopback reply"
