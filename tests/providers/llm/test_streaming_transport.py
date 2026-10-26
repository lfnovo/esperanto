"""Regression tests for the non-buffering HTTP streaming transport."""

import asyncio
import concurrent.futures
import json
import threading
from typing import Any, Dict, Type

import httpx
import pytest

from esperanto.providers.llm.anthropic import AnthropicLanguageModel
from esperanto.providers.llm.azure import AzureLanguageModel
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.llm.cohere import CohereLanguageModel
from esperanto.providers.llm.google import GoogleLanguageModel
from esperanto.providers.llm.groq import GroqLanguageModel
from esperanto.providers.llm.mistral import MistralLanguageModel
from esperanto.providers.llm.ollama import OllamaLanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel
from esperanto.providers.llm.openrouter import OpenRouterLanguageModel
from esperanto.providers.llm.perplexity import PerplexityLanguageModel

STREAMING_PROVIDERS = [
    AnthropicLanguageModel,
    AzureLanguageModel,
    CohereLanguageModel,
    GoogleLanguageModel,
    GroqLanguageModel,
    MistralLanguageModel,
    OllamaLanguageModel,
    OpenAILanguageModel,
    OpenAICompatibleLanguageModel,
    OpenRouterLanguageModel,
    PerplexityLanguageModel,
]
MESSAGES = [{"role": "user", "content": "hello"}]


def _model_kwargs(provider: Type[LanguageModel]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"api_key": "test-key", "base_url": "https://example.test"}
    if provider is AzureLanguageModel:
        kwargs.update({"model_name": "deployment", "config": {"api_version": "2024-01-01"}})
    elif provider is OllamaLanguageModel:
        kwargs.pop("api_key")
        kwargs["model_name"] = "test-model"
    elif provider in (OpenAICompatibleLanguageModel, OpenRouterLanguageModel):
        kwargs["model_name"] = "test-model"
    elif provider is GoogleLanguageModel:
        kwargs["model_name"] = "gemini-test"
    elif provider is PerplexityLanguageModel:
        kwargs["model_name"] = "sonar"
    return kwargs


def _sse(data: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def _wire_events(provider: Type[LanguageModel]) -> tuple[bytes, bytes]:
    if provider is AnthropicLanguageModel:
        return (
            _sse({"type": "content_block_delta", "index": 0, "delta": {"text": "first"}}),
            _sse({"type": "message_delta", "index": 1, "delta": {"stop_reason": "end_turn"}}),
        )
    if provider is CohereLanguageModel:
        return (
            _sse({"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "first"}}}}),
            _sse({"type": "message-end", "delta": {"finish_reason": "COMPLETE"}}),
        )
    if provider is OllamaLanguageModel:
        return (
            b'{"model":"test-model","message":{"role":"assistant","content":"first"},"done":false}\n',
            b'{"model":"test-model","message":{"role":"assistant","content":""},"done":true}\n',
        )
    if provider is GoogleLanguageModel:
        return (
            _sse({"candidates": [{"content": {"parts": [{"text": "first"}], "role": "model"}}]}),
            _sse({"candidates": [{"finishReason": "STOP"}]}),
        )
    return (
        _sse({"id": "stream-test", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "first"}, "finish_reason": None}], "created": 1, "model": "test-model"}),
        b"data: [DONE]\n\n",
    )


class _SyncBody(httpx.SyncByteStream):
    def __init__(self, first: bytes, rest: bytes, release: threading.Event):
        self.first = first
        self.rest = rest
        self.release = release

    def __iter__(self):
        yield self.first
        assert self.release.wait(5)
        yield self.rest


class _AsyncBody(httpx.AsyncByteStream):
    def __init__(self, first: bytes, rest: bytes, release: asyncio.Event):
        self.first = first
        self.rest = rest
        self.release = release

    async def __aiter__(self):
        yield self.first
        await asyncio.wait_for(self.release.wait(), timeout=1)
        yield self.rest


def _install_clients(model: LanguageModel, handler, async_handler) -> None:
    model.client = httpx.Client(transport=httpx.MockTransport(handler))
    model.async_client = httpx.AsyncClient(transport=httpx.MockTransport(async_handler))


def _new_model(provider: Type[LanguageModel]) -> LanguageModel:
    return provider(**_model_kwargs(provider))


@pytest.mark.parametrize("provider", STREAMING_PROVIDERS)
def test_provider_chat_streaming_opens_before_body_finishes(provider):
    release = threading.Event()
    first, rest = _wire_events(provider)

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_SyncBody(first, rest, release))

    model = _new_model(provider)
    _install_clients(model, handler, handler)

    def receive_first():
        stream = model.chat_complete(MESSAGES, stream=True)
        return stream, next(stream)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(receive_first)
        try:
            stream, chunk = future.result(timeout=1)
            assert chunk.choices[0].delta.content == "first"
            stream.close()
        finally:
            release.set()
    model.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", STREAMING_PROVIDERS)
async def test_provider_achat_streaming_opens_before_body_finishes(provider):
    release = asyncio.Event()
    first, rest = _wire_events(provider)

    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_AsyncBody(first, rest, release))

    model = _new_model(provider)
    _install_clients(model, lambda request: None, handler)

    async def receive_first():
        stream = await model.achat_complete(MESSAGES, stream=True)
        return stream, await anext(stream)

    stream, chunk = await asyncio.wait_for(receive_first(), timeout=1)
    assert chunk.choices[0].delta.content == "first"
    release.set()
    await stream.aclose()
    await model.async_client.aclose()


@pytest.mark.parametrize("provider", STREAMING_PROVIDERS)
def test_provider_chat_streaming_raises_open_error(provider):
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    model = _new_model(provider)
    _install_clients(model, handler, handler)
    with pytest.raises(RuntimeError):
        model.chat_complete(MESSAGES, stream=True)
    model.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", STREAMING_PROVIDERS)
async def test_provider_achat_streaming_raises_open_error(provider):
    async def handler(request):
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    model = _new_model(provider)
    _install_clients(model, lambda request: None, handler)
    with pytest.raises(RuntimeError):
        await model.achat_complete(MESSAGES, stream=True)
    await model.async_client.aclose()
