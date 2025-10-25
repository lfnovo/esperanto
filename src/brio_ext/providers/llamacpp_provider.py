"""Local llama.cpp provider integration."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

import httpx

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    Message,
    Model,
    StreamChoice,
    Usage,
)
from esperanto.providers.llm.base import LanguageModel


class LlamaCppLanguageModel(LanguageModel):
    """LanguageModel implementation that targets a local llama.cpp server."""

    def __post_init__(self):
        super().__post_init__()
        self.base_url = (
            self.base_url
            or os.getenv("LLAMACPP_BASE_URL")
            or "http://localhost:8080"
        )
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                payload = response.json()
                message = payload.get("error") or payload.get("message") or response.text
            except Exception:
                message = response.text
            raise RuntimeError(f"llama.cpp error ({response.status_code}): {message}")

    def _get_api_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        config = self.get_completion_kwargs()

        for key, value in config.items():
            if value is None:
                continue
            if key == "max_tokens":
                kwargs["n_predict"] = value
            elif key in {"temperature", "top_p"}:
                kwargs[key] = value

        # Pass-through any explicit overrides from the config dictionary.
        extras = getattr(self, "_config", {}) or {}
        for key, value in extras.items():
            if value is None:
                continue
            if key in {"model_name", "streaming", "structured"}:
                continue
            if key == "max_tokens":
                kwargs["n_predict"] = value
            elif key == "stop":
                kwargs["stop"] = value
            else:
                kwargs.setdefault(key, value)

        return kwargs

    def models(self) -> List[Model]:
        response = self.client.get(f"{self.base_url}/v1/models", headers=self._get_headers())
        self._handle_error(response)
        data = response.json()
        models: List[Model] = []
        for item in data.get("data", []):
            models.append(
                Model(
                    id=item.get("id", ""),
                    owned_by="local",
                    context_window=item.get("context_length"),
                    type="language",
                )
            )
        return models

    def prompt_complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        should_stream = stream if stream is not None else self.streaming
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "prompt": prompt,
            "stream": should_stream,
            **self._get_api_kwargs(),
        }
        if stop:
            payload["stop"] = stop

        response = self.client.post(
            f"{self.base_url}/v1/completions",
            headers=self._get_headers(),
            json=payload,
        )
        self._handle_error(response)

        if should_stream:
            return (self._normalize_chunk(chunk) for chunk in self._parse_stream(response))
        return self._normalize_response(response.json())

    async def aprompt_complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        should_stream = stream if stream is not None else self.streaming
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "prompt": prompt,
            "stream": should_stream,
            **self._get_api_kwargs(),
        }
        if stop:
            payload["stop"] = stop

        response = await self.async_client.post(
            f"{self.base_url}/v1/completions",
            headers=self._get_headers(),
            json=payload,
        )
        self._handle_error(response)

        if should_stream:
            async def _stream() -> AsyncGenerator[ChatCompletionChunk, None]:
                async for chunk in self._parse_stream_async(response):
                    yield self._normalize_chunk(chunk)

            return _stream()
        return self._normalize_response(response.json())

    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        prompt = "\n".join(message.get("content", "") for message in messages)
        stop = getattr(self, "_config", {}).get("stop")
        return self.prompt_complete(prompt, stop=stop, stream=stream)

    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        prompt = "\n".join(message.get("content", "") for message in messages)
        stop = getattr(self, "_config", {}).get("stop")
        return await self.aprompt_complete(prompt, stop=stop, stream=stream)

    def _normalize_response(self, data: Dict[str, Any]) -> ChatCompletion:
        choices = data.get("choices", [])
        first = choices[0] if choices else {}
        text = first.get("text") or first.get("content") or ""
        finish_reason = first.get("finish_reason")

        usage_data = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return ChatCompletion(
            id=data.get("id", str(uuid.uuid4())),
            choices=[
                Choice(
                    index=first.get("index", 0),
                    message=Message(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            created=data.get("created"),
            model=data.get("model", self.get_model_name()),
            provider=self.provider,
            usage=usage,
        )

    def _normalize_chunk(self, chunk: Dict[str, Any]) -> ChatCompletionChunk:
        choices = chunk.get("choices", [])
        normalized_choices: List[StreamChoice] = []
        for choice in choices:
            normalized_choices.append(
                StreamChoice(
                    index=choice.get("index", 0),
                    delta=DeltaMessage(
                        role="assistant",
                        content=choice.get("text") or choice.get("content") or "",
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            )
        return ChatCompletionChunk(
            id=chunk.get("id", str(uuid.uuid4())),
            choices=normalized_choices,
            created=chunk.get("created"),
            model=chunk.get("model", self.get_model_name()),
        )

    def _parse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    async def _parse_stream_async(self, response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def _get_default_model(self) -> str:
        return "llama-3.1-8b-instruct"

    @property
    def provider(self) -> str:
        return "llamacpp"

    def to_langchain(self) -> Any:
        raise NotImplementedError("LangChain integration not implemented for llama.cpp")
