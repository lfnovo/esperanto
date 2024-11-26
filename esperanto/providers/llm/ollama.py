"""Ollama language model provider."""

import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Union

from langchain_ollama import ChatOllama
from ollama import AsyncClient, Client

from esperanto.providers.llm.base import LanguageModel
from esperanto.types import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
    DeltaMessage,
    StreamChoice,
)


class OllamaLanguageModel(LanguageModel):
    """Ollama language model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set default base URL if not provided
        self.base_url = (
            kwargs.get("base_url")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )

        # Initialize clients
        self.client = Client(host=self.base_url)
        self.async_client = AsyncClient(host=self.base_url)

    def _get_api_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = self.get_completion_kwargs(**kwargs)

        # Remove provider-specific kwargs that Ollama doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("base_url", None)
        kwargs.pop("streaming", None)

        # Handle JSON format if structured output is requested
        if self.structured == "json":
            kwargs["format"] = "json"

        # Move parameters to options dict as expected by Ollama client
        options = {}
        for key in ["temperature", "top_p", "max_tokens"]:
            if key in kwargs:
                options[key] = kwargs.pop(key)

        if options:
            kwargs["options"] = options

        return kwargs

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Generate a chat completion for the given messages."""
        if not messages:
            raise ValueError("Messages cannot be empty")

        # Validate message format
        for message in messages:
            if "role" not in message:
                raise ValueError("Missing role in message")
            if message["role"] not in ["user", "assistant", "system", "tool"]:
                raise ValueError("Invalid role in message")
            if "content" not in message:
                raise ValueError("Missing content in message")

        api_kwargs = self._get_api_kwargs(**kwargs)

        if stream:
            return self._stream_chat_complete(messages, api_kwargs)
        return self._chat_complete(messages, api_kwargs)

    def _stream_chat_complete(
        self, messages: List[Dict[str, str]], api_kwargs: Dict[str, Any]
    ) -> Iterator[ChatCompletionChunk]:
        """Stream chat completion chunks."""
        response = self.client.chat(
            model=self.get_model_name(),
            messages=messages,
            stream=True,
            **api_kwargs,
        )
        for chunk in response:
            if isinstance(chunk, str):
                # Skip non-dict chunks (e.g., 'model' string)
                continue
            yield self._normalize_chunk(chunk)

    def _chat_complete(
        self, messages: List[Dict[str, str]], api_kwargs: Dict[str, Any]
    ) -> ChatCompletion:
        """Generate a non-streaming chat completion."""
        response = self.client.chat(
            model=self.get_model_name(),
            messages=messages,
            stream=False,
            **api_kwargs,
        )
        return self._normalize_response(response)

    async def achat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Generate a chat completion for the given messages asynchronously."""
        api_kwargs = self._get_api_kwargs(**kwargs)

        if stream:
            return self._astream_chat_complete(messages, api_kwargs)
        return await self._achat_complete(messages, api_kwargs)

    async def _astream_chat_complete(
        self, messages: List[Dict[str, str]], api_kwargs: Dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chat completion chunks asynchronously."""
        response = await self.async_client.chat(
            model=self.get_model_name(),
            messages=messages,
            stream=True,
            **api_kwargs,
        )
        async for chunk in response:
            if isinstance(chunk, str):
                # Skip non-dict chunks (e.g., 'model' string)
                continue

            yield ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            content=chunk.get("message", {}).get("content", ""),
                            role="assistant",
                        ),
                        finish_reason=chunk.get("done") and "stop" or None,
                    )
                ],
                model=self.get_model_name(),
                created=int(time.time()),
                object="chat.completion.chunk",
            )

    async def _achat_complete(
        self, messages: List[Dict[str, str]], api_kwargs: Dict[str, Any]
    ) -> ChatCompletion:
        """Generate a non-streaming chat completion asynchronously."""
        response = await self.async_client.chat(
            model=self.get_model_name(),
            messages=messages,
            stream=False,
            **api_kwargs,
        )
        return self._normalize_response(response)

    def _normalize_response(self, response: Dict[str, Any]) -> ChatCompletion:
        """Normalize a chat completion response."""
        message = response.get("message", {})
        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                    ),
                    finish_reason="stop",
                )
            ],
            model=response.get("model", self.get_model_name()),
            provider="ollama",
            created=int(time.time()),
            usage=None,
        )

    def _normalize_chunk(self, chunk: Dict[str, Any]) -> ChatCompletionChunk:
        """Normalize a streaming chat completion chunk."""
        message = chunk.get("message", {})
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                    ),
                    finish_reason="stop" if chunk.get("done", False) else None,
                )
            ],
            model=chunk.get("model", self.get_model_name()),
            created=int(time.time()),
        )

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "gemma2"  # Default model available on the server

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "ollama"

    def to_langchain(self) -> ChatOllama:
        """Convert to a LangChain chat model."""
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.max_tokens,
            streaming=self.streaming,
            base_url=self.base_url,
        )
