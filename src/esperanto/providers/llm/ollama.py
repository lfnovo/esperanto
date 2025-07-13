"""Ollama language model provider."""

import json
import os
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

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

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama


class OllamaLanguageModel(LanguageModel):
    """Ollama language model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Set default base URL if not provided
        self.base_url = (
            self.base_url or os.getenv("OLLAMA_BASE_URL")  or os.getenv("OLLAMA_API_BASE") or "http://localhost:11434"
        )

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Ollama API requests."""
        return {
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Ollama API error: {error_message}")

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = {}
        config = self.get_completion_kwargs()
        options = {}

        # Only include non-provider-specific args that were explicitly set
        for key, value in config.items():
            if key not in ["model_name", "base_url", "streaming"]:
                if key in ["temperature", "top_p"]:
                    options[key] = value
                elif key == "max_tokens":
                    # Convert max_tokens to num_predict for Ollama
                    options["num_predict"] = value
                else:
                    kwargs[key] = value

        # Handle JSON format if structured output is requested
        if self.structured:
            if not isinstance(self.structured, dict):
                raise TypeError("structured parameter must be a dictionary")
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                kwargs["format"] = "json"

        # Add options if any were set
        if options:
            kwargs["options"] = options

        return kwargs

    def _parse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse streaming response from Ollama."""
        for line in response.iter_lines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def _parse_stream_async(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse streaming response from Ollama asynchronously."""
        async for line in response.aiter_lines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Generate a chat completion for the given messages."""
        should_stream = stream if stream is not None else self.streaming

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

        # Prepare request payload
        payload = {
            "model": self.get_model_name(),
            "messages": messages,
            "stream": should_stream,
            **self._get_api_kwargs(),
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/api/chat",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            return (self._normalize_chunk(chunk) for chunk in self._parse_stream(response))
        
        response_data = response.json()
        return self._normalize_response(response_data)


    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Generate a chat completion for the given messages asynchronously."""
        should_stream = stream if stream is not None else self.streaming

        if not messages:
            raise ValueError("Messages cannot be empty")

        # Prepare request payload
        payload = {
            "model": self.get_model_name(),
            "messages": messages,
            "stream": should_stream,
            **self._get_api_kwargs(),
        }

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/api/chat",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            async def generate():
                async for chunk in self._parse_stream_async(response):
                    yield self._normalize_chunk(chunk)

            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)


    def _normalize_response(self, response: Dict[str, Any]) -> ChatCompletion:
        """Normalize a chat completion response."""
        message = response.get("message", {})
        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                    ),
                    finish_reason="stop",
                )
            ],
            model=response.get("model", self.get_model_name()),
            provider=self.provider,
            created=int(time.time()),
            usage=Usage(
                completion_tokens=response.get("eval_count", 0),
                prompt_tokens=response.get("prompt_eval_count", 0),
                total_tokens=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
            ),
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
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        response = self.client.get(
            f"{self.base_url}/api/tags",
            headers=self._get_headers()
        )
        self._handle_error(response)
        
        models_data = response.json()
        return [
            Model(
                id=model["name"],
                owned_by="Ollama",
                context_window=32768,  # Default context window for most Ollama models
                type="language",
            )
            for model in models_data.get("models", [])
        ]

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "ollama"

    def to_langchain(self) -> "ChatOllama":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_ollama is not installed.
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_ollama. "
                "Install with: uv add langchain_ollama or pip install langchain_ollama"
            ) from e

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")

        return ChatOllama(
            model=model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.max_tokens,
            # streaming=self.streaming, # Not a constructor arg
            base_url=self.base_url,
        )
