"""Anthropic language model implementation."""

import json
import os
import time
import uuid
from dataclasses import dataclass
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
from esperanto.utils.logging import logger

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic


@dataclass
class AnthropicLanguageModel(LanguageModel):
    """Anthropic language model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable."
            )

        # Set base URL
        self.base_url = self.base_url or "https://api.anthropic.com/v1"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Anthropic API requests."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Anthropic API error: {error_message}")

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        try:
            response = self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            )
            self._handle_error(response)
            
            models_data = response.json()
            return [
                Model(
                    id=model["id"],
                    owned_by="Anthropic",
                    context_window=model.get("max_tokens", 200000),
                    type="language",
                )
                for model in models_data.get("data", [])
            ]
        except Exception:
            # Fallback to known models if API call fails
            return [
                Model(
                    id="claude-3-7-sonnet-20250219",
                    owned_by="Anthropic",
                    context_window=200000,
                    type="language",
                ),
                Model(
                    id="claude-3-opus-20240229",
                    owned_by="Anthropic",
                    context_window=200000,
                    type="language",
                ),
                Model(
                    id="claude-3-sonnet-20240229",
                    owned_by="Anthropic",
                    context_window=200000,
                    type="language",
                ),
                Model(
                    id="claude-3-haiku-20240307",
                    owned_by="Anthropic",
                    context_window=200000,
                    type="language",
                ),
            ]

    def _prepare_messages(
        self, messages: List[Dict[str, str]]
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """Handle Anthropic-specific message preparation."""
        system_message = None
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                formatted_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"],
                })
        
        return system_message, formatted_messages

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Anthropic response to our format."""
        created = int(time.time())

        # Extract content text from response
        content_text = ""
        if "content" in response_data and response_data["content"]:
            for block in response_data["content"]:
                if block.get("type") == "text":
                    content_text = block.get("text", "")
                    break

        return ChatCompletion(
            id=response_data.get("id", str(uuid.uuid4())),
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        content=content_text,
                        role="assistant",
                    ),
                    finish_reason=response_data.get("stop_reason", "stop"),
                )
            ],
            created=created,
            model=response_data.get("model", self.get_model_name()),
            provider=self.provider,
            usage=Usage(
                completion_tokens=response_data.get("usage", {}).get("output_tokens", 0),
                prompt_tokens=response_data.get("usage", {}).get("input_tokens", 0),
                total_tokens=response_data.get("usage", {}).get("input_tokens", 0) + response_data.get("usage", {}).get("output_tokens", 0),
            ),
        )

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse Server-Sent Events stream from Anthropic."""
        for chunk in response.iter_text():
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    async def _parse_sse_stream_async(self, response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
        """Parse Server-Sent Events stream from Anthropic asynchronously."""
        async for chunk in response.aiter_text():
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip() == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    def _normalize_stream_event(self, event_data: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """Normalize Anthropic stream event to our format."""
        event_type = event_data.get("type")
        
        # Handle content delta events
        if event_type == "content_block_delta":
            delta = event_data.get("delta", {})
            if "text" in delta:
                return ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                content=delta["text"],
                                role="assistant",
                            ),
                            finish_reason=None,
                        )
                    ],
                    created=int(time.time()),
                    model=self.get_model_name(),
                )

        # Handle message completion event
        elif event_type == "message_delta":
            delta = event_data.get("delta", {})
            return ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            content=None,
                            role="assistant",
                        ),
                        finish_reason=delta.get("stop_reason", "stop"),
                    )
                ],
                created=int(time.time()),
                model=self.get_model_name(),
            )

        # Ignore other event types
        return None

    # Removed the faulty _prepare_api_kwargs method

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = self.get_completion_kwargs()

        # Remove provider-specific kwargs that Anthropic doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        kwargs.pop("organization", None)

        # Handle streaming
        if exclude_stream:
            kwargs.pop("streaming", None)
        elif "streaming" in kwargs:
            kwargs["stream"] = kwargs.pop("streaming")

        # Handle temperature - Anthropic expects 0-1 range
        if "temperature" in kwargs:
            temp = kwargs["temperature"]
            if temp is not None:
                kwargs["temperature"] = max(0.0, min(1.0, float(temp)))

        # Handle max_tokens - required by Anthropic
        if "max_tokens" in kwargs:
            max_tokens = kwargs["max_tokens"]
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)

        return kwargs

    def get_model_name(self) -> str:
        """Get the model name to use."""
        return self.model_name or self._get_default_model()

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "claude-3-7-sonnet-20250219"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "anthropic"

    def _create_request_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """Create request payload for Anthropic API."""
        system_message, formatted_messages = self._prepare_messages(messages)
        
        payload = {
            "model": self.get_model_name(),
            "messages": formatted_messages,
            "max_tokens": self.max_tokens or 1024,
        }
        
        if system_message:
            payload["system"] = system_message
            
        if self.temperature is not None:
            payload["temperature"] = max(0.0, min(1.0, float(self.temperature)))
            
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
            
        if stream:
            payload["stream"] = True
            
        return payload

    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request."""

        should_stream = stream if stream is not None else self.streaming
        payload = self._create_request_payload(messages, should_stream)
        
        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/messages",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            def generate():
                for event_data in self._parse_sse_stream(response):
                    chunk = self._normalize_stream_event(event_data)
                    if chunk:
                        yield chunk
            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)

    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request."""

        should_stream = stream if stream is not None else self.streaming
        payload = self._create_request_payload(messages, should_stream)
        
        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/messages",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            async def generate():
                async for event_data in self._parse_sse_stream_async(response):
                    chunk = self._normalize_stream_event(event_data)
                    if chunk:
                        yield chunk
            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)

    def to_langchain(self) -> "ChatAnthropic":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_anthropic is not installed.
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_anthropic. "
                "Install with: uv add langchain_anthropic or pip install langchain_anthropic"
            ) from e


        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")

        # Pass arguments directly to ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            api_key=self.api_key,
        )
