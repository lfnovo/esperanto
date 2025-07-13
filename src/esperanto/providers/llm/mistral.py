"""Mistral language model provider."""

import json
import os
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

if TYPE_CHECKING:
    from langchain_mistralai import ChatMistralAI

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

MISTRAL_DEFAULT_MODEL_NAME = "mistral-large-latest"


class MistralLanguageModel(LanguageModel):
    """Mistral language model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")

        # Set base URL
        self.base_url = self.base_url or "https://api.mistral.ai/v1"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)


    def _get_default_model(self) -> str:
        """Get the default model name for Mistral."""
        return MISTRAL_DEFAULT_MODEL_NAME

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Mistral API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Mistral API error: {error_message}")

    @property
    def models(self) -> List[Model]:
        """List available Mistral models."""
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
                    owned_by=model.get("owned_by", "mistralai"),
                    context_window=None,  # Context window not provided in API response
                    type="language",
                )
                for model in models_data["data"]
                if "embed" not in model["id"]  # Filter out embedding models
            ]
        except Exception as e:
            print(f"Warning: Could not dynamically list models from Mistral API: {e}. Falling back to a known list.")
            # Fallback to a known list if API call fails
            known_models = [
                {"id": "open-mistral-7b", "context_window": 32000, "owned_by": "mistralai"},
                {"id": "open-mixtral-8x7b", "context_window": 32000, "owned_by": "mistralai"},
                {"id": "mistral-small-latest", "context_window": 32000, "owned_by": "mistralai"},
                {"id": "mistral-medium-latest", "context_window": 32000, "owned_by": "mistralai"},
                {"id": "mistral-large-latest", "context_window": 32000, "owned_by": "mistralai"},
            ]
            return [
                Model(id=m["id"], owned_by=m["owned_by"], context_window=m["context_window"], type="language")
                for m in known_models
            ]

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Mistral response to our format."""
        return ChatCompletion(
            id=response_data["id"],
            choices=[
                Choice(
                    index=choice["index"],
                    message=Message(
                        content=choice["message"]["content"] or "",
                        role=choice["message"]["role"],
                    ),
                    finish_reason=choice["finish_reason"],
                )
                for choice in response_data["choices"]
            ],
            created=response_data["created"],
            model=response_data["model"],
            provider=self.provider,
            usage=Usage(
                completion_tokens=response_data.get("usage", {}).get("completion_tokens", 0),
                prompt_tokens=response_data.get("usage", {}).get("prompt_tokens", 0),
                total_tokens=response_data.get("usage", {}).get("total_tokens", 0),
            ),
        )

    def _normalize_chunk(self, chunk_data: Dict[str, Any]) -> ChatCompletionChunk:
        """Normalize Mistral stream chunk to our format."""
        return ChatCompletionChunk(
            id=chunk_data["id"],
            choices=[
                StreamChoice(
                    index=choice["index"],
                    delta=DeltaMessage(
                        content=choice.get("delta", {}).get("content", ""),
                        role=choice.get("delta", {}).get("role", "assistant"),
                        function_call=choice.get("delta", {}).get("function_call"),
                        tool_calls=choice.get("delta", {}).get("tool_calls"),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
                for choice in chunk_data["choices"]
            ],
            created=chunk_data["created"],
            model=chunk_data.get("model", ""),
        )

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse Server-Sent Events stream from Mistral chat completions."""
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
        """Parse Server-Sent Events stream from Mistral chat completions asynchronously."""
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

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for Mistral API calls."""
        kwargs = {}
        config = self.get_completion_kwargs() 
        
        supported_params = ["temperature", "top_p", "max_tokens", "safe_prompt", "random_seed"]

        for key, value in config.items():
            if key in supported_params and value is not None:
                kwargs[key] = value
        
        # Handle streaming parameter
        if exclude_stream:
            kwargs.pop("streaming", None)
        elif "streaming" in config:
            kwargs["stream"] = config["streaming"]
        
        if self.structured and isinstance(self.structured, dict):
            if self.structured.get("type") == "json_object" or self.structured.get("type") == "json":
                 kwargs["response_format"] = {"type": "json_object"}

        return kwargs


    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request."""
        should_stream = stream if stream is not None else self.streaming
        
        # Prepare request payload
        payload = {
            "model": self.get_model_name(),
            "messages": messages,
            "stream": should_stream,
            **self._get_api_kwargs(exclude_stream=True),
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            return (self._normalize_chunk(chunk_data) for chunk_data in self._parse_sse_stream(response))
        
        response_data = response.json()
        return self._normalize_response(response_data)

    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request."""
        should_stream = stream if stream is not None else self.streaming
        
        # Prepare request payload
        payload = {
            "model": self.get_model_name(),
            "messages": messages,
            "stream": should_stream,
            **self._get_api_kwargs(exclude_stream=True),
        }

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            async def generate():
                async for chunk_data in self._parse_sse_stream_async(response):
                    yield self._normalize_chunk(chunk_data)

            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "mistral-large-latest"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "mistral"

    def to_langchain(self) -> "ChatMistralAI":
        """Convert to a LangChain ChatMistralAI model."""
        try:
            from langchain_mistralai import ChatMistralAI
        except ImportError:
            raise ImportError(
                "langchain_mistralai package not found. "
                "Install with: uv add langchain_mistralai or pip install langchain_mistralai"
            )

        lc_kwargs = {
            "mistral_api_key": self.api_key,
            "model": self.get_model_name(), 
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.base_url: 
            lc_kwargs["endpoint"] = self.base_url

        lc_kwargs = {k: v for k, v in lc_kwargs.items() if v is not None}
        
        return ChatMistralAI(**lc_kwargs)
