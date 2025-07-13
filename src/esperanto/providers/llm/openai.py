"""OpenAI language model provider."""

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
    from langchain_openai import ChatOpenAI


class OpenAILanguageModel(LanguageModel):
    """OpenAI language model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Get API key
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        # Set base URL
        self.base_url = self.base_url or "https://api.openai.com/v1"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if hasattr(self, 'organization') and self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI API error: {error_message}")

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        response = self.client.get(
            f"{self.base_url}/models",
            headers=self._get_headers()
        )
        self._handle_error(response)
        
        models_data = response.json()
        return [
            Model(
                id=model["id"],
                owned_by=model.get("owned_by", "openai"),
                context_window=model.get("context_window", None),
                type="language",
            )
            for model in models_data["data"]
            if model["id"].startswith(
                ("gpt-")
            )  # Only include GPT models for language tasks
        ]

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize OpenAI response to our format."""
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
        """Normalize OpenAI stream chunk to our format."""
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
        """Parse Server-Sent Events stream from OpenAI chat completions."""
        # For streaming responses, we need to iterate over the text content
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
        """Parse Server-Sent Events stream from OpenAI chat completions asynchronously."""
        # For async streaming responses, we need to iterate over the text content
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

    def _transform_messages_for_o1(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Transform messages for o1 models by replacing system role with user role."""
        return [
            {**msg, "role": "user"} if msg["role"] == "system" else {**msg}
            for msg in messages
        ]

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args.

        Args:
            exclude_stream: If True, excludes streaming-related parameters.
        """
        kwargs = {}
        config = self.get_completion_kwargs()
        is_reasoning_model = self._is_reasoning_model()

        # Only include non-provider-specific args that were explicitly set
        for key, value in config.items():
            if key not in [
                "model_name",
                "api_key",
                "base_url",
                "organization",
                "structured",
            ]:
                # Skip max_tokens if it's the default value (850) and we're using an o1 model
                if key == "max_tokens" and value == 850 and is_reasoning_model:
                    continue
                kwargs[key] = value

        # Special handling for o1 models
        if is_reasoning_model:
            # Replace max_tokens with max_completion_tokens
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)

        # Handle streaming parameter
        if exclude_stream:
            kwargs.pop("streaming", None)
        elif "streaming" in kwargs:
            kwargs["stream"] = kwargs.pop("streaming")

        # Handle structured output
        if self.structured:
            if not isinstance(self.structured, dict):
                raise TypeError("structured parameter must be a dictionary")
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                kwargs["response_format"] = {"type": "json_object"}

        return kwargs

    def _is_reasoning_model(self) -> bool:
        return self.get_model_name().startswith("o1") or self.get_model_name().startswith("o3") or self.get_model_name().startswith("o4")
    
    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation.
            stream: Whether to stream the response. If None, uses the instance's streaming setting.

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks if streaming.
        """
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        is_reasoning_model = self._is_reasoning_model()
        
        # Transform messages for o1 models
        if is_reasoning_model:
            messages = self._transform_messages_for_o1(
                [{**msg} for msg in messages]
            )  # Deep copy each message dict

        # Prepare request payload
        payload = {
            "model": model_name,
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
        """Send an async chat completion request.

        Args:
            messages: List of messages in the conversation.
            stream: Whether to stream the response. If None, uses the instance's streaming setting.

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks if streaming.
        """
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        is_reasoning_model = self._is_reasoning_model()
        
        # Transform messages for o1 models
        if is_reasoning_model:
            messages = self._transform_messages_for_o1(
                [{**msg} for msg in messages]
            )  # Deep copy each message dict

        # Prepare request payload
        payload = {
            "model": model_name,
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
        return "gpt-4o-mini"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai"

    def to_langchain(self) -> "ChatOpenAI":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_openai is not installed.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_openai. "
                "Install with: uv add langchain_openai or pip install langchain_openai"
            ) from e

        model_kwargs = {}
        if self.structured == "json":
            model_kwargs["response_format"] = {"type": "json_object"}

        langchain_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "organization": self.organization,
            "model": self.get_model_name(),
            "model_kwargs": model_kwargs,
        }

        is_reasoning_model = self._is_reasoning_model()

        if is_reasoning_model:
            # Replace max_tokens with max_completion_tokens
            if "max_tokens" in langchain_kwargs:
                langchain_kwargs["max_completion_tokens"] = langchain_kwargs.pop(
                    "max_tokens"
                )
            langchain_kwargs["temperature"] = 1
            langchain_kwargs["top_p"] = None

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
