"""OpenRouter language model implementation."""

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # Added Optional

from esperanto.common_types import Model
from esperanto.providers.llm.openai import OpenAILanguageModel

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@dataclass
class OpenRouterLanguageModel(OpenAILanguageModel):
    """OpenRouter language model implementation using OpenAI-compatible API."""

    base_url: Optional[str] = None  # Changed type hint
    api_key: Optional[str] = None  # Changed type hint

    def __post_init__(self):
        # Initialize OpenRouter-specific configuration
        self.base_url = self.base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable."
            )

        # Call parent's post_init to set up HTTP clients
        super().__post_init__()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests with required headers."""
        headers = super()._get_headers()
        # Add OpenRouter-specific required headers
        headers.update({
            "HTTP-Referer": "https://github.com/lfnovo/esperanto",
            "X-Title": "Esperanto",
        })
        return headers

    def _handle_error(self, response) -> None:
        """Handle HTTP error responses with detailed OpenRouter logging."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception as e:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI API error: {error_message}")

    def _make_http_request(self, payload: Dict[str, Any]) -> Any:
        """Make HTTP request in OpenRouter's expected format."""
        # OpenRouter expects data as JSON string, not json parameter
        headers = self._get_headers()
        
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload)  # Use data= instead of json=
        )
        self._handle_error(response)
        return response

    async def _make_async_http_request(self, payload: Dict[str, Any]) -> Any:
        """Make async HTTP request in OpenRouter's expected format."""
        # OpenRouter expects data as JSON string, not json parameter
        response = await self.async_client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            data=json.dumps(payload)  # Use data= instead of json=
        )
        self._handle_error(response)
        return response

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args.

        Note: OpenRouter doesn't support JSON response format for non-OpenAI models.
        """
        kwargs = super()._get_api_kwargs(exclude_stream)

        # Remove response_format for non-OpenAI models
        model = self.get_model_name().lower()
        if "response_format" in kwargs and not model.startswith(("openai/", "gpt-")):
            kwargs.pop("response_format")

        return kwargs

    def chat_complete(self, messages, stream=None):
        """Override to use OpenRouter-specific HTTP format."""
        from typing import Generator, Union

        from esperanto.common_types import ChatCompletion, ChatCompletionChunk
        
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        is_reasoning_model = model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4")
        
        # Transform messages for o1 models
        if is_reasoning_model:
            messages = self._transform_messages_for_o1(
                [{**msg} for msg in messages]
            )

        # Prepare request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": should_stream,
            **self._get_api_kwargs(exclude_stream=True),
        }

        # Make HTTP request using OpenRouter format
        response = self._make_http_request(payload)

        if should_stream:
            return (self._normalize_chunk(chunk_data) for chunk_data in self._parse_sse_stream(response))
        
        response_data = response.json()
        return self._normalize_response(response_data)

    async def achat_complete(self, messages, stream=None):
        """Override to use OpenRouter-specific async HTTP format."""
        from typing import AsyncGenerator, Union

        from esperanto.common_types import ChatCompletion, ChatCompletionChunk
        
        should_stream = stream if stream is not None else self.streaming
        model_name = self.get_model_name()
        is_reasoning_model = model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4")
        
        # Transform messages for o1 models
        if is_reasoning_model:
            messages = self._transform_messages_for_o1(
                [{**msg} for msg in messages]
            )

        # Prepare request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": should_stream,
            **self._get_api_kwargs(exclude_stream=True),
        }

        # Make async HTTP request using OpenRouter format
        response = await self._make_async_http_request(payload)

        if should_stream:
            async def generate():
                async for chunk_data in self._parse_sse_stream_async(response):
                    yield self._normalize_chunk(chunk_data)

            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "anthropic/claude-2"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openrouter"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        headers = self._get_headers()
        
        response = self.client.get(
            f"{self.base_url}/models",
            headers=headers
        )
        self._handle_error(response)
        
        models_data = response.json()
        return [
            Model(
                id=model["id"],
                owned_by=model["id"].split("/")[0] if "/" in model["id"] else "OpenRouter",
                context_window=model.get("context_window", None),
                type="language",
            )
            for model in models_data["data"]
            if not any(
                model["id"].startswith(prefix)
                for prefix in [
                    "text-embedding",  # Exclude embedding models
                    "whisper",  # Exclude speech models
                    "tts",  # Exclude text-to-speech models
                ]
            )
        ]

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
        if self.structured and isinstance(self.structured, dict):
            structured_type = self.structured.get("type")
            if structured_type in [
                "json",
                "json_object",
            ] and self.get_model_name().lower().startswith(("openai/", "gpt-")):
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
            "default_headers": {
                "HTTP-Referer": "https://github.com/lfnovo/esperanto",  # Required by OpenRouter
                "X-Title": "Esperanto",  # Required by OpenRouter
            },
        }

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name is required for Langchain integration.")
        langchain_kwargs["model"] = model_name  # Update model name in kwargs

        return ChatOpenAI(**self._clean_config(langchain_kwargs))
