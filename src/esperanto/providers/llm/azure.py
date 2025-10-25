"""Azure OpenAI language model provider."""

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
from pydantic import SecretStr

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
    from langchain_openai import AzureChatOpenAI


class AzureLanguageModel(LanguageModel):
    """Azure OpenAI language model implementation using direct HTTP."""

    def __post_init__(self):
        """Initialize with Azure-specific configuration."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Resolve configuration with priority: config dict → modality env var → generic env var
        self.api_key = (
            self.api_key or
            self._config.get("api_key") or
            os.getenv("AZURE_OPENAI_API_KEY_LLM") or
            os.getenv("AZURE_OPENAI_API_KEY")
        )

        self.azure_endpoint = (
            self.base_url or
            self._config.get("azure_endpoint") or
            os.getenv("AZURE_OPENAI_ENDPOINT_LLM") or
            os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        self.api_version = (
            self._config.get("api_version") or
            os.getenv("AZURE_OPENAI_API_VERSION_LLM") or
            os.getenv("OPENAI_API_VERSION") or  # Backward compatibility
            os.getenv("AZURE_OPENAI_API_VERSION")
        )

        # deployment_name is model_name for Azure
        self.deployment_name = self.model_name or self._get_default_model()

        # Validate required parameters
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY_LLM "
                "or AZURE_OPENAI_API_KEY environment variable, or provide in config."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT_LLM "
                "or AZURE_OPENAI_ENDPOINT environment variable, or provide in config."
            )
        if not self.api_version:
            raise ValueError(
                "Azure OpenAI API version not found. Set AZURE_OPENAI_API_VERSION_LLM "
                "or AZURE_OPENAI_API_VERSION environment variable, or provide in config."
            )
        if not self.deployment_name:
            raise ValueError("Azure OpenAI deployment name (model_name) not found")

        # Initialize httpx clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Azure API requests."""
        return {
            "api-key": self.api_key,  # Azure uses api-key, not Bearer
            "Content-Type": "application/json",
        }

    def _build_url(self, path: str) -> str:
        """Build Azure OpenAI URL with deployment name."""
        # Remove trailing slash from endpoint
        endpoint = self.azure_endpoint.rstrip('/')
        # Azure URL pattern: {endpoint}/openai/deployments/{deployment}/{path}?api-version={version}
        return f"{endpoint}/openai/deployments/{self.deployment_name}/{path}?api-version={self.api_version}"

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Azure OpenAI API error: {error_message}")

    def _get_models(self) -> List[Model]:
        """List available models for this provider.

        Note: Azure doesn't have a models API endpoint - it uses deployments.
        Returns an empty list since model discovery isn't available.
        """
        return []

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Azure response to our format."""
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
        """Normalize Azure stream chunk to our format."""
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
        """Parse Server-Sent Events stream from Azure chat completions."""
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
        """Parse Server-Sent Events stream from Azure chat completions asynchronously."""
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

    def _is_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model (o1, o3, o4, gpt-5 series)."""
        model_name = self.deployment_name.lower()
        return (model_name.startswith("o1") or
                model_name.startswith("o3") or
                model_name.startswith("o4") or
                model_name.startswith("gpt-5"))

    def _get_api_kwargs(
        self, override_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get kwargs for Azure API calls, using current instance attributes and overrides."""
        is_reasoning_model = self._is_reasoning_model()

        effective_kwargs = {
            "model": self.deployment_name,
        }

        # Handle token parameters
        if is_reasoning_model:
            # Skip max_tokens if it's the default value (850) for reasoning models
            if self.max_tokens != 850:
                effective_kwargs["max_completion_tokens"] = self.max_tokens
        else:
            effective_kwargs["max_tokens"] = self.max_tokens

        # Handle temperature and top_p - reasoning models don't support these
        if not is_reasoning_model:
            effective_kwargs["temperature"] = self.temperature
            effective_kwargs["top_p"] = self.top_p

        effective_kwargs["stream"] = self.streaming

        if self.structured is not None:
            is_json_mode = False
            if isinstance(self.structured, dict):
                struct_type = self.structured.get("type")
                if struct_type == "json_object" or struct_type == "json":
                    is_json_mode = True
                else:
                    raise TypeError(
                        f"Invalid 'type' in structured_output dictionary: {struct_type}. Expected 'json' or 'json_object'."
                    )
            elif isinstance(self.structured, str):
                if self.structured == "json":
                    is_json_mode = True
                else:
                    raise TypeError(
                        f"Invalid string for structured_output: '{self.structured}'. Expected 'json'."
                    )
            else:
                raise TypeError(
                    f"Invalid type for structured_output: {type(self.structured)}. Expected dict or str 'json'."
                )

            if is_json_mode:
                effective_kwargs["response_format"] = {"type": "json_object"}

        if override_kwargs:
            effective_kwargs.update(override_kwargs)

        return {k: v for k, v in effective_kwargs.items() if v is not None}

    def _chat_complete_streaming(
        self, messages: List[Dict[str, str]], api_kwargs: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Handle streaming chat completion."""
        url = self._build_url("chat/completions")
        with self.client.stream(
            "POST",
            url,
            headers=self._get_headers(),
            json={"messages": messages, "stream": True, **api_kwargs},
        ) as response:
            self._handle_error(response)
            for chunk_data in self._parse_sse_stream(response):
                yield self._normalize_chunk(chunk_data)

    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request."""
        call_override_kwargs = {}
        if stream is not None:
            call_override_kwargs["stream"] = stream

        api_kwargs = self._get_api_kwargs(override_kwargs=call_override_kwargs)
        effective_stream_setting = api_kwargs.pop("stream", False)

        if effective_stream_setting:
            # Return streaming generator
            return self._chat_complete_streaming(messages, api_kwargs)
        else:
            # Non-streaming request
            url = self._build_url("chat/completions")
            response = self.client.post(
                url,
                headers=self._get_headers(),
                json={"messages": messages, "stream": False, **api_kwargs},
            )
            self._handle_error(response)
            return self._normalize_response(response.json())

    async def _achat_complete_streaming(
        self, messages: List[Dict[str, str]], api_kwargs: Dict[str, Any]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Handle async streaming chat completion."""
        url = self._build_url("chat/completions")
        async with self.async_client.stream(
            "POST",
            url,
            headers=self._get_headers(),
            json={"messages": messages, "stream": True, **api_kwargs},
        ) as response:
            self._handle_error(response)
            async for chunk_data in self._parse_sse_stream_async(response):
                yield self._normalize_chunk(chunk_data)

    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request."""
        call_override_kwargs = {}
        if stream is not None:
            call_override_kwargs["stream"] = stream

        api_kwargs = self._get_api_kwargs(override_kwargs=call_override_kwargs)
        effective_stream_setting = api_kwargs.pop("stream", False)

        if effective_stream_setting:
            # Return async streaming generator
            return self._achat_complete_streaming(messages, api_kwargs)
        else:
            # Non-streaming async request
            url = self._build_url("chat/completions")
            response = await self.async_client.post(
                url,
                headers=self._get_headers(),
                json={"messages": messages, "stream": False, **api_kwargs},
            )
            self._handle_error(response)
            return self._normalize_response(response.json())

    def to_langchain(
        self, **kwargs: Any
    ) -> "AzureChatOpenAI":
        """Convert to a LangChain chat model.

        Raises:
            ImportError: If langchain_openai is not installed.
        """
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise ImportError(
                "LangChain or langchain-openai not installed. "
                "Please install with `pip install langchain_openai`"
            ) from e

        model_kwargs = {}
        if self.structured is not None:
            # Handle different structured formats
            if isinstance(self.structured, dict):
                struct_type = self.structured.get("type")
                if struct_type == "json_object" or struct_type == "json":
                    model_kwargs["response_format"] = {"type": "json_object"}
            elif self.structured == "json":
                model_kwargs["response_format"] = {"type": "json_object"}

        is_reasoning_model = self._is_reasoning_model()

        langchain_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "api_key": SecretStr(self.api_key) if self.api_key else None,
            "azure_deployment": self.deployment_name,
            "api_version": self.api_version,
            "azure_endpoint": self.azure_endpoint,
            "model_kwargs": model_kwargs,
        }

        if is_reasoning_model:
            # For reasoning models, put max_completion_tokens in model_kwargs
            if self.max_tokens != 850:
                model_kwargs["max_completion_tokens"] = self.max_tokens
            langchain_kwargs["temperature"] = 1
            langchain_kwargs["top_p"] = None
        else:
            langchain_kwargs["max_tokens"] = self.max_tokens

        langchain_kwargs.update(kwargs)

        final_lc_kwargs = {k: v for k, v in langchain_kwargs.items() if v is not None or k == "api_key"}

        # Remove model_kwargs if it's empty and not explicitly passed
        if not final_lc_kwargs.get("model_kwargs") and "model_kwargs" not in kwargs:
            final_lc_kwargs.pop("model_kwargs", None)

        return AzureChatOpenAI(**self._clean_config(final_lc_kwargs))

    @property
    def provider(self) -> str:
        return "azure"

    def _get_default_model(self) -> str:
        """Get the default model name (deployment name for Azure).

        For Azure, model_name (deployment name) is required for actual usage.
        Returns empty string if no model is configured - the validation in __post_init__
        will catch this and raise an error.
        """
        return ""
