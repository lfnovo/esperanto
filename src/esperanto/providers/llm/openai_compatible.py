"""OpenAI-compatible language model implementation."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from esperanto.common_types import Model
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@dataclass
class OpenAICompatibleLanguageModel(OpenAILanguageModel):
    """OpenAI-compatible language model implementation for custom endpoints."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        """Initialize OpenAI-compatible configuration."""
        # Initialize _config first (from base class)
        if not hasattr(self, '_config'):
            self._config = {}
        
        # Update with any provided config
        if hasattr(self, "config") and self.config:
            self._config.update(self.config)
        
        # Configuration precedence: Factory config > Environment variables > Default
        self.base_url = (
            self.base_url or 
            self._config.get("base_url") or 
            os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        )
        self.api_key = (
            self.api_key or 
            self._config.get("api_key") or 
            os.getenv("OPENAI_COMPATIBLE_API_KEY")
        )

        # Validation
        if not self.base_url:
            raise ValueError(
                "OpenAI-compatible base URL is required. "
                "Set OPENAI_COMPATIBLE_BASE_URL environment variable or provide base_url in config."
            )
        # Use a default API key if none is provided (some endpoints don't require authentication)
        if not self.api_key:
            self.api_key = "not-required"

        # Ensure base_url doesn't end with trailing slash for consistency
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

        # Call parent's post_init to set up HTTP clients and normalized response handling
        super().__post_init__()

    def _handle_error(self, response) -> None:
        """Handle HTTP error responses with graceful degradation."""
        if response.status_code >= 400:
            # Log original response for debugging
            logger.debug(f"OpenAI-compatible endpoint error: {response.text}")
            
            # Try to parse OpenAI-format error
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                # Fall back to HTTP status code
                error_message = f"HTTP {response.status_code}: {response.text}"
            
            raise RuntimeError(f"OpenAI-compatible endpoint error: {error_message}")
    
    def _normalize_response(self, response_data: Dict[str, Any]) -> "ChatCompletion":
        """Normalize OpenAI-compatible response to our format with graceful fallback."""
        from esperanto.common_types import ChatCompletion, Choice, Message, Usage
        
        # Handle missing or incomplete response fields gracefully
        response_id = response_data.get("id", "chatcmpl-unknown")
        created = response_data.get("created", 0)
        model = response_data.get("model", self.get_model_name())
        
        # Handle choices array
        choices = response_data.get("choices", [])
        normalized_choices = []
        
        for choice in choices:
            message = choice.get("message", {})
            normalized_choice = Choice(
                index=choice.get("index", 0),
                message=Message(
                    content=message.get("content", ""),
                    role=message.get("role", "assistant"),
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )
            normalized_choices.append(normalized_choice)
        
        # If no choices, create a default one
        if not normalized_choices:
            normalized_choices = [Choice(
                index=0,
                message=Message(content="", role="assistant"),
                finish_reason="stop"
            )]
        
        # Handle usage information
        usage_data = response_data.get("usage", {})
        usage = Usage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        
        return ChatCompletion(
            id=response_id,
            choices=normalized_choices,
            created=created,
            model=model,
            provider=self.provider,
            usage=usage,
        )

    def _normalize_chunk(self, chunk_data: Dict[str, Any]) -> "ChatCompletionChunk":
        """Normalize OpenAI-compatible stream chunk to our format with graceful fallback."""
        from esperanto.common_types import ChatCompletionChunk, StreamChoice, DeltaMessage
        
        # Handle missing or incomplete chunk fields gracefully
        chunk_id = chunk_data.get("id", "chatcmpl-unknown")
        created = chunk_data.get("created", 0)
        model = chunk_data.get("model", self.get_model_name())
        
        # Handle choices array
        choices = chunk_data.get("choices", [])
        normalized_choices = []
        
        for choice in choices:
            delta = choice.get("delta", {})
            normalized_choice = StreamChoice(
                index=choice.get("index", 0),
                delta=DeltaMessage(
                    content=delta.get("content", ""),
                    role=delta.get("role", "assistant"),
                    function_call=delta.get("function_call"),
                    tool_calls=delta.get("tool_calls"),
                ),
                finish_reason=choice.get("finish_reason"),
            )
            normalized_choices.append(normalized_choice)
        
        # If no choices, create a default one
        if not normalized_choices:
            normalized_choices = [StreamChoice(
                index=0,
                delta=DeltaMessage(content="", role="assistant"),
                finish_reason=None
            )]
        
        return ChatCompletionChunk(
            id=chunk_id,
            choices=normalized_choices,
            created=created,
            model=model,
        )

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get API kwargs with graceful feature fallback.
        
        Args:
            exclude_stream: If True, excludes streaming-related parameters.
            
        Returns:
            Dict containing API parameters for the request.
        """
        # Get base kwargs from parent
        kwargs = super()._get_api_kwargs(exclude_stream)
        
        # For OpenAI-compatible endpoints, we attempt all features
        # and let the endpoint handle graceful degradation
        # This includes streaming, JSON mode, and other OpenAI features
        
        return kwargs

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.
        
        Note: This attempts to fetch models from the /models endpoint.
        If the endpoint doesn't support this, it will return an empty list.
        """
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
                    owned_by=model.get("owned_by", "custom"),
                    context_window=model.get("context_window", None),
                    type="language",
                )
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            # Log the error but don't fail completely
            logger.debug(f"Could not fetch models from OpenAI-compatible endpoint: {e}")
            return []

    def _get_default_model(self) -> str:
        """Get the default model name.
        
        For OpenAI-compatible endpoints, we use a generic default
        that users should override with their specific model.
        """
        return "gpt-3.5-turbo"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "openai-compatible"

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
            if structured_type in ["json", "json_object"]:
                model_kwargs["response_format"] = {"type": "json_object"}

        langchain_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.get_model_name(),
            "model_kwargs": model_kwargs,
        }

        # Handle reasoning models (o1, o3, o4)
        is_reasoning_model = self._is_reasoning_model()
        if is_reasoning_model:
            # Replace max_tokens with max_completion_tokens
            if "max_tokens" in langchain_kwargs:
                langchain_kwargs["max_completion_tokens"] = langchain_kwargs.pop("max_tokens")
            langchain_kwargs["temperature"] = 1
            langchain_kwargs["top_p"] = None

        return ChatOpenAI(**self._clean_config(langchain_kwargs))