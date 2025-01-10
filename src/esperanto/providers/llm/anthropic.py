"""Anthropic language model implementation."""
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message as AnthropicMessage
from langchain_anthropic import ChatAnthropic

from esperanto.providers.llm.base import LanguageModel
from esperanto.types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Message,
    Model,
    StreamChoice,
    Usage,
)
from esperanto.utils.logging import logger


@dataclass
class AnthropicLanguageModel(LanguageModel):
    """Anthropic language model implementation."""

    def __post_init__(self):
        """Initialize Anthropic client."""
        super().__post_init__()
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable.")
        
        config = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        config = self._clean_config(config)
        
        self.client = Anthropic(**config)
        self.async_client = AsyncAnthropic(**config)

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        available_models = [
            Model(
                id="claude-3-opus-20240229",
                owned_by="Anthropic",
                context_window=200000,
                type="language"
            ),
            Model(
                id="claude-3-sonnet-20240229",
                owned_by="Anthropic",
                context_window=200000,
                type="language"
            ),
            Model(
                id="claude-3-haiku-20240307",
                owned_by="Anthropic",
                context_window=200000,
                type="language"
            ),
            Model(
                id="claude-2.1",
                owned_by="Anthropic",
                context_window=200000,
                type="language"
            ),
            Model(
                id="claude-2.0",
                owned_by="Anthropic",
                context_window=100000,
                type="language"
            ),
            Model(
                id="claude-instant-1.2",
                owned_by="Anthropic",
                context_window=100000,
                type="language"
            ),
        ]
        return available_models

    def _prepare_messages(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[Dict[str, str]]]:
        """Handle Anthropic-specific message preparation."""
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = None
            
        formatted_messages = [
            {
                "role": "assistant" if msg["role"] == "assistant" else "user",
                "content": msg["content"]
            }
            for msg in messages
        ]
        return system_message, formatted_messages

    def _normalize_response(self, response: AnthropicMessage) -> ChatCompletion:
        """Normalize Anthropic response to our format."""
        # Extract timestamp from message ID (msg_01L8KAo2eC4nf2HUCXNF9SWb)
        # Using a simple timestamp as fallback since message ID format might change
        created = int(time.time())
        
        return ChatCompletion(
            id=response.id,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        content=response.content[0].text,
                        role="assistant",
                    ),
                    finish_reason=response.stop_reason or "stop",
                )
            ],
            created=created,
            model=response.model,
            provider=self.provider,
            usage=Usage(
                completion_tokens=response.usage.output_tokens,
                prompt_tokens=response.usage.input_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
        )

    def _normalize_stream_event(self, event: Any) -> Optional[ChatCompletionChunk]:
        """Normalize Anthropic stream event to our format."""
        # Handle content delta events
        if event.type == "content_block_delta" and hasattr(event.delta, "text"):
            return ChatCompletionChunk(
                id=str(event.index),  # Using index as a temporary ID
                choices=[
                    StreamChoice(
                        index=0,
                        delta={
                            "content": event.delta.text,
                            "role": "assistant",
                        },
                        finish_reason=None,
                    )
                ],
                created=0,  # Not available in delta events
                model="",  # Not available in delta events
            )
        
        # Handle message completion event
        elif event.type == "message_delta":
            return ChatCompletionChunk(
                id="message_complete",
                choices=[
                    StreamChoice(
                        index=0,
                        delta={
                            "content": None,
                            "role": "assistant",
                        },
                        finish_reason=event.delta.stop_reason or "stop",
                    )
                ],
                created=0,
                model="",
            )
        
        # Ignore other event types
        return None

    def _prepare_api_kwargs(self, **kwargs) -> dict:
        """Prepare kwargs for API call."""
        kwargs = super()._prepare_api_kwargs(**kwargs)
        
        # Handle temperature - Anthropic expects 0-1 range
        if "temperature" in kwargs:
            temp = kwargs["temperature"]
            if temp is not None:
                kwargs["temperature"] = max(0.0, min(1.0, float(temp)))

        # Handle max_tokens
        if "max_tokens" in kwargs:
            max_tokens = kwargs["max_tokens"]
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)

        return kwargs

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
        return "claude-3-opus-20240229"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "anthropic"

    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean configuration dictionary."""
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        # List of supported parameters for the Anthropic client
        supported_params = {
            "api_key",
            "base_url",
            "timeout",
            "max_retries",
            "default_headers",
            "default_query",
        }
        
        # Remove unsupported parameters
        return {k: v for k, v in config.items() if k in supported_params}

    def chat_complete(
        self, 
        messages: List[Dict[str, str]], 
        stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request."""
        if self.structured:
            logger.warning("Structured output not supported for Anthropic.")
        
        should_stream = stream if stream is not None else self.streaming
        system_message, messages = self._prepare_messages(messages)
        
        response = self.client.messages.create(
            model=self.get_model_name(),
            system=system_message,
            messages=messages,
            stream=should_stream,
            **self._get_api_kwargs(exclude_stream=True)
        )
        
        if should_stream:
            def generate():
                for event in response:
                    chunk = self._normalize_stream_event(event)
                    if chunk:
                        yield chunk
            return generate()
        return self._normalize_response(response)

    async def achat_complete(
        self, 
        messages: List[Dict[str, str]], 
        stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request."""
        if self.structured:
            logger.warning("Structured output not supported for Anthropic.")
        
        should_stream = stream if stream is not None else self.streaming
        system_message, messages = self._prepare_messages(messages)
        
        response = await self.async_client.messages.create(
            model=self.get_model_name(),
            system=system_message,
            messages=messages,
            stream=should_stream,
            **self._get_api_kwargs(exclude_stream=True)
        )
        
        if should_stream:
            async def generate():
                async for event in response:
                    chunk = self._normalize_stream_event(event)
                    if chunk:
                        yield chunk
            return generate()
        return self._normalize_response(response)

    def to_langchain(self) -> ChatAnthropic:
        """Convert to a LangChain chat model."""
        if self.structured:
            logger.warning("Structured output not supported for Anthropic.")
        
        # Only include supported parameters
        supported_params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "model": self.get_model_name(),
            "anthropic_api_key": self.api_key,
        }

        if self.base_url:
            supported_params["anthropic_api_url"] = self.base_url

        # Remove None values
        langchain_kwargs = {k: v for k, v in supported_params.items() if v is not None}

        return ChatAnthropic(**langchain_kwargs)
