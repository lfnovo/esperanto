"""Groq language model provider."""
import os
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from groq import AsyncGroq, Groq
from groq.types.chat import ChatCompletion as GroqChatCompletion
from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk
from langchain_groq import ChatGroq

from esperanto.providers.llm.base import LanguageModel
from esperanto.types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Message,
    StreamChoice,
    Usage,
)


class GroqLanguageModel(LanguageModel):
    """Groq language model implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get API key
        self.api_key = kwargs.get("api_key") or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found")
        
        # Update config with model_name if provided
        if "model_name" in kwargs:
            self._config["model_name"] = kwargs["model_name"]
        
        # Initialize clients
        self.client = Groq(
            api_key=self.api_key,
        )
        self.async_client = AsyncGroq(
            api_key=self.api_key,
        )

    def _normalize_response(self, response: GroqChatCompletion) -> ChatCompletion:
        """Normalize Groq response to our format."""
        return ChatCompletion(
            id=response.id,
            choices=[
                Choice(
                    index=choice.index,
                    message=Message(
                        content=choice.message.content or "",
                        role=choice.message.role,
                    ),
                    finish_reason=choice.finish_reason,
                )
                for choice in response.choices
            ],
            created=response.created,
            model=response.model,
            provider=self.provider,
            usage=Usage(
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    def _normalize_chunk(self, chunk: GroqChatCompletionChunk) -> ChatCompletionChunk:
        """Normalize Groq stream chunk to our format."""
        return ChatCompletionChunk(
            id=chunk.id,
            choices=[
                StreamChoice(
                    index=choice.index,
                    delta={
                        "content": choice.delta.content,
                        "role": choice.delta.role,
                        "function_call": choice.delta.function_call,
                        "tool_calls": choice.delta.tool_calls,
                    },
                    finish_reason=choice.finish_reason,
                )
                for choice in chunk.choices
            ],
            created=chunk.created,
            model=chunk.model,
        )

    def _get_api_kwargs(self, exclude_stream: bool = False) -> Dict[str, Any]:
        """Get kwargs for API calls, filtering out provider-specific args."""
        kwargs = self.get_completion_kwargs()
        
        # Remove provider-specific kwargs that Groq doesn't expect
        kwargs.pop("model_name", None)
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        kwargs.pop("organization", None)
        kwargs.pop("structured", None)
        
        # Handle streaming parameter
        if exclude_stream:
            kwargs.pop("streaming", None)
        elif "streaming" in kwargs:
            kwargs["stream"] = kwargs.pop("streaming")
            
        return kwargs

    def chat_complete(
        self, 
        messages: List[Dict[str, str]], 
        stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation.
            stream: Whether to stream the response. If None, uses the instance's streaming setting.

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks if streaming.
        """
        should_stream = stream if stream is not None else self.streaming
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.get_model_name(),
            stream=should_stream,
            **self._get_api_kwargs(exclude_stream=True)
        )
        
        if should_stream:
            return (self._normalize_chunk(chunk) for chunk in response)
        return self._normalize_response(response)

    async def achat_complete(
        self, 
        messages: List[Dict[str, str]], 
        stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request.

        Args:
            messages: List of messages in the conversation.
            stream: Whether to stream the response. If None, uses the instance's streaming setting.

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks if streaming.
        """
        should_stream = stream if stream is not None else self.streaming
        
        response = await self.async_client.chat.completions.create(
            messages=messages,
            model=self.get_model_name(),
            stream=should_stream,
            **self._get_api_kwargs(exclude_stream=True)
        )
        
        if should_stream:
            async def generate():
                async for chunk in response:
                    yield self._normalize_chunk(chunk)
            return generate()
        return self._normalize_response(response)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "mixtral-8x7b-32768"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "groq"

    def to_langchain(self) -> ChatGroq:
        """Convert to a LangChain chat model."""
        return ChatGroq(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            streaming=self.streaming,
            groq_api_key=self.api_key,
            model=self.get_model_name(),
        )
