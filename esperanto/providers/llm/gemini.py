"""Gemini language model provider."""
import datetime
import os
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

import google.generativeai as genai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from esperanto.providers.llm.base import LanguageModel
from esperanto.types import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    DeltaMessage,
    Message,
    StreamChoice,
    Usage,
)


class GeminiLanguageModel(LanguageModel):
    """Gemini language model implementation."""

    def __post_init__(self):
        """Initialize Gemini client."""
        super().__post_init__()

        # Get API key
        self.api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model_name = self.model_name or self._get_default_model()
        self._client = genai.GenerativeModel(model_name=self.model_name)
        self._langchain_model = None

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "gemini"

    def _get_default_model(self) -> str:
        """Get the default model name.

        Returns:
            str: The default model name.
        """
        return "gemini-1.5-pro"

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model.
        
        Returns:
            BaseChatModel: A LangChain chat model instance specific to the provider.
        """
        if not self._langchain_model:
            self._langchain_model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                streaming=self.streaming,
                google_api_key=self.api_key,
            )
        return self._langchain_model

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single string for Gemini.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            Formatted string of messages
        """
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted)

    def _create_generation_config(self) -> Any:
        """Create generation config for Gemini."""
        config = genai.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens if self.max_tokens else None,
        )
        
        if self.structured == "json":
            config.response_mime_type = "application/json"
            
        return config

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation
            stream: Whether to stream the response

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks if streaming
        """
        formatted_messages = self._format_messages(messages)
        stream = stream if stream is not None else self.streaming

        if stream:
            return self._stream_response(formatted_messages)
        
        response = self._client.generate_content(
            contents=formatted_messages,
            generation_config=self._create_generation_config(),
        )

        if not response.text:
            raise ValueError("Empty response from Gemini API")

        return ChatCompletion(
            id=f"gemini-{str(hash(formatted_messages))}",
            choices=[Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=response.text
                ),
                finish_reason=response.prompt_feedback.block_reason or "stop"
            )],
            created=int(datetime.datetime.now().timestamp()),
            model=self.model_name,
            provider=self.provider,
            usage=Usage(
                completion_tokens=0,  # Gemini doesn't provide token counts
                prompt_tokens=0,
                total_tokens=0
            )
        )

    def _stream_response(
        self, formatted_messages: str
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Stream response from Gemini.

        Args:
            formatted_messages: Formatted string of messages

        Returns:
            ChatCompletionChunk objects
        """
        response_stream = self._client.generate_content(
            contents=formatted_messages,
            generation_config=self._create_generation_config(),
            stream=True
        )

        for chunk in response_stream:
            if not chunk.text:  # Skip empty chunks
                continue
                
            yield ChatCompletionChunk(
                id=f"gemini-chunk-{str(hash(formatted_messages))}",
                choices=[StreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role="assistant",
                        content=chunk.text
                    ),
                    finish_reason=None
                )],
                model=self.model_name,
                created=int(datetime.datetime.now().timestamp())
            )

    async def achat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request.

        Args:
            messages: List of messages in the conversation
            stream: Whether to stream the response

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks if streaming
        """
        formatted_messages = self._format_messages(messages)
        stream = stream if stream is not None else self.streaming

        if stream:
            async def astream_response():
                response_stream = await self._client.generate_content_async(
                    contents=formatted_messages,
                    generation_config=self._create_generation_config(),
                    stream=True
                )
                async for chunk in response_stream:
                    if not chunk.text:  # Skip empty chunks
                        continue
                        
                    yield ChatCompletionChunk(
                        id=f"gemini-chunk-{str(hash(formatted_messages))}",
                        choices=[StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                role="assistant",
                                content=chunk.text
                            ),
                            finish_reason=None
                        )],
                        model=self.model_name,
                        created=int(datetime.datetime.now().timestamp())
                    )
            return astream_response()

        response = await self._client.generate_content_async(
            contents=formatted_messages,
            generation_config=self._create_generation_config(),
            stream=False
        )

        # Get the first candidate's content
        candidate = response.candidates[0]
        text = candidate.content.parts[0].text.strip()

        # Map Gemini's STOP to our stop finish reason
        finish_reason = "stop" if candidate.finish_reason == "STOP" else candidate.finish_reason

        return ChatCompletion(
            id="gemini-" + str(hash(formatted_messages)),
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=text
                ),
                finish_reason=finish_reason
            )],
            model=self.model_name,
            provider=self.provider
        )
