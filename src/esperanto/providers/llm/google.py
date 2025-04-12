"""Google GenAI language model provider."""

import datetime
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

from google import genai
from google.genai import types
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
    from langchain_core.language_models.chat_models import BaseChatModel


class GoogleLanguageModel(LanguageModel):
    """Google GenAI language model implementation."""

    def __post_init__(self):
        """Initialize Google client."""
        super().__post_init__()

        # Get API key
        self.api_key = (
            self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY environment variable."
            )

        # Initialize model
        self.model_name = self.model_name or self._get_default_model()
        self._client = genai.Client(api_key=self.api_key)
        self._langchain_model = None

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        models_list = genai.list_models()
        return [
            Model(
                id=model.name.split("/")[-1],
                owned_by="Google",
                context_window=(
                    model.input_token_limit
                    if hasattr(model, "input_token_limit")
                    else None
                ),
                type="language",
            )
            for model in models_list
            if "generateContent"
            in model.supported_generation_methods  # Only include text generation models
        ]

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "google"

    def _get_default_model(self) -> str:
        """Get the default model name.

        Returns:
            str: The default model name.
        """
        return "gemini-1.5-pro"

    def to_langchain(self) -> "BaseChatModel":
        """Convert to a LangChain chat model.

        Returns:
            BaseChatModel: A LangChain chat model instance specific to the provider.

        Raises:
            ImportError: If langchain_google_genai is not installed.
        """
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_google_genai. "
                "Install with: uv add esperanto[google,langchain] or pip install esperanto[google,langchain]"
            ) from e

        if not self._langchain_model:
            # Ensure model name is a string
            model_name = self.get_model_name()
            if not model_name:
                raise ValueError("Model name must be set to use Langchain integration.")

            self._langchain_model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                # Removed streaming and google_api_key as they cause errors
            )
        return self._langchain_model

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single string for Google.

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
        """Create generation config for Google."""
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens if self.max_tokens else None,
        )

        if self.structured:
            if not isinstance(self.structured, dict):
                raise TypeError("structured parameter must be a dictionary")
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
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

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=formatted_messages,
            config=self._create_generation_config(),
        )

        print(response)
        if not response.text:
            raise ValueError("Empty response from Google API")

        return ChatCompletion(
            id=f"google-{str(hash(formatted_messages))}",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response.text),
                    finish_reason=response.prompt_feedback.block_reason or "stop",
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
            model=self.model_name or self._get_default_model(),
            provider=self.provider,
            usage=Usage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
        )

    def _stream_response(
        self, formatted_messages: str
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Stream response from Google.

        Args:
            formatted_messages: Formatted string of messages

        Returns:
            ChatCompletionChunk objects
        """
        response_stream = self._client.models.generate_content(
            model=self.model_name,
            contents=formatted_messages,
            config=self._create_generation_config(),
            stream=True,
        )

        for chunk in response_stream:
            if not chunk.text:  # Skip empty chunks
                continue

            yield ChatCompletionChunk(
                id=f"google-chunk-{str(hash(formatted_messages))}",
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant", content=chunk.text),
                        finish_reason=None,
                    )
                ],
                model=self.model_name or self._get_default_model(),
                created=int(datetime.datetime.now().timestamp()),
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
                response_stream = await self._client.models.generate_content_async(
                    model=self.model_name,
                    contents=formatted_messages,
                    config=self._create_generation_config(),
                    stream=True,
                )
                async for chunk in response_stream:
                    if not chunk.text:  # Skip empty chunks
                        continue

                    yield ChatCompletionChunk(
                        id=f"google-chunk-{str(hash(formatted_messages))}",
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    role="assistant", content=chunk.text
                                ),
                                finish_reason=None,
                            )
                        ],
                        model=self.model_name,
                        created=int(datetime.datetime.now().timestamp()),
                    )

            return astream_response()

        response = await self._client.models.generate_content_async(
            model=self.model_name,
            contents=formatted_messages,
            config=self._create_generation_config(),
            stream=False,
        )

        # Get the first candidate's content
        candidate = response.candidates[0]
        text = candidate.content.parts[0].text.strip()

        # Map Google's STOP to our stop finish reason
        finish_reason = (
            "stop" if candidate.finish_reason == "STOP" else candidate.finish_reason
        )

        return ChatCompletion(
            id="google-" + str(hash(formatted_messages)),
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            model=self.model_name or self._get_default_model(),
            provider=self.provider,
        )
