"""Mistral language model provider."""

import os
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)

from mistralai import Mistral
from mistralai.models import (
    AssistantMessage,
    ChatCompletionResponse as MistralChatCompletion,
    CompletionChunk as MistralChatCompletionChunk,
    SystemMessage,
    UserMessage,
)
if TYPE_CHECKING:  # Grouped with mistralai/langchain
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
        """Initialize Mistral client."""
        if self.model_name is None:
            self.model_name = self._get_default_model()
        super().__post_init__()

        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")

        self.client = Mistral(api_key=self.api_key)
        # Mistral Python client does not have a separate async client,
        # the same client handles both sync and async operations.
        self.async_client = self.client # For consistency


    def _get_default_model(self) -> str:
        """Get the default model name for Mistral."""
        return MISTRAL_DEFAULT_MODEL_NAME

    @property
    def models(self) -> List[Model]:
        """List available Mistral models."""
        try:
            client_models_response = self.client.models.list()
            # Note: The model object from client.models.list() might not have 'context_window'.
            # This information might need to be supplemented or managed differently if crucial.
            return [
                Model(
                    id=model.id,
                    owned_by=model.owned_by if hasattr(model, 'owned_by') else 'mistralai',
                    context_window=None,  # Placeholder, as context_window is not directly in model object
                    type="language",
                )
                for model in client_models_response.data
                if "embed" not in model.id  # Filter out embedding models
            ]
        except Exception as e:
            print(f"Warning: Could not dynamically list models from Mistral API: {e}. Falling back to a known list.")
            # Fallback to a known list if API call fails or if dynamic listing is not preferred
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

    def _normalize_response(self, response: MistralChatCompletion) -> ChatCompletion:
        """Normalize Mistral response to our format."""
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
                completion_tokens=(
                    response.usage.completion_tokens if response.usage else 0
                ),
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
        )

    def _normalize_chunk(self, chunk: MistralChatCompletionChunk) -> ChatCompletionChunk:
        """Normalize Mistral stream chunk to our format."""
        delta_content = ""
        delta_role = "assistant"  # Default role for stream delta
        finish_reason = None
        tool_calls = None # Placeholder for future tool call support

        # Accessing data as per MIGRATION.MD: chunk.data.choices[0].delta.content
        choice_data = None
        if chunk.data and chunk.data.choices:
            choice_data = chunk.data.choices[0]
            if choice_data.delta:
                delta_content = choice_data.delta.content or ""
                if choice_data.delta.role:
                    delta_role = choice_data.delta.role
                # Placeholder for tool calls if they appear in delta
                # if hasattr(choice_data.delta, 'tool_calls') and choice_data.delta.tool_calls:
                #     tool_calls = [dict(tc.model_dump()) for tc in choice_data.delta.tool_calls]

            if choice_data.finish_reason:
                finish_reason = choice_data.finish_reason
        
        choice_index = choice_data.index if choice_data else 0

        # Attempt to get id and model from chunk.data if available, otherwise use None or a default.
        # The CompletionEvent itself (chunk) might not have id/model directly.
        # These are often part of the ChatCompletionResponse which chunk.data might resemble.
        chunk_id = getattr(chunk.data, 'id', None) if hasattr(chunk, 'data') else getattr(chunk, 'id', None)
        chunk_model = getattr(chunk.data, 'model', None) if hasattr(chunk, 'data') else getattr(chunk, 'model', None)
        # If chunk.created is not available, we might need to omit it or use a default
        chunk_created = getattr(chunk.data, 'created', None) if hasattr(chunk, 'data') else getattr(chunk, 'created', None)

        return ChatCompletionChunk(
            id=chunk_id,
            choices=[
                StreamChoice(
                    index=choice_index,
                    delta=DeltaMessage(
                        content=delta_content,
                        role=delta_role,
                        function_call=None,  # Mistral doesn't use 'function_call'
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            created=chunk_created,
            model=chunk_model,
        )

    def _get_api_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for Mistral API calls."""
        kwargs = {}
        config = self.get_completion_kwargs() 
        
        supported_params = ["temperature", "top_p", "max_tokens", "safe_prompt", "random_seed"]

        for key, value in config.items():
            if key in supported_params and value is not None:
                kwargs[key] = value
            elif key == "streaming": # map streaming to stream for Mistral
                kwargs["stream"] = value
        
        if self.structured and isinstance(self.structured, dict):
            if self.structured.get("type") == "json_object" or self.structured.get("type") == "json":
                 kwargs["response_format"] = {"type": "json_object"}

        return kwargs

    def _convert_messages_to_mistral_format(
        self, messages: List[Dict[str, str]]
    ) -> List[Union[UserMessage, AssistantMessage, SystemMessage]]:
        mistral_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                mistral_messages.append(UserMessage(content=content))
            elif role == "assistant":
                mistral_messages.append(AssistantMessage(content=content))
            elif role == "system":
                mistral_messages.append(SystemMessage(content=content))
            # TODO: Add support for 'tool' role if ToolMessage is fully integrated
            else:
                # Consider raising ValueError for unsupported roles for stricter error handling
                print(f"Warning: Unsupported message role '{role}' encountered. Message content: '{content}'")
        return mistral_messages

    def chat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request."""
        actual_stream = stream if stream is not None else self.streaming
        api_kwargs = self._get_api_kwargs()
        # Override stream based on actual_stream for the current call
        api_kwargs["stream"] = actual_stream
        
        mistral_messages = self._convert_messages_to_mistral_format(messages)

        if actual_stream:
            response_generator = self.client.chat.stream(
                model=self.get_model_name(),
                messages=mistral_messages,
                **{k: v for k, v in api_kwargs.items() if k != 'stream'}
            )
            return (self._normalize_chunk(chunk) for chunk in response_generator)
        else:
            response = self.client.chat.complete(
                model=self.get_model_name(),
                messages=mistral_messages,
                **{k: v for k, v in api_kwargs.items() if k != 'stream'}
            )
            return self._normalize_response(response)

    async def achat_complete(
        self, messages: List[Dict[str, str]], stream: Optional[bool] = None
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request."""
        actual_stream = stream if stream is not None else self.streaming
        api_kwargs = self._get_api_kwargs()
        api_kwargs["stream"] = actual_stream

        mistral_messages = self._convert_messages_to_mistral_format(messages)

        if actual_stream:
            # Correctly await the stream_async method to get the async generator
            async_response_generator = await self.async_client.chat.stream_async(
                model=self.get_model_name(),
                messages=mistral_messages,
                **{k: v for k, v in api_kwargs.items() if k != 'stream'}
            )
            async def normalize_async_chunks():
                # Iterate over the actual async generator
                async for chunk in async_response_generator:
                    yield self._normalize_chunk(chunk)
            return normalize_async_chunks()
        else:
            response = await self.async_client.chat.complete_async(
                model=self.get_model_name(),
                messages=mistral_messages,
                **{k: v for k, v in api_kwargs.items() if k != 'stream'}
            )
            return self._normalize_response(response)

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
                "Install with: uv add esperanto[mistral,langchain] or pip install esperanto[mistral,langchain]"
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
