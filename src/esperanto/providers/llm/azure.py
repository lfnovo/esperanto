"""Azure OpenAI language model provider."""

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

from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from openai.types.chat import ChatCompletionChunk as OpenAIChatCompletionChunk
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
    """Azure OpenAI language model implementation."""

    def __post_init__(self):
        """Initialize Azure OpenAI client."""
        super().__post_init__()

        self.api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = self._config.get("azure_endpoint") or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        self.api_version = self._config.get("api_version") or os.getenv(
            "OPENAI_API_VERSION"
        ) or os.getenv(
            "AZURE_OPENAI_API_VERSION"
        )
        # self.model_name is the Azure deployment name, set by base class constructor

        # Validate required parameters and provide specific error messages
        if not self.api_key:
            raise ValueError("Azure OpenAI API key not found")
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not found")
        if not self.api_version:
            raise ValueError("Azure OpenAI API version not found")
        if not self.model_name:
            raise ValueError("Azure OpenAI deployment name (model_name) not found")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        self.async_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )

    @property
    def models(self) -> List[Model]:
        """List available models for this provider.

        Note: For Azure, this typically means the configured deployment.
        Azure OpenAI doesn't provide a models listing API like regular OpenAI.
        Returns an empty list since Azure uses deployments rather than model discovery.
        """
        # Azure doesn't have a models API endpoint - it uses deployments
        # Return empty list since model discovery isn't available
        return []

    def _normalize_response(self, response: OpenAIChatCompletion) -> ChatCompletion:
        """Normalize OpenAI response to our format."""
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
            model=response.model,  # This will be the actual model name from Azure, not deployment
            provider=self.provider,
            usage=Usage(
                completion_tokens=(
                    response.usage.completion_tokens if response.usage else 0
                ),
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
        )

    def _normalize_azure_chunk_to_chat_completion_chunk(
        self,
        chunk: OpenAIChatCompletionChunk,
        finish_reason_str: Optional[str] = None,
    ) -> ChatCompletionChunk:
        """Normalize an Azure streaming chunk to a ChatCompletionChunk."""
        choices_list = []
        if chunk.choices:
            azure_choice = chunk.choices[0]  # Assuming one choice in stream chunk as is typical
            delta = azure_choice.delta
            delta_content = delta.content if delta else None
            # Role might not always be in delta, default or carry over if needed
            role = delta.role if delta and delta.role else "assistant"

            stream_choice = StreamChoice(
                index=azure_choice.index,
                delta=DeltaMessage(
                    content=delta_content or "",  # Ensure content is not None
                    role=role,
                    # Azure delta doesn't typically include function/tool calls directly in this part of stream
                    # It comes as separate message types or accumulated at the end.
                    # For simplicity, keeping it compatible with basic content streaming.
                ),
                finish_reason=azure_choice.finish_reason,
            )
            choices_list.append(stream_choice)

        return ChatCompletionChunk(
            id=chunk.id,
            model=chunk.model or self.model_name,
            created=chunk.created,
            provider=self.provider,
            choices=choices_list,
            is_stream=True,
            raw_response=chunk
        )

    def _get_api_kwargs(
        self, override_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get kwargs for Azure API calls, using current instance attributes and overrides."""
        effective_kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.streaming,
        }

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
            response_stream = self.client.chat.completions.create(
                messages=messages,
                stream=True,  
                **api_kwargs,  
            )
            return (self._normalize_azure_chunk_to_chat_completion_chunk(chunk) for chunk in response_stream)

        response = self.client.chat.completions.create(
            messages=messages,
            stream=False,  
            **api_kwargs,  
        )
        return self._normalize_response(response)

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
            response_stream = await self.async_client.chat.completions.create(
                messages=messages,
                stream=True,
                **api_kwargs,  
            )

            async def generator():
                async for chunk in response_stream:
                    yield self._normalize_azure_chunk_to_chat_completion_chunk(chunk)
            return generator()

        response = await self.async_client.chat.completions.create(
            messages=messages,
            stream=False,
            **api_kwargs,  
        )
        return self._normalize_response(response)

    def to_langchain(
        self, **kwargs: Any
    ) -> "BaseChatModel":  
        """Convert to a LangChain chat model."""
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "LangChain or langchain-openai not installed. "
                "Please install with `pip install langchain_openai`"
            )

        lc_kwargs: Dict[str, Any] = {
            "azure_deployment": self.model_name,
            "api_version": self.api_version,
            "azure_endpoint": self.azure_endpoint,
            "api_key": SecretStr(self.api_key) if self.api_key else None,
            "temperature": self.temperature,  
            "max_tokens": self.max_tokens,  
        }

        if self.top_p is not None:  
            lc_kwargs["top_p"] = self.top_p
        
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
                lc_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        lc_kwargs.update(kwargs)  

        final_lc_kwargs = {k: v for k, v in lc_kwargs.items() if v is not None or k == "api_key"}

        # Remove model_kwargs if it's empty and not explicitly passed in kwargs
        if not final_lc_kwargs.get("model_kwargs") and "model_kwargs" not in kwargs:
            final_lc_kwargs.pop("model_kwargs", None)

        return AzureChatOpenAI(**final_lc_kwargs)

    @property
    def provider(self) -> str:
        return "azure"

    def _get_default_model(self) -> str:
        """Get the default model name (deployment name for Azure).

        For Azure, model_name (deployment name) is required for actual usage.
        Returns the configured model_name or a placeholder for models listing.
        """
        if not self.model_name:
            return "azure-deployment"  # Placeholder when no deployment is configured
        return self.model_name
