"""Base language model interface."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from httpx import AsyncClient, Client

from esperanto.common_types import ChatCompletion, ChatCompletionChunk, Model, Tool
from esperanto.utils.connect import HttpConnectionMixin


@dataclass
class LanguageModel(HttpConnectionMixin, ABC):
    """Base class for all language models."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 850
    temperature: float = 1.0
    streaming: bool = False
    top_p: float = 0.9
    structured: Optional[Dict[str, Any]] = None
    organization: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    # Tool-related fields
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    _config: Dict[str, Any] = field(default_factory=dict)
    client: Optional[Client] = None
    async_client: Optional[AsyncClient] = None

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.

        .. deprecated:: 2.8.0
            The `.models` property is deprecated and will be removed in version 3.0.
            Use `AIFactory.get_provider_models(provider_name)` instead for static
            model discovery without creating provider instances.

        Returns:
            List[Model]: List of available models
        """
        warnings.warn(
            f"The `.models` property is deprecated and will be removed in version 3.0. "
            f"Use AIFactory.get_provider_models('{self.provider}') instead for static "
            f"model discovery without creating provider instances.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._get_models()

    @abstractmethod
    def _get_models(self) -> List[Model]:
        """Internal method to get available models.

        This method should be implemented by providers. The public `.models` property
        will emit a deprecation warning and call this method.

        Returns:
            List[Model]: List of available models
        """
        pass

    def __post_init__(self):
        """Initialize configuration after dataclass initialization."""
        # Initialize config with default values
        self._config = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
            "structured": self.structured,
            # Tool-related config
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
        }

        # Update with any provided config
        if hasattr(self, "config") and self.config:
            self._config.update(self.config)

            # Update instance attributes from config
            for key, value in self._config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from config dictionary."""
        return {k: v for k, v in config.items() if v is not None}

    def _resolve_tools(
        self, tools: Optional[List[Tool]] = None
    ) -> Optional[List[Tool]]:
        """Resolve tools from parameter or instance config.

        Call-time tools take precedence over instance-level tools.

        Args:
            tools: Tools passed at call time, or None to use instance tools.

        Returns:
            The resolved list of tools, or None if no tools configured.
        """
        if tools is not None:
            return tools
        return self.tools

    def _resolve_tool_choice(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Resolve tool_choice from parameter or instance config.

        Call-time tool_choice takes precedence over instance-level tool_choice.

        Args:
            tool_choice: Tool choice passed at call time, or None to use instance setting.

        Returns:
            The resolved tool_choice, or None if not configured.
        """
        if tool_choice is not None:
            return tool_choice
        return self.tool_choice

    def _resolve_parallel_tool_calls(
        self, parallel_tool_calls: Optional[bool] = None
    ) -> Optional[bool]:
        """Resolve parallel_tool_calls from parameter or instance config.

        Call-time parallel_tool_calls takes precedence over instance-level setting.

        Args:
            parallel_tool_calls: Parallel tool calls setting passed at call time,
                or None to use instance setting.

        Returns:
            The resolved parallel_tool_calls setting, or None for provider default.
        """
        if parallel_tool_calls is not None:
            return parallel_tool_calls
        return self.parallel_tool_calls

    def _warn_if_validate_with_streaming(
        self,
        validate_tool_calls: bool,
        stream: Optional[bool],
    ) -> None:
        """Emit a warning if validate_tool_calls is used with streaming.

        Tool call validation requires the complete response to validate arguments
        against the tool schema. With streaming, we receive partial chunks and
        cannot validate until all chunks are collected. This method warns users
        that their validate_tool_calls flag will be ignored.

        Args:
            validate_tool_calls: Whether validation was requested.
            stream: The stream parameter passed to chat_complete.
        """
        should_stream = stream if stream is not None else self.streaming
        if validate_tool_calls and should_stream:
            warnings.warn(
                "validate_tool_calls=True is ignored when streaming is enabled. "
                "Tool call validation requires the complete response.",
                UserWarning,
                stacklevel=3,
            )

    @abstractmethod
    def chat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation. Messages can include
                tool call results with role="tool" and tool_call_id.
            stream: Whether to stream the response. If None, uses the instance's
                streaming setting.
            tools: List of tools the model can call. If None, uses instance tools.
            tool_choice: Controls tool usage. Values:
                - "auto": Model decides whether to call tools (default)
                - "required": Model must call at least one tool
                - "none": Model cannot call tools
                - {"type": "function", "function": {"name": "..."}}: Force specific tool
            parallel_tool_calls: Whether to allow multiple tool calls in one response.
                None uses provider default (usually True). Set False to force single
                tool call per response.
            validate_tool_calls: If True, validate tool call arguments against the
                tool's JSON schema. Raises ToolCallValidationError on validation
                failure. Requires jsonschema package.

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks
            if streaming. When the model calls tools, the response message will
            have tool_calls populated.
        """
        pass

    @abstractmethod
    async def achat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request.

        Args:
            messages: List of messages in the conversation. Messages can include
                tool call results with role="tool" and tool_call_id.
            stream: Whether to stream the response. If None, uses the instance's
                streaming setting.
            tools: List of tools the model can call. If None, uses instance tools.
            tool_choice: Controls tool usage. Values:
                - "auto": Model decides whether to call tools (default)
                - "required": Model must call at least one tool
                - "none": Model cannot call tools
                - {"type": "function", "function": {"name": "..."}}: Force specific tool
            parallel_tool_calls: Whether to allow multiple tool calls in one response.
                None uses provider default (usually True). Set False to force single
                tool call per response.
            validate_tool_calls: If True, validate tool call arguments against the
                tool's JSON schema. Raises ToolCallValidationError on validation
                failure. Requires jsonschema package.

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks
            if streaming. When the model calls tools, the response message will
            have tool_calls populated.
        """
        pass

    def clean_config(self) -> Dict[str, Any]:
        """Clean the configuration dictionary.

        Returns:
            Dict[str, Any]: The cleaned configuration.
        """
        config = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and value is not None:
                config[key] = value
        return config

    def get_completion_kwargs(
        self, override_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get kwargs for completion API calls."""
        kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
        }

        if override_kwargs:
            kwargs.update(override_kwargs)

        return kwargs

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            str: The model name.
        """
        # First try to get from config
        model_name = self._config.get("model_name")
        if model_name:
            return model_name

        # If not in config, use default
        return self._get_default_model()

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider name."""
        pass

    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model name.

        Returns:
            str: The default model name.
        """
        pass

    def _get_provider_type(self) -> str:
        """Return provider type for timeout configuration.

        Returns:
            str: "language" for LLM providers
        """
        return "language"

    @abstractmethod
    def to_langchain(self) -> Any:
        """Convert to a LangChain chat model.

        Returns:
            BaseChatModel: A LangChain chat model instance specific to the provider.

        Raises:
            ImportError: If langchain_core is not installed.
        """
        pass

    def to_pydantic_ai(self) -> Any:
        """Convert to a Pydantic AI compatible model.

        Returns a model that can be used with Pydantic AI agents. This works
        for all Esperanto providers - no provider-specific implementation needed.

        Example:
            ```python
            from esperanto import AIFactory
            from pydantic_ai import Agent

            model = AIFactory.create_language("openai", "gpt-4o")
            agent = Agent(model.to_pydantic_ai())
            result = await agent.run("Hello!")
            ```

        Returns:
            EsperantoPydanticModel: A Pydantic AI Model instance wrapping this provider.

        Raises:
            ImportError: If pydantic-ai is not installed.
        """
        try:
            from esperanto.integrations.pydantic_ai import EsperantoPydanticModel
        except ImportError as e:
            raise ImportError(
                "Pydantic AI integration requires pydantic-ai. "
                "Install with: uv add pydantic-ai or pip install pydantic-ai"
            ) from e

        return EsperantoPydanticModel(self)
