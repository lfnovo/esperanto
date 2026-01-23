"""Tests for base model."""

from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from esperanto import LanguageModel
from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Message,
    Tool,
    ToolFunction,
    Usage,
)


class TestLanguageModel(LanguageModel):
    """Test implementation of LanguageModel."""

    def _get_models(self):
        """Get available models (internal method)."""
        return []

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "test"

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "test-default-model"

    def chat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        return ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            provider="test",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="test response"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def achat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        return ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            provider="test",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="test response"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        from langchain_core.chat_models.fake import FakeListChatModel

        return FakeListChatModel(responses=["test response"])


def test_client_properties():
    """Test that client properties are available in base class."""
    model = TestLanguageModel()
    assert hasattr(model, "client")
    assert hasattr(model, "async_client")
    assert model.client is None  # Default value should be None
    assert model.async_client is None  # Default value should be None


def test_language_model_config():
    """Test language model configuration initialization."""
    config = {
        "model_name": "test-model",
        "api_key": "test-key",
        "base_url": "test-url",
        "max_tokens": 1000,
        "temperature": 0.8,
        "streaming": True,
        "top_p": 0.95,
        "structured": {"format": "json"},
        "organization": "test-org",
    }
    model = TestLanguageModel(config=config)

    # Test that all config values are set correctly
    assert model.model_name == "test-model"
    assert model.api_key == "test-key"
    assert model.base_url == "test-url"
    assert model.max_tokens == 1000
    assert model.temperature == 0.8
    assert model.streaming is True
    assert model.top_p == 0.95
    assert model.structured == {"format": "json"}
    assert model.organization == "test-org"


def test_language_model_clean_config():
    """Test clean_config method."""
    model = TestLanguageModel(
        model_name="test-model",
        api_key="test-key",
        base_url=None,  # This should be excluded
        max_tokens=1000,
        temperature=0.8,
    )

    config = model.clean_config()
    assert "model_name" in config
    assert "api_key" in config
    assert "base_url" not in config
    assert config["max_tokens"] == 1000
    assert config["temperature"] == 0.8


def test_language_model_get_completion_kwargs():
    """Test get_completion_kwargs method."""
    model = TestLanguageModel(
        model_name="test-model",
        max_tokens=1000,
        temperature=0.8,
        top_p=0.95,
        streaming=True,
    )

    # Test without override
    kwargs = model.get_completion_kwargs()
    assert kwargs["max_tokens"] == 1000
    assert kwargs["temperature"] == 0.8
    assert kwargs["top_p"] == 0.95
    assert kwargs["streaming"] is True

    # Test with override
    override = {"max_tokens": 500, "temperature": 0.5}
    kwargs = model.get_completion_kwargs(override)
    assert kwargs["max_tokens"] == 500
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.95
    assert kwargs["streaming"] is True


def test_language_model_get_model_name():
    """Test get_model_name method."""
    # Test with model name in config
    model = TestLanguageModel(model_name="test-model")
    assert model.get_model_name() == "test-model"

    # Test fallback to default model
    model = TestLanguageModel()
    assert model.get_model_name() == "test-default-model"



def test_language_model_tool_fields_default():
    """Test that tool fields default to None."""
    model = TestLanguageModel()
    assert model.tools is None
    assert model.tool_choice is None
    assert model.parallel_tool_calls is None


def test_language_model_tool_fields_initialization():
    """Test tool fields can be set during initialization."""
    tools = [
        Tool(
            function=ToolFunction(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            )
        )
    ]
    model = TestLanguageModel(
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    assert model.tools is not None
    assert len(model.tools) == 1
    assert model.tools[0].function.name == "get_weather"
    assert model.tool_choice == "auto"
    assert model.parallel_tool_calls is False


def test_language_model_tool_fields_via_config():
    """Test tool fields can be set via config dict."""
    tools = [
        Tool(
            function=ToolFunction(
                name="search",
                description="Search for information",
            )
        )
    ]
    config = {
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": True,
    }
    model = TestLanguageModel(config=config)

    assert model.tools is not None
    assert len(model.tools) == 1
    assert model.tools[0].function.name == "search"
    assert model.tool_choice == "required"
    assert model.parallel_tool_calls is True


def test_language_model_tool_fields_in_internal_config():
    """Test that tool fields are included in _config."""
    tools = [Tool(function=ToolFunction(name="test", description="Test"))]
    model = TestLanguageModel(
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    assert model._config["tools"] == tools
    assert model._config["tool_choice"] == "auto"
    assert model._config["parallel_tool_calls"] is True


def test_resolve_tools_call_time_precedence():
    """Test that call-time tools take precedence over instance tools."""
    instance_tools = [
        Tool(function=ToolFunction(name="instance_tool", description="Instance"))
    ]
    call_tools = [
        Tool(function=ToolFunction(name="call_tool", description="Call"))
    ]

    model = TestLanguageModel(tools=instance_tools)

    # Call-time tools should take precedence
    resolved = model._resolve_tools(call_tools)
    assert resolved is call_tools
    assert resolved[0].function.name == "call_tool"

    # None should fall back to instance tools
    resolved = model._resolve_tools(None)
    assert resolved is instance_tools
    assert resolved[0].function.name == "instance_tool"


def test_resolve_tools_no_tools():
    """Test _resolve_tools when no tools are configured."""
    model = TestLanguageModel()

    resolved = model._resolve_tools(None)
    assert resolved is None


def test_resolve_tool_choice_call_time_precedence():
    """Test that call-time tool_choice takes precedence over instance setting."""
    model = TestLanguageModel(tool_choice="auto")

    # Call-time should take precedence
    resolved = model._resolve_tool_choice("required")
    assert resolved == "required"

    # None should fall back to instance setting
    resolved = model._resolve_tool_choice(None)
    assert resolved == "auto"


def test_resolve_tool_choice_dict_format():
    """Test _resolve_tool_choice with dict format."""
    specific_tool = {"type": "function", "function": {"name": "get_weather"}}
    model = TestLanguageModel(tool_choice=specific_tool)

    resolved = model._resolve_tool_choice(None)
    assert resolved == specific_tool
    assert resolved["function"]["name"] == "get_weather"


def test_resolve_tool_choice_no_choice():
    """Test _resolve_tool_choice when no choice is configured."""
    model = TestLanguageModel()

    resolved = model._resolve_tool_choice(None)
    assert resolved is None


def test_resolve_parallel_tool_calls_call_time_precedence():
    """Test that call-time parallel_tool_calls takes precedence."""
    model = TestLanguageModel(parallel_tool_calls=True)

    # Call-time should take precedence
    resolved = model._resolve_parallel_tool_calls(False)
    assert resolved is False

    # None should fall back to instance setting
    resolved = model._resolve_parallel_tool_calls(None)
    assert resolved is True


def test_resolve_parallel_tool_calls_no_setting():
    """Test _resolve_parallel_tool_calls when no setting is configured."""
    model = TestLanguageModel()

    resolved = model._resolve_parallel_tool_calls(None)
    assert resolved is None


def test_chat_complete_accepts_tool_parameters():
    """Test that chat_complete accepts all tool-related parameters."""
    model = TestLanguageModel()
    tools = [Tool(function=ToolFunction(name="test", description="Test"))]

    # Should not raise - all parameters should be accepted
    response = model.chat_complete(
        messages=[{"role": "user", "content": "Hello"}],
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=True,
        validate_tool_calls=False,
    )

    assert response is not None
    assert isinstance(response, ChatCompletion)


async def test_achat_complete_accepts_tool_parameters():
    """Test that achat_complete accepts all tool-related parameters."""
    model = TestLanguageModel()
    tools = [Tool(function=ToolFunction(name="test", description="Test"))]

    # Should not raise - all parameters should be accepted
    response = await model.achat_complete(
        messages=[{"role": "user", "content": "Hello"}],
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        validate_tool_calls=True,
    )

    assert response is not None
    assert isinstance(response, ChatCompletion)


def test_chat_complete_accepts_messages_with_tool_results():
    """Test that chat_complete accepts messages with tool results."""
    model = TestLanguageModel()

    # Messages with tool call results (Dict[str, Any] format)
    messages = [
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"temperature": 72}',
        },
    ]

    # Should not raise - Dict[str, Any] messages should be accepted
    response = model.chat_complete(messages=messages)
    assert response is not None


def test_language_model_backward_compatible_without_tools():
    """Test that models work without any tool parameters (backward compatibility)."""
    model = TestLanguageModel()

    # Should work with minimal parameters (original signature)
    response = model.chat_complete(
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert response is not None
    assert isinstance(response, ChatCompletion)
