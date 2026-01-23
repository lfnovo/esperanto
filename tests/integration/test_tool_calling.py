"""Integration tests for tool calling across providers.

These tests verify that tool calling works consistently across all providers
that support it. The tests use mocked responses to ensure consistent behavior
without requiring actual API keys.
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.common_types import (
    Tool,
    ToolFunction,
    ToolCall,
    FunctionCall,
    ChatCompletion,
    ToolCallValidationError,
)
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.llm.anthropic import AnthropicLanguageModel
from esperanto.providers.llm.google import GoogleLanguageModel
from esperanto.providers.llm.groq import GroqLanguageModel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tools():
    """Standard tools for testing across providers."""
    return [
        Tool(
            function=ToolFunction(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. 'San Francisco'"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            )
        ),
        Tool(
            function=ToolFunction(
                name="get_time",
                description="Get the current time for a timezone",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone, e.g. 'America/Los_Angeles'"
                        }
                    }
                }
            )
        )
    ]


@pytest.fixture
def mock_openai_tool_response():
    """Mock OpenAI response with tool calls."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco", "unit": "celsius"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
    }


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic response with tool calls."""
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "get_weather",
                "input": {"location": "San Francisco", "unit": "celsius"}
            }
        ],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 20}
    }


@pytest.fixture
def mock_google_tool_response():
    """Mock Google response with tool calls."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "San Francisco", "unit": "celsius"}
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 20,
            "totalTokenCount": 70
        }
    }


# =============================================================================
# Test Tool Types
# =============================================================================


class TestToolTypes:
    """Tests for tool type definitions."""

    def test_tool_creation(self, sample_tools):
        """Test that tools can be created with the standard format."""
        assert len(sample_tools) == 2
        assert sample_tools[0].type == "function"
        assert sample_tools[0].function.name == "get_weather"
        assert "location" in sample_tools[0].function.parameters["properties"]

    def test_tool_function_default_parameters(self):
        """Test that ToolFunction has sensible defaults for parameters."""
        func = ToolFunction(name="test", description="Test function")
        assert func.parameters == {"type": "object", "properties": {}}

    def test_tool_call_creation(self):
        """Test that ToolCall objects can be created."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "NYC"}'
            )
        )
        assert tool_call.id == "call_123"
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args["location"] == "NYC"

    def test_tool_serialization(self, sample_tools):
        """Test that tools can be serialized to dict."""
        tool_dict = sample_tools[0].model_dump()
        assert tool_dict["type"] == "function"
        assert tool_dict["function"]["name"] == "get_weather"


# =============================================================================
# Provider-Specific Tool Conversion Tests
# =============================================================================


class TestProviderToolConversion:
    """Test that each provider converts tools to their expected format."""

    def test_openai_tool_conversion(self, sample_tools):
        """Test OpenAI converts tools to OpenAI format."""
        model = OpenAILanguageModel(api_key="test-key")
        converted = model._convert_tools_to_openai(sample_tools)

        assert len(converted) == 2
        # OpenAI format
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "get_weather"
        assert converted[0]["function"]["parameters"]["type"] == "object"

    def test_anthropic_tool_conversion(self, sample_tools):
        """Test Anthropic converts tools to Anthropic format."""
        model = AnthropicLanguageModel(api_key="test-key")
        converted = model._convert_tools_to_anthropic(sample_tools)

        assert len(converted) == 2
        # Anthropic format uses input_schema instead of parameters
        assert converted[0]["name"] == "get_weather"
        assert "input_schema" in converted[0]
        assert converted[0]["input_schema"]["type"] == "object"

    def test_google_tool_conversion(self, sample_tools):
        """Test Google converts tools to Google format."""
        model = GoogleLanguageModel(api_key="test-key")
        converted = model._convert_tools_to_google(sample_tools)

        assert len(converted) == 1  # Google wraps all in one object
        # Google format uses function_declarations
        assert "function_declarations" in converted[0]
        declarations = converted[0]["function_declarations"]
        assert len(declarations) == 2
        assert declarations[0]["name"] == "get_weather"

    def test_groq_tool_conversion(self, sample_tools):
        """Test Groq converts tools to OpenAI-compatible format."""
        model = GroqLanguageModel(api_key="test-key")
        converted = model._convert_tools_to_openai(sample_tools)

        assert len(converted) == 2
        # Groq uses OpenAI format
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "get_weather"


# =============================================================================
# Provider Response Normalization Tests
# =============================================================================


class TestProviderResponseNormalization:
    """Test that each provider normalizes tool call responses consistently."""

    def test_openai_response_has_tool_calls(self, mock_openai_tool_response):
        """Test OpenAI response normalization includes tool calls."""
        model = OpenAILanguageModel(api_key="test-key")

        # Mock the HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openai_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        response = model.chat_complete([{"role": "user", "content": "What's the weather?"}])

        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tool_call = response.choices[0].message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

    def test_anthropic_response_has_tool_calls(self, mock_anthropic_tool_response):
        """Test Anthropic response normalization includes tool calls."""
        model = AnthropicLanguageModel(api_key="test-key")

        # Mock the HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_anthropic_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        response = model.chat_complete([{"role": "user", "content": "What's the weather?"}])

        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tool_call = response.choices[0].message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

    def test_google_response_has_tool_calls(self, mock_google_tool_response):
        """Test Google response normalization includes tool calls."""
        model = GoogleLanguageModel(api_key="test-key")

        # Mock the HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_google_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        response = model.chat_complete([{"role": "user", "content": "What's the weather?"}])

        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tool_call = response.choices[0].message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"


# =============================================================================
# Tool Call Validation Tests
# =============================================================================


class TestToolCallValidation:
    """Test tool call validation works across providers."""

    def test_validation_passes_valid_arguments(self, sample_tools, mock_openai_tool_response):
        """Test validation passes for valid tool call arguments."""
        pytest.importorskip("jsonschema")

        model = OpenAILanguageModel(api_key="test-key")

        # Mock the HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openai_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        # Should not raise
        response = model.chat_complete(
            [{"role": "user", "content": "What's the weather?"}],
            tools=sample_tools,
            validate_tool_calls=True
        )

        assert response.choices[0].message.tool_calls is not None

    def test_validation_fails_missing_required(self, sample_tools):
        """Test validation fails when required argument is missing."""
        pytest.importorskip("jsonschema")

        # Response with missing required 'location' field
        invalid_response = {
            "id": "chatcmpl-invalid",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_invalid",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"unit": "celsius"}'  # Missing 'location'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

        model = OpenAILanguageModel(api_key="test-key")
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = invalid_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        with pytest.raises(ToolCallValidationError):
            model.chat_complete(
                [{"role": "user", "content": "What's the weather?"}],
                tools=sample_tools,
                validate_tool_calls=True
            )


# =============================================================================
# Tool Choice Tests
# =============================================================================


class TestToolChoice:
    """Test that tool_choice parameter works across providers."""

    def test_openai_tool_choice_auto(self, sample_tools, mock_openai_tool_response):
        """Test OpenAI with tool_choice='auto'."""
        model = OpenAILanguageModel(api_key="test-key")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openai_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        model.chat_complete(
            [{"role": "user", "content": "What's the weather?"}],
            tools=sample_tools,
            tool_choice="auto"
        )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload.get("tool_choice") == "auto"

    def test_openai_tool_choice_required(self, sample_tools, mock_openai_tool_response):
        """Test OpenAI with tool_choice='required'."""
        model = OpenAILanguageModel(api_key="test-key")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openai_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        model.chat_complete(
            [{"role": "user", "content": "What's the weather?"}],
            tools=sample_tools,
            tool_choice="required"
        )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload.get("tool_choice") == "required"

    def test_anthropic_tool_choice_conversion(self, sample_tools, mock_anthropic_tool_response):
        """Test Anthropic tool_choice is converted correctly."""
        model = AnthropicLanguageModel(api_key="test-key")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_anthropic_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        model.chat_complete(
            [{"role": "user", "content": "What's the weather?"}],
            tools=sample_tools,
            tool_choice="auto"
        )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        # Anthropic converts "auto" to {"type": "auto"}
        assert payload.get("tool_choice") == {"type": "auto"}


# =============================================================================
# Multi-Turn Conversation Tests
# =============================================================================


class TestMultiTurnToolConversation:
    """Test multi-turn conversations with tool results."""

    def test_tool_result_message_format(self):
        """Test that tool result messages have the expected format."""
        # Standard format for tool result messages
        tool_result_message = {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"temperature": 72, "condition": "sunny"}'
        }

        assert tool_result_message["role"] == "tool"
        assert tool_result_message["tool_call_id"] == "call_abc123"
        assert "temperature" in tool_result_message["content"]

    def test_openai_multi_turn_with_tool_result(self, mock_openai_tool_response):
        """Test OpenAI handles multi-turn conversation with tool results."""
        model = OpenAILanguageModel(api_key="test-key")

        # Second response after tool execution
        final_response = {
            "id": "chatcmpl-final",
            "object": "chat.completion",
            "created": 1677652289,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The weather in San Francisco is 72°F and sunny."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}
        }

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = final_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        # Multi-turn conversation with tool result
        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temperature": 72, "condition": "sunny"}'
            }
        ]

        response = model.chat_complete(messages)

        assert response.choices[0].message.content is not None
        assert "72" in response.choices[0].message.content or "sunny" in response.choices[0].message.content


# =============================================================================
# Instance-Level Tool Configuration Tests
# =============================================================================


class TestInstanceLevelTools:
    """Test instance-level tool configuration."""

    def test_instance_tools_used_when_no_call_tools(self, sample_tools, mock_openai_tool_response):
        """Test that instance-level tools are used when not overridden."""
        model = OpenAILanguageModel(
            api_key="test-key",
            tools=sample_tools
        )

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openai_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        model.chat_complete([{"role": "user", "content": "What's the weather?"}])

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert "tools" in payload
        assert len(payload["tools"]) == 2

    def test_call_tools_override_instance_tools(self, sample_tools, mock_openai_tool_response):
        """Test that call-time tools override instance-level tools."""
        instance_tool = Tool(
            function=ToolFunction(name="instance_tool", description="Instance tool")
        )
        model = OpenAILanguageModel(
            api_key="test-key",
            tools=[instance_tool]
        )

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openai_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        model.chat_complete(
            [{"role": "user", "content": "What's the weather?"}],
            tools=sample_tools  # Override with call-time tools
        )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        # Should use call-time tools, not instance tools
        assert payload["tools"][0]["function"]["name"] == "get_weather"


# =============================================================================
# Parallel Tool Calls Tests
# =============================================================================


class TestParallelToolCalls:
    """Test parallel tool call handling."""

    def test_multiple_tool_calls_in_response(self):
        """Test handling multiple tool calls in a single response."""
        # Response with multiple tool calls
        multi_tool_response = {
            "id": "chatcmpl-multi",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}'
                                }
                            },
                            {
                                "id": "call_time",
                                "type": "function",
                                "function": {
                                    "name": "get_time",
                                    "arguments": '{"timezone": "America/Los_Angeles"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 40, "total_tokens": 90}
        }

        model = OpenAILanguageModel(api_key="test-key")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = multi_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        response = model.chat_complete([{"role": "user", "content": "Weather and time?"}])

        assert len(response.choices[0].message.tool_calls) == 2
        tool_names = [tc.function.name for tc in response.choices[0].message.tool_calls]
        assert "get_weather" in tool_names
        assert "get_time" in tool_names

    def test_parallel_tool_calls_disabled(self, sample_tools, mock_openai_tool_response):
        """Test that parallel_tool_calls=False is passed to provider."""
        model = OpenAILanguageModel(api_key="test-key")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_openai_tool_response
        mock_client.post.return_value = mock_response
        model.client = mock_client

        model.chat_complete(
            [{"role": "user", "content": "What's the weather?"}],
            tools=sample_tools,
            parallel_tool_calls=False
        )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload.get("parallel_tool_calls") is False
