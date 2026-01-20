"""Tests for the Google/Gemini LLM provider."""
import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from esperanto.common_types import (
    FunctionCall,
    Tool,
    ToolCall,
    ToolFunction,
)
from esperanto.common_types.exceptions import ToolCallValidationError
from esperanto.providers.llm.google import GoogleLanguageModel


@pytest.fixture
def mock_google_chat_response():
    """Mock HTTP response for Google chat completions API."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help you today?"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30
        }
    }


@pytest.fixture
def mock_google_tool_call_response():
    """Mock HTTP response for Google API when model calls a tool."""
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
            "promptTokenCount": 25,
            "candidatesTokenCount": 15,
            "totalTokenCount": 40
        }
    }


@pytest.fixture
def mock_google_tool_call_with_text_response():
    """Mock HTTP response where model returns both text and tool calls."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Let me check the weather for you."},
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "San Francisco"}
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 20,
            "totalTokenCount": 45
        }
    }


@pytest.fixture
def mock_google_models_response():
    """Mock HTTP response for Google models API."""
    return {
        "models": [
            {
                "name": "models/gemini-2.0-flash",
                "inputTokenLimit": 1000000
            },
            {
                "name": "models/gemini-1.5-pro",
                "inputTokenLimit": 2000000
            }
        ]
    }


@pytest.fixture
def mock_google_chat_stream_chunks():
    """Mock SSE chunks for Google streaming chat completions."""
    return [
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":""}],"role":"model"},"finishReason":"STOP"}]}',
        'data: [DONE]'
    ]


@pytest.fixture
def mock_google_stream_with_tool_call():
    """Mock SSE chunks for streaming with tool call."""
    return [
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"SF"}}}],"role":"model"},"finishReason":"STOP"}]}',
        'data: [DONE]'
    ]


@pytest.fixture
def mock_httpx_clients(mock_google_chat_response, mock_google_models_response, mock_google_chat_stream_chunks):
    """Mock httpx clients for Google LLM."""
    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data=None, stream_lines=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        if stream_lines is not None:
            response.iter_text.return_value = stream_lines
        return response

    def make_async_response(status_code, json_data=None, stream_lines=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        if stream_lines is not None:
            async def async_iter():
                for line in stream_lines:
                    yield line
            response.aiter_text = async_iter
        return response

    def mock_post_side_effect(url, **kwargs):
        if "generateContent" in url:
            json_payload = kwargs.get("json", {})
            if "streamGenerateContent" in url:
                return make_response(200, stream_lines=mock_google_chat_stream_chunks)
            else:
                return make_response(200, json_data=mock_google_chat_response)
        return make_response(404, json_data={"error": {"message": "Not found"}})

    def mock_get_side_effect(url, **kwargs):
        if "/models" in url:
            return make_response(200, json_data=mock_google_models_response)
        return make_response(404, json_data={"error": {"message": "Not found"}})

    async def mock_async_post_side_effect(url, **kwargs):
        if "generateContent" in url:
            json_payload = kwargs.get("json", {})
            if "streamGenerateContent" in url:
                return make_async_response(200, stream_lines=mock_google_chat_stream_chunks)
            else:
                return make_async_response(200, json_data=mock_google_chat_response)
        return make_async_response(404, json_data={"error": {"message": "Not found"}})

    client.post.side_effect = mock_post_side_effect
    client.get.side_effect = mock_get_side_effect
    async_client.post.side_effect = mock_async_post_side_effect

    return client, async_client


@pytest.fixture
def google_model(mock_httpx_clients):
    """Create a Google model with mocked HTTP clients."""
    model = GoogleLanguageModel(api_key="test-key", model_name="gemini-2.0-flash")
    client, async_client = mock_httpx_clients
    model.client = client
    model.async_client = async_client
    return model


@pytest.fixture
def sample_tools():
    """Sample tools for testing."""
    return [
        Tool(
            function=ToolFunction(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
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
                        "timezone": {"type": "string", "description": "The timezone"}
                    }
                }
            )
        )
    ]


@pytest.fixture
def google_model_with_tool_response(mock_google_tool_call_response):
    """Create a Google model with tool call response mocked."""
    model = GoogleLanguageModel(api_key="test-key", model_name="gemini-2.0-flash")

    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    def make_async_response(status_code, json_data):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data
        return response

    client.post.return_value = make_response(200, mock_google_tool_call_response)
    async_client.post.return_value = make_async_response(200, mock_google_tool_call_response)

    model.client = client
    model.async_client = async_client
    return model


class TestGoogleProviderBasic:
    """Basic tests for Google provider."""

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        model = GoogleLanguageModel(api_key="test-key", model_name="gemini-2.0-flash")
        assert model.api_key == "test-key"
        assert model.get_model_name() == "gemini-2.0-flash"

    def test_initialization_from_env(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            model = GoogleLanguageModel(model_name="gemini-2.0-flash")
            assert model.api_key == "env-key"

    def test_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear both GOOGLE_API_KEY and GEMINI_API_KEY
            with pytest.raises(ValueError, match="Google API key not found"):
                GoogleLanguageModel(model_name="gemini-2.0-flash")

    def test_provider_name(self, google_model):
        """Test provider property returns 'google'."""
        assert google_model.provider == "google"

    def test_default_model(self, google_model):
        """Test default model is gemini-2.0-flash."""
        model = GoogleLanguageModel(api_key="test-key")
        # Override client to avoid actual HTTP calls
        model.client = Mock()
        assert model._get_default_model() == "gemini-2.0-flash"


class TestToolConversion:
    """Tests for tool conversion to Google format."""

    def test_convert_single_tool(self, google_model, sample_tools):
        """Test converting a single tool to Google format."""
        result = google_model._convert_tools_to_google([sample_tools[0]])

        assert len(result) == 1
        assert "function_declarations" in result[0]
        declarations = result[0]["function_declarations"]
        assert len(declarations) == 1
        assert declarations[0]["name"] == "get_weather"
        assert declarations[0]["description"] == "Get the current weather for a location"
        assert declarations[0]["parameters"]["type"] == "object"
        assert "location" in declarations[0]["parameters"]["properties"]

    def test_convert_multiple_tools(self, google_model, sample_tools):
        """Test converting multiple tools to Google format."""
        result = google_model._convert_tools_to_google(sample_tools)

        # All tools should be in a single function_declarations array
        assert len(result) == 1
        declarations = result[0]["function_declarations"]
        assert len(declarations) == 2
        assert declarations[0]["name"] == "get_weather"
        assert declarations[1]["name"] == "get_time"

    def test_convert_none_tools(self, google_model):
        """Test converting None returns None."""
        result = google_model._convert_tools_to_google(None)
        assert result is None

    def test_convert_empty_tools(self, google_model):
        """Test converting empty list returns None."""
        result = google_model._convert_tools_to_google([])
        assert result is None


class TestToolChoiceConversion:
    """Tests for tool choice conversion to Google format."""

    def test_convert_auto(self, google_model):
        """Test converting 'auto' tool_choice."""
        result = google_model._convert_tool_choice_to_google("auto")
        assert result == {"function_calling_config": {"mode": "AUTO"}}

    def test_convert_required(self, google_model):
        """Test converting 'required' tool_choice."""
        result = google_model._convert_tool_choice_to_google("required")
        assert result == {"function_calling_config": {"mode": "ANY"}}

    def test_convert_none(self, google_model):
        """Test converting 'none' tool_choice."""
        result = google_model._convert_tool_choice_to_google("none")
        assert result == {"function_calling_config": {"mode": "NONE"}}

    def test_convert_specific_tool(self, google_model):
        """Test converting specific tool choice."""
        specific_choice = {"type": "function", "function": {"name": "get_weather"}}
        result = google_model._convert_tool_choice_to_google(specific_choice)
        assert result == {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["get_weather"]
            }
        }

    def test_convert_none_value(self, google_model):
        """Test converting None returns None."""
        result = google_model._convert_tool_choice_to_google(None)
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, google_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = google_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = google_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload
        assert "function_declarations" in json_payload["tools"][0]

        # Check response has tool calls
        assert len(response.choices) == 1
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tool_call = response.choices[0].message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args["location"] == "San Francisco"

    def test_chat_complete_with_tool_choice(self, google_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        google_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = google_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tool_config" in json_payload
        assert json_payload["tool_config"]["function_calling_config"]["mode"] == "ANY"

    def test_chat_complete_with_specific_tool_choice(self, google_model_with_tool_response, sample_tools):
        """Test chat_complete with specific tool choice."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        specific_choice = {"type": "function", "function": {"name": "get_weather"}}

        google_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice=specific_choice
        )

        call_args = google_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tool_config" in json_payload
        assert json_payload["tool_config"]["function_calling_config"]["allowed_function_names"] == ["get_weather"]

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, google_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await google_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = google_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestInstanceLevelTools:
    """Tests for instance-level tool configuration."""

    def test_instance_tools_used_when_no_call_tools(self, mock_google_tool_call_response, sample_tools):
        """Test that instance-level tools are used when not passed at call time."""
        model = GoogleLanguageModel(
            api_key="test-key",
            model_name="gemini-2.0-flash",
            tools=sample_tools
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_google_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload
        declarations = json_payload["tools"][0]["function_declarations"]
        assert len(declarations) == 2

    def test_call_tools_override_instance_tools(self, mock_google_tool_call_response, sample_tools):
        """Test that call-time tools override instance-level tools."""
        instance_tool = Tool(
            function=ToolFunction(name="instance_tool", description="Instance tool")
        )
        model = GoogleLanguageModel(
            api_key="test-key",
            model_name="gemini-2.0-flash",
            tools=[instance_tool]
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_google_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages, tools=sample_tools)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        # Should have call-time tools, not instance tools
        declarations = json_payload["tools"][0]["function_declarations"]
        assert declarations[0]["name"] == "get_weather"

    def test_instance_tool_choice(self, mock_google_tool_call_response, sample_tools):
        """Test instance-level tool_choice is used."""
        model = GoogleLanguageModel(
            api_key="test-key",
            model_name="gemini-2.0-flash",
            tools=sample_tools,
            tool_choice="required"
        )

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = mock_google_tool_call_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        model.chat_complete(messages)

        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        assert json_payload["tool_config"]["function_calling_config"]["mode"] == "ANY"


class TestToolCallValidation:
    """Tests for tool call validation."""

    def test_validation_passes_for_valid_tool_call(
        self, google_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        # Should not raise
        response = google_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, validate_tool_calls=True
        )
        assert response.choices[0].message.tool_calls is not None

    def test_validation_fails_for_invalid_tool_call(self, sample_tools):
        """Test that validation fails for invalid tool calls."""
        # Create a model that returns invalid args
        model = GoogleLanguageModel(api_key="test-key", model_name="gemini-2.0-flash")

        invalid_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    # Missing required "location"
                                    "args": {"unit": "celsius"}
                                }
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        }

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = invalid_response
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        with pytest.raises(ToolCallValidationError):
            model.chat_complete(messages, tools=sample_tools, validate_tool_calls=True)


class TestMessageFormatting:
    """Tests for message formatting with tool-related content."""

    def test_format_tool_result_message(self, google_model):
        """Test formatting tool result messages."""
        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "get_weather",
                "content": '{"temperature": 72, "condition": "sunny"}'
            }
        ]

        formatted, system_instruction = google_model._format_messages(messages)

        # User message
        assert formatted[0]["role"] == "user"
        assert formatted[0]["parts"][0]["text"] == "What's the weather in SF?"

        # Assistant with tool call
        assert formatted[1]["role"] == "model"
        assert "functionCall" in formatted[1]["parts"][0]
        assert formatted[1]["parts"][0]["functionCall"]["name"] == "get_weather"

        # Tool result (converted to user with functionResponse)
        assert formatted[2]["role"] == "user"
        assert "functionResponse" in formatted[2]["parts"][0]
        assert formatted[2]["parts"][0]["functionResponse"]["name"] == "get_weather"

    def test_format_assistant_with_text_and_tool_call(self, google_model):
        """Test formatting assistant message with both text and tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}'
                        }
                    }
                ]
            }
        ]

        formatted, _ = google_model._format_messages(messages)

        assert formatted[0]["role"] == "model"
        # First part should be text
        assert formatted[0]["parts"][0]["text"] == "Let me check"
        # Second part should be functionCall
        assert "functionCall" in formatted[0]["parts"][1]


class TestNormalizeResponse:
    """Tests for response normalization."""

    def test_normalize_response_with_tool_calls(self, google_model, mock_google_tool_call_response):
        """Test normalizing response with tool calls."""
        result = google_model._normalize_response(mock_google_tool_call_response)

        assert len(result.choices) == 1
        message = result.choices[0].message
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1

        tool_call = message.tool_calls[0]
        assert tool_call.function.name == "get_weather"
        assert tool_call.type == "function"
        assert tool_call.id.startswith("call_")

        args = json.loads(tool_call.function.arguments)
        assert args["location"] == "San Francisco"
        assert args["unit"] == "celsius"

    def test_normalize_response_with_text_and_tool_calls(
        self, google_model, mock_google_tool_call_with_text_response
    ):
        """Test normalizing response with both text and tool calls."""
        result = google_model._normalize_response(mock_google_tool_call_with_text_response)

        message = result.choices[0].message
        assert message.content == "Let me check the weather for you."
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1

    def test_normalize_response_finish_reason_tool_calls(self, google_model, mock_google_tool_call_response):
        """Test that finish_reason is 'tool_calls' when tools are called."""
        result = google_model._normalize_response(mock_google_tool_call_response)
        assert result.choices[0].finish_reason == "tool_calls"

    def test_normalize_response_without_tool_calls(self, google_model, mock_google_chat_response):
        """Test normalizing response without tool calls."""
        result = google_model._normalize_response(mock_google_chat_response)

        message = result.choices[0].message
        assert message.content == "Hello! How can I help you today?"
        assert message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"


class TestStreamingWithTools:
    """Tests for streaming with tool calls."""

    def test_normalize_chunk_with_tool_call(self, google_model):
        """Test normalizing streaming chunk with tool call."""
        chunk_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "SF"}
                                }
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }
            ]
        }

        result = google_model._normalize_chunk(chunk_data)

        assert result is not None
        delta = result.choices[0].delta
        assert delta.tool_calls is not None
        assert len(delta.tool_calls) == 1
        # Streaming tool_calls are ToolCall objects due to Message validation
        tool_call = delta.tool_calls[0]
        assert tool_call.function.name == "get_weather"

    def test_streaming_chat_complete_with_tools(self, sample_tools, mock_google_stream_with_tool_call):
        """Test streaming chat_complete with tools."""
        model = GoogleLanguageModel(api_key="test-key", model_name="gemini-2.0-flash")

        client = Mock()
        response = Mock()
        response.status_code = 200
        response.iter_text.return_value = mock_google_stream_with_tool_call
        client.post.return_value = response
        model.client = client

        messages = [{"role": "user", "content": "What's the weather?"}]
        result = model.chat_complete(messages, tools=sample_tools, stream=True)

        chunks = list(result)
        assert len(chunks) > 0

        # Check that tools were included in payload
        call_args = client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload


class TestErrorHandling:
    """Tests for error handling."""

    def test_handle_api_error(self, google_model):
        """Test handling API errors."""
        client = Mock()
        response = Mock()
        response.status_code = 400
        response.json.return_value = {"error": {"message": "Invalid request"}}
        client.post.return_value = response
        google_model.client = client

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RuntimeError, match="Google API error"):
            google_model.chat_complete(messages)
