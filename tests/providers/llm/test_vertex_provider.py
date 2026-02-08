"""Tests for the Vertex AI LLM provider."""
import json
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from esperanto.common_types import (
    FunctionCall,
    Tool,
    ToolCall,
    ToolFunction,
)
from esperanto.common_types.exceptions import ToolCallValidationError
from esperanto.providers.llm.vertex import VertexLanguageModel


@pytest.fixture
def mock_vertex_chat_response():
    """Mock HTTP response for Vertex AI chat completions API."""
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
def mock_vertex_tool_call_response():
    """Mock HTTP response for Vertex AI API when model calls a tool."""
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
def mock_vertex_tool_call_with_text_response():
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
def mock_google_auth():
    """Mock google.auth.default to prevent real ADC pickup in tests."""
    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds.token = "mock-adc-token"
    with patch("google.auth.default", return_value=(mock_creds, "test-project")) as mock_default:
        yield mock_default, mock_creds


@pytest.fixture
def mock_gcloud_auth(mock_google_auth):
    """Mock gcloud authentication (also mocks google.auth.default)."""
    with patch("subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.stdout = "mock-access-token"
        mock_run.return_value = mock_result
        yield mock_run


@pytest.fixture
def mock_httpx_clients(mock_vertex_chat_response):
    """Mock httpx clients for Vertex AI LLM."""
    client = Mock()
    async_client = AsyncMock()

    def make_response(status_code, json_data=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        return response

    def make_async_response(status_code, json_data=None):
        response = Mock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        return response

    def mock_post_side_effect(url, **kwargs):
        if "generateContent" in url:
            return make_response(200, json_data=mock_vertex_chat_response)
        return make_response(404, json_data={"error": {"message": "Not found"}})

    async def mock_async_post_side_effect(url, **kwargs):
        if "generateContent" in url:
            return make_async_response(200, json_data=mock_vertex_chat_response)
        return make_async_response(404, json_data={"error": {"message": "Not found"}})

    client.post.side_effect = mock_post_side_effect
    async_client.post.side_effect = mock_async_post_side_effect

    return client, async_client


@pytest.fixture
def vertex_model(mock_gcloud_auth, mock_httpx_clients):
    """Create a Vertex AI model with mocked HTTP clients."""
    with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
        model = VertexLanguageModel(model_name="gemini-2.0-flash")
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
def vertex_model_with_tool_response(mock_gcloud_auth, mock_vertex_tool_call_response):
    """Create a Vertex AI model with tool call response mocked."""
    with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
        model = VertexLanguageModel(model_name="gemini-2.0-flash")

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

        client.post.return_value = make_response(200, mock_vertex_tool_call_response)
        async_client.post.return_value = make_async_response(200, mock_vertex_tool_call_response)

        model.client = client
        model.async_client = async_client
        return model


class TestVertexProviderBasic:
    """Basic tests for Vertex AI provider."""

    def test_initialization_with_project(self, mock_gcloud_auth):
        """Test initialization with project ID."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
            model = VertexLanguageModel(model_name="gemini-2.0-flash")
            assert model.project_id == "test-project"
            assert model.get_model_name() == "gemini-2.0-flash"

    def test_initialization_from_env(self, mock_gcloud_auth):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "env-project"}):
            model = VertexLanguageModel(model_name="gemini-2.0-flash")
            assert model.project_id == "env-project"

    def test_missing_project_raises(self, mock_gcloud_auth):
        """Test that missing project raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear VERTEX_PROJECT and GOOGLE_CLOUD_PROJECT
            with pytest.raises(ValueError, match="Google Cloud project ID not found"):
                VertexLanguageModel(model_name="gemini-2.0-flash")

    def test_provider_name(self, vertex_model):
        """Test provider property returns 'vertex'."""
        assert vertex_model.provider == "vertex"

    def test_default_model(self, mock_gcloud_auth):
        """Test default model is gemini-2.0-flash."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
            model = VertexLanguageModel()
            # Override client to avoid actual HTTP calls
            model.client = Mock()
            assert model._get_default_model() == "gemini-2.0-flash"


class TestToolConversion:
    """Tests for tool conversion to Vertex AI format."""

    def test_convert_single_tool(self, vertex_model, sample_tools):
        """Test converting a single tool to Vertex AI format."""
        result = vertex_model._convert_tools_to_vertex([sample_tools[0]])

        assert len(result) == 1
        assert "function_declarations" in result[0]
        declarations = result[0]["function_declarations"]
        assert len(declarations) == 1
        assert declarations[0]["name"] == "get_weather"
        assert declarations[0]["description"] == "Get the current weather for a location"
        assert declarations[0]["parameters"]["type"] == "object"
        assert "location" in declarations[0]["parameters"]["properties"]

    def test_convert_multiple_tools(self, vertex_model, sample_tools):
        """Test converting multiple tools to Vertex AI format."""
        result = vertex_model._convert_tools_to_vertex(sample_tools)

        # All tools should be in a single function_declarations array
        assert len(result) == 1
        declarations = result[0]["function_declarations"]
        assert len(declarations) == 2
        assert declarations[0]["name"] == "get_weather"
        assert declarations[1]["name"] == "get_time"

    def test_convert_none_tools(self, vertex_model):
        """Test converting None returns None."""
        result = vertex_model._convert_tools_to_vertex(None)
        assert result is None

    def test_convert_empty_tools(self, vertex_model):
        """Test converting empty list returns None."""
        result = vertex_model._convert_tools_to_vertex([])
        assert result is None


class TestToolChoiceConversion:
    """Tests for tool choice conversion to Vertex AI format."""

    def test_convert_auto(self, vertex_model):
        """Test converting 'auto' tool_choice."""
        result = vertex_model._convert_tool_choice_to_vertex("auto")
        assert result == {"function_calling_config": {"mode": "AUTO"}}

    def test_convert_required(self, vertex_model):
        """Test converting 'required' tool_choice."""
        result = vertex_model._convert_tool_choice_to_vertex("required")
        assert result == {"function_calling_config": {"mode": "ANY"}}

    def test_convert_none(self, vertex_model):
        """Test converting 'none' tool_choice."""
        result = vertex_model._convert_tool_choice_to_vertex("none")
        assert result == {"function_calling_config": {"mode": "NONE"}}

    def test_convert_specific_tool(self, vertex_model):
        """Test converting specific tool choice."""
        specific_choice = {"type": "function", "function": {"name": "get_weather"}}
        result = vertex_model._convert_tool_choice_to_vertex(specific_choice)
        assert result == {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["get_weather"]
            }
        }

    def test_convert_none_value(self, vertex_model):
        """Test converting None returns None."""
        result = vertex_model._convert_tool_choice_to_vertex(None)
        assert result is None


class TestToolCallResponse:
    """Tests for handling tool call responses."""

    def test_chat_complete_with_tools(self, vertex_model_with_tool_response, sample_tools):
        """Test chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = vertex_model_with_tool_response.chat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = vertex_model_with_tool_response.client.post.call_args
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

    def test_chat_complete_with_tool_choice(self, vertex_model_with_tool_response, sample_tools):
        """Test chat_complete with tool_choice parameter."""
        messages = [{"role": "user", "content": "What's the weather?"}]

        vertex_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice="required"
        )

        call_args = vertex_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tool_config" in json_payload
        assert json_payload["tool_config"]["function_calling_config"]["mode"] == "ANY"

    def test_chat_complete_with_specific_tool_choice(self, vertex_model_with_tool_response, sample_tools):
        """Test chat_complete with specific tool choice."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        specific_choice = {"type": "function", "function": {"name": "get_weather"}}

        vertex_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, tool_choice=specific_choice
        )

        call_args = vertex_model_with_tool_response.client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tool_config" in json_payload
        assert json_payload["tool_config"]["function_calling_config"]["allowed_function_names"] == ["get_weather"]

    @pytest.mark.asyncio
    async def test_achat_complete_with_tools(self, vertex_model_with_tool_response, sample_tools):
        """Test async chat_complete with tools returns tool calls."""
        messages = [{"role": "user", "content": "What's the weather in SF?"}]

        response = await vertex_model_with_tool_response.achat_complete(
            messages, tools=sample_tools
        )

        # Check payload included tools
        call_args = vertex_model_with_tool_response.async_client.post.call_args
        json_payload = call_args[1]["json"]
        assert "tools" in json_payload

        # Check response has tool calls
        assert response.choices[0].message.tool_calls is not None
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"


class TestInstanceLevelTools:
    """Tests for instance-level tool configuration."""

    def test_instance_tools_used_when_no_call_tools(
        self, mock_gcloud_auth, mock_vertex_tool_call_response, sample_tools
    ):
        """Test that instance-level tools are used when not passed at call time."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
            model = VertexLanguageModel(
                model_name="gemini-2.0-flash",
                tools=sample_tools
            )

            client = Mock()
            response = Mock()
            response.status_code = 200
            response.json.return_value = mock_vertex_tool_call_response
            client.post.return_value = response
            model.client = client

            messages = [{"role": "user", "content": "What's the weather?"}]
            model.chat_complete(messages)

            call_args = client.post.call_args
            json_payload = call_args[1]["json"]
            assert "tools" in json_payload
            declarations = json_payload["tools"][0]["function_declarations"]
            assert len(declarations) == 2

    def test_call_tools_override_instance_tools(
        self, mock_gcloud_auth, mock_vertex_tool_call_response, sample_tools
    ):
        """Test that call-time tools override instance-level tools."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
            instance_tool = Tool(
                function=ToolFunction(name="instance_tool", description="Instance tool")
            )
            model = VertexLanguageModel(
                model_name="gemini-2.0-flash",
                tools=[instance_tool]
            )

            client = Mock()
            response = Mock()
            response.status_code = 200
            response.json.return_value = mock_vertex_tool_call_response
            client.post.return_value = response
            model.client = client

            messages = [{"role": "user", "content": "What's the weather?"}]
            model.chat_complete(messages, tools=sample_tools)

            call_args = client.post.call_args
            json_payload = call_args[1]["json"]
            # Should have call-time tools, not instance tools
            declarations = json_payload["tools"][0]["function_declarations"]
            assert declarations[0]["name"] == "get_weather"

    def test_instance_tool_choice(
        self, mock_gcloud_auth, mock_vertex_tool_call_response, sample_tools
    ):
        """Test instance-level tool_choice is used."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
            model = VertexLanguageModel(
                model_name="gemini-2.0-flash",
                tools=sample_tools,
                tool_choice="required"
            )

            client = Mock()
            response = Mock()
            response.status_code = 200
            response.json.return_value = mock_vertex_tool_call_response
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
        self, vertex_model_with_tool_response, sample_tools
    ):
        """Test that validation passes for valid tool calls."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        # Should not raise
        response = vertex_model_with_tool_response.chat_complete(
            messages, tools=sample_tools, validate_tool_calls=True
        )
        assert response.choices[0].message.tool_calls is not None

    def test_validation_fails_for_invalid_tool_call(self, mock_gcloud_auth, sample_tools):
        """Test that validation fails for invalid tool calls."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
            model = VertexLanguageModel(model_name="gemini-2.0-flash")

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

    def test_format_tool_result_message(self, vertex_model):
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

        formatted, system_instruction = vertex_model._format_messages(messages)

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

    def test_format_assistant_with_text_and_tool_call(self, vertex_model):
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

        formatted, _ = vertex_model._format_messages(messages)

        assert formatted[0]["role"] == "model"
        # First part should be text
        assert formatted[0]["parts"][0]["text"] == "Let me check"
        # Second part should be functionCall
        assert "functionCall" in formatted[0]["parts"][1]


class TestNormalizeResponse:
    """Tests for response normalization."""

    def test_normalize_response_with_tool_calls(self, vertex_model, mock_vertex_tool_call_response):
        """Test normalizing response with tool calls."""
        result = vertex_model._normalize_response(mock_vertex_tool_call_response)

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
        self, vertex_model, mock_vertex_tool_call_with_text_response
    ):
        """Test normalizing response with both text and tool calls."""
        result = vertex_model._normalize_response(mock_vertex_tool_call_with_text_response)

        message = result.choices[0].message
        assert message.content == "Let me check the weather for you."
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1

    def test_normalize_response_finish_reason_tool_calls(
        self, vertex_model, mock_vertex_tool_call_response
    ):
        """Test that finish_reason is 'tool_calls' when tools are called."""
        result = vertex_model._normalize_response(mock_vertex_tool_call_response)
        assert result.choices[0].finish_reason == "tool_calls"

    def test_normalize_response_without_tool_calls(self, vertex_model, mock_vertex_chat_response):
        """Test normalizing response without tool calls."""
        result = vertex_model._normalize_response(mock_vertex_chat_response)

        message = result.choices[0].message
        assert message.content == "Hello! How can I help you today?"
        assert message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"


class TestStreamingWithTools:
    """Tests for streaming with tool calls."""

    def test_normalize_chunk_with_tool_call(self, vertex_model):
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

        result = vertex_model._normalize_chunk(chunk_data)

        assert result is not None
        delta = result.choices[0].delta
        assert delta.tool_calls is not None
        assert len(delta.tool_calls) == 1
        # Streaming tool_calls are ToolCall objects due to Message validation
        tool_call = delta.tool_calls[0]
        assert tool_call.function.name == "get_weather"

    def test_streaming_chat_complete_with_tools(
        self, mock_gcloud_auth, sample_tools, mock_vertex_tool_call_response
    ):
        """Test streaming chat_complete with tools."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}):
            model = VertexLanguageModel(model_name="gemini-2.0-flash")

            client = Mock()
            response = Mock()
            response.status_code = 200
            response.json.return_value = mock_vertex_tool_call_response
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

    def test_handle_api_error(self, vertex_model):
        """Test handling API errors."""
        client = Mock()
        response = Mock()
        response.status_code = 400
        response.json.return_value = {"error": {"message": "Invalid request"}}
        client.post.return_value = response
        vertex_model.client = client

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RuntimeError, match="Vertex AI API error"):
            vertex_model.chat_complete(messages)


class TestCredentials:
    """Tests for credential loading and token management."""

    def test_credentials_file_loading(self):
        """Test loading credentials from an explicit credentials_file parameter."""
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "sa-token"

        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            with patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                return_value=mock_creds,
            ) as mock_from_file:
                model = VertexLanguageModel(
                    model_name="gemini-2.0-flash",
                    credentials_file="/path/to/sa.json",
                )
                mock_from_file.assert_called_once_with(
                    "/path/to/sa.json",
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                assert model._credentials is mock_creds

    def test_google_application_credentials_env_var(self):
        """Test loading credentials from GOOGLE_APPLICATION_CREDENTIALS env var."""
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "env-sa-token"

        env = {
            "VERTEX_PROJECT": "test-project",
            "GOOGLE_APPLICATION_CREDENTIALS": "/env/path/sa.json",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                return_value=mock_creds,
            ) as mock_from_file:
                model = VertexLanguageModel(model_name="gemini-2.0-flash")
                mock_from_file.assert_called_once_with(
                    "/env/path/sa.json",
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                assert model._credentials is mock_creds

    def test_credentials_file_takes_priority_over_env_var(self):
        """Test that explicit credentials_file takes priority over env var."""
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "explicit-token"

        env = {
            "VERTEX_PROJECT": "test-project",
            "GOOGLE_APPLICATION_CREDENTIALS": "/env/path/sa.json",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                return_value=mock_creds,
            ) as mock_from_file:
                model = VertexLanguageModel(
                    model_name="gemini-2.0-flash",
                    credentials_file="/explicit/sa.json",
                )
                mock_from_file.assert_called_once_with(
                    "/explicit/sa.json",
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

    def test_adc_fallback_when_no_credentials_file(self, mock_google_auth):
        """Test that ADC is used when no credentials file is provided."""
        mock_default, mock_creds = mock_google_auth
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            # Remove GOOGLE_APPLICATION_CREDENTIALS if set
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            model = VertexLanguageModel(model_name="gemini-2.0-flash")
            mock_default.assert_called_once()
            assert model._credentials is mock_creds

    def test_gcloud_fallback_when_no_google_auth_installed(self):
        """Test that gcloud CLI fallback works when google-auth is not installed."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

            # Mock _load_credentials to simulate google-auth not being installed
            with patch.object(
                VertexLanguageModel,
                "_load_credentials",
                lambda self: setattr(self, "_credentials", None),
            ):
                model = VertexLanguageModel(model_name="gemini-2.0-flash")
                assert model._credentials is None

    def test_token_from_credentials(self, mock_google_auth):
        """Test that _get_access_token uses credentials when available."""
        _, mock_creds = mock_google_auth
        mock_creds.valid = True
        mock_creds.token = "creds-token"

        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            model = VertexLanguageModel(model_name="gemini-2.0-flash")
            with patch("google.auth.transport.requests.Request"):
                token = model._get_access_token()
                assert token == "creds-token"

    def test_token_refresh_when_expired(self, mock_google_auth):
        """Test that expired credentials are refreshed."""
        _, mock_creds = mock_google_auth
        mock_creds.valid = False
        mock_creds.token = "refreshed-token"

        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            model = VertexLanguageModel(model_name="gemini-2.0-flash")
            with patch("google.auth.transport.requests.Request") as mock_request:
                token = model._get_access_token()
                mock_creds.refresh.assert_called_once_with(mock_request())
                assert token == "refreshed-token"

    def test_gcloud_cli_fallback_for_token(self):
        """Test gcloud CLI fallback when _credentials is None."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            with patch("google.auth.default", side_effect=Exception("no ADC")):
                model = VertexLanguageModel(model_name="gemini-2.0-flash")
                assert model._credentials is None

                with patch("subprocess.run") as mock_run:
                    mock_result = Mock()
                    mock_result.stdout = "gcloud-token"
                    mock_run.return_value = mock_result
                    token = model._get_access_token()
                    assert token == "gcloud-token"

    def test_credentials_file_raises_on_invalid_file(self):
        """Test that an invalid credentials_file raises instead of silently falling back."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            with patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                side_effect=FileNotFoundError("No such file"),
            ):
                with pytest.raises(FileNotFoundError, match="No such file"):
                    VertexLanguageModel(
                        model_name="gemini-2.0-flash",
                        credentials_file="/nonexistent/sa.json",
                    )

    def test_credentials_file_raises_on_missing_google_auth(self):
        """Test that credentials_file raises ImportError when google-auth is not installed."""
        import builtins

        original_import = builtins.__import__

        def selective_import(name, *args, **kwargs):
            if name == "google.oauth2.service_account" or (
                name == "google.oauth2" and args and "service_account" in (args[2] or ())
            ):
                raise ImportError("no google-auth")
            return original_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            with patch("builtins.__import__", side_effect=selective_import):
                with pytest.raises(ImportError, match="google-auth"):
                    VertexLanguageModel(
                        model_name="gemini-2.0-flash",
                        credentials_file="/path/sa.json",
                    )

    def test_env_var_credentials_raises_on_invalid_file(self):
        """Test that GOOGLE_APPLICATION_CREDENTIALS raises on invalid file instead of falling back."""
        env = {
            "VERTEX_PROJECT": "test-project",
            "GOOGLE_APPLICATION_CREDENTIALS": "/bad/path/sa.json",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                side_effect=ValueError("Invalid service account info"),
            ):
                with pytest.raises(ValueError, match="Invalid service account"):
                    VertexLanguageModel(model_name="gemini-2.0-flash")


class TestLangChainIntegration:
    """Tests for LangChain integration using ChatGoogleGenerativeAI."""

    def test_to_langchain_uses_chat_google_generative_ai(self, mock_google_auth):
        """Test that to_langchain creates ChatGoogleGenerativeAI."""
        _, mock_creds = mock_google_auth
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            model = VertexLanguageModel(model_name="gemini-2.0-flash")

            mock_lc_class = MagicMock()
            with patch.dict(
                "sys.modules",
                {"langchain_google_genai": MagicMock(ChatGoogleGenerativeAI=mock_lc_class)},
            ):
                with patch(
                    "esperanto.providers.llm.vertex.ChatGoogleGenerativeAI",
                    mock_lc_class,
                    create=True,
                ):
                    # Patch the import inside to_langchain
                    import importlib
                    lc_module = MagicMock()
                    lc_module.ChatGoogleGenerativeAI = mock_lc_class
                    with patch.dict("sys.modules", {"langchain_google_genai": lc_module}):
                        result = model.to_langchain()
                        mock_lc_class.assert_called_once()
                        call_kwargs = mock_lc_class.call_args[1]
                        assert call_kwargs["model"] == "gemini-2.0-flash"
                        assert call_kwargs["project"] == "test-project"
                        assert call_kwargs["location"] == "us-central1"
                        assert call_kwargs["credentials"] is mock_creds

    def test_to_langchain_caches_model(self, mock_google_auth):
        """Test that to_langchain caches the LangChain model instance."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            model = VertexLanguageModel(model_name="gemini-2.0-flash")

            mock_lc_class = MagicMock()
            lc_module = MagicMock()
            lc_module.ChatGoogleGenerativeAI = mock_lc_class
            with patch.dict("sys.modules", {"langchain_google_genai": lc_module}):
                first = model.to_langchain()
                second = model.to_langchain()
                # Should only instantiate once
                assert mock_lc_class.call_count == 1
                assert first is second

    def test_to_langchain_without_credentials(self):
        """Test that to_langchain omits credentials when None."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            with patch("google.auth.default", side_effect=Exception("no ADC")):
                model = VertexLanguageModel(model_name="gemini-2.0-flash")
                assert model._credentials is None

                mock_lc_class = MagicMock()
                lc_module = MagicMock()
                lc_module.ChatGoogleGenerativeAI = mock_lc_class
                with patch.dict("sys.modules", {"langchain_google_genai": lc_module}):
                    model.to_langchain()
                    call_kwargs = mock_lc_class.call_args[1]
                    assert "credentials" not in call_kwargs

    def test_to_langchain_import_error(self, mock_google_auth):
        """Test that to_langchain raises ImportError when langchain_google_genai is missing."""
        with patch.dict(os.environ, {"VERTEX_PROJECT": "test-project"}, clear=False):
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            model = VertexLanguageModel(model_name="gemini-2.0-flash")

            with patch.dict("sys.modules", {"langchain_google_genai": None}):
                with pytest.raises(ImportError, match="langchain_google_genai"):
                    model.to_langchain()
