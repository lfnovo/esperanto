"""Real integration tests for tool calling - these call actual APIs.

These tests verify that tool calling works correctly with real API calls.
They require API keys to be configured in the environment.

Run with: uv run pytest tests/integration/test_tool_calling_real.py -v -s
"""

import json
import os

import pytest

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction, ToolCall


# =============================================================================
# Test Configuration
# =============================================================================

# Simple tool for testing - get weather for a city
WEATHER_TOOL = Tool(
    type="function",
    function=ToolFunction(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'San Francisco'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
    ),
)

# Message that should trigger tool use
TOOL_TRIGGER_MESSAGE = [
    {"role": "user", "content": "What's the weather like in Tokyo?"}
]


def simulate_weather_response(city: str, unit: str = "celsius") -> str:
    """Simulate a weather API response."""
    return json.dumps({
        "city": city,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny",
    })


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def weather_tools():
    """Return the weather tool for testing."""
    return [WEATHER_TOOL]


# =============================================================================
# OpenAI Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not configured",
)
class TestOpenAIToolCalling:
    """Real integration tests for OpenAI tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that OpenAI returns a tool call for a weather query."""
        model = AIFactory.create_language("openai", "gpt-4o-mini")

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        # Should have a response with tool calls
        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nOpenAI response: {message}")

        # Should have tool calls
        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        # Verify tool call structure
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        # Verify arguments are valid JSON with city
        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language("openai", "gpt-4o-mini")

        # First call - should get tool call
        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Second call - send tool result back
        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        # Should have a text response using the weather data
        message = response2.choices[0].message
        print(f"\nOpenAI final response: {message.content}")

        assert message.content is not None
        # Response should mention temperature or weather condition
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# Anthropic Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not configured",
)
class TestAnthropicToolCalling:
    """Real integration tests for Anthropic tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that Anthropic returns a tool call for a weather query."""
        model = AIFactory.create_language("anthropic", "claude-3-5-haiku-latest")

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nAnthropic response: {message}")

        # Should have tool calls
        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        # Verify tool call structure
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        # Verify arguments are valid JSON with city
        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language("anthropic", "claude-3-5-haiku-latest")

        # First call - should get tool call
        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Second call - send tool result back
        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nAnthropic final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# Groq Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not configured",
)
class TestGroqToolCalling:
    """Real integration tests for Groq tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that Groq returns a tool call for a weather query."""
        model = AIFactory.create_language("groq", "llama-3.3-70b-versatile")

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nGroq response: {message}")

        # Should have tool calls
        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        # Verify tool call structure
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        # Verify arguments are valid JSON with city
        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language("groq", "llama-3.3-70b-versatile")

        # First call - should get tool call
        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Second call - send tool result back
        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nGroq final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# Mistral Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not configured",
)
class TestMistralToolCalling:
    """Real integration tests for Mistral tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that Mistral returns a tool call for a weather query."""
        model = AIFactory.create_language("mistral", "mistral-small-latest")

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nMistral response: {message}")

        # Should have tool calls
        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        # Verify tool call structure
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        # Verify arguments are valid JSON with city
        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language("mistral", "mistral-small-latest")

        # First call - should get tool call
        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Second call - send tool result back
        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nMistral final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# DeepSeek Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not configured",
)
class TestDeepSeekToolCalling:
    """Real integration tests for DeepSeek tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that DeepSeek returns a tool call for a weather query."""
        model = AIFactory.create_language("deepseek", "deepseek-chat")

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nDeepSeek response: {message}")

        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language("deepseek", "deepseek-chat")

        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nDeepSeek final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# xAI Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("XAI_API_KEY"),
    reason="XAI_API_KEY not configured",
)
class TestXAIToolCalling:
    """Real integration tests for xAI tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that xAI returns a tool call for a weather query."""
        model = AIFactory.create_language("xai", "grok-3")

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nxAI response: {message}")

        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language("xai", "grok-3")

        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nxAI final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# OpenRouter Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not configured",
)
class TestOpenRouterToolCalling:
    """Real integration tests for OpenRouter tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that OpenRouter returns a tool call for a weather query."""
        # Use a model that supports tool calling via OpenRouter
        model = AIFactory.create_language("openrouter", "openai/gpt-4o-mini")

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nOpenRouter response: {message}")

        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language("openrouter", "openai/gpt-4o-mini")

        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nOpenRouter final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# Perplexity Tests
# =============================================================================


@pytest.mark.skip(reason="Perplexity does not support tool calling")
class TestPerplexityToolCalling:
    """Real integration tests for Perplexity tool calling.

    NOTE: Perplexity API does not support tool calling. These tests are skipped.
    """

    def test_basic_tool_call(self, weather_tools):
        """Test that Perplexity returns a tool call for a weather query."""
        model = AIFactory.create_language("perplexity", "sonar")

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nPerplexity response: {message}")

        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language("perplexity", "sonar")

        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nPerplexity final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# Azure Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY_LLM"),
    reason="AZURE_OPENAI_API_KEY_LLM not configured",
)
class TestAzureToolCalling:
    """Real integration tests for Azure OpenAI tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that Azure returns a tool call for a weather query."""
        model = AIFactory.create_language(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LLM", "gpt-4o-mini"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_LLM"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_LLM"),
                "api_version": os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            },
        )

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nAzure response: {message}")

        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LLM", "gpt-4o-mini"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_LLM"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_LLM"),
                "api_version": os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            },
        )

        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nAzure final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# Ollama Tests
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("OLLAMA_BASE_URL"),
    reason="OLLAMA_BASE_URL not configured",
)
class TestOllamaToolCalling:
    """Real integration tests for Ollama tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that Ollama returns a tool call for a weather query."""
        # Use a model that supports tool calling
        model = AIFactory.create_language(
            "ollama",
            "qwen3:32b",
            config={"base_url": os.getenv("OLLAMA_BASE_URL")},
        )

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nOllama response: {message}")

        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        model = AIFactory.create_language(
            "ollama",
            "qwen3:32b",
            config={"base_url": os.getenv("OLLAMA_BASE_URL")},
        )

        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nOllama final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"


# =============================================================================
# Google Tests
# =============================================================================


@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
class TestGoogleToolCalling:
    """Real integration tests for Google tool calling."""

    def test_basic_tool_call(self, weather_tools):
        """Test that Google returns a tool call for a weather query."""
        # Google provider accepts either GOOGLE_API_KEY or GEMINI_API_KEY
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_language(
            "google", "gemini-2.0-flash", config={"api_key": api_key}
        )

        response = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response is not None
        assert len(response.choices) > 0

        message = response.choices[0].message
        print(f"\nGoogle response: {message}")

        # Should have tool calls
        assert message.tool_calls is not None, "Expected tool_calls but got None"
        assert len(message.tool_calls) > 0, "Expected at least one tool call"

        # Verify tool call structure
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.function.name == "get_weather"

        # Verify arguments are valid JSON with city
        args = json.loads(tool_call.function.arguments)
        assert "city" in args
        print(f"Tool call arguments: {args}")

    def test_multi_turn_with_tool_result(self, weather_tools):
        """Test multi-turn conversation with tool result."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_language(
            "google", "gemini-2.0-flash", config={"api_key": api_key}
        )

        # First call - should get tool call
        response1 = model.chat_complete(
            messages=TOOL_TRIGGER_MESSAGE,
            tools=weather_tools,
        )

        assert response1.choices[0].message.tool_calls is not None
        tool_call = response1.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Second call - send tool result back
        messages = [
            TOOL_TRIGGER_MESSAGE[0],
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": simulate_weather_response(args.get("city", "Tokyo")),
            },
        ]

        response2 = model.chat_complete(messages=messages, tools=weather_tools)

        message = response2.choices[0].message
        print(f"\nGoogle final response: {message.content}")

        assert message.content is not None
        content_lower = message.content.lower()
        assert any(
            word in content_lower
            for word in ["22", "72", "sunny", "temperature", "weather", "tokyo"]
        ), f"Expected weather-related response, got: {message.content}"
