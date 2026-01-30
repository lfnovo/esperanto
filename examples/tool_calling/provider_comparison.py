#!/usr/bin/env python3
"""Provider comparison example for tool calling with Esperanto.

This example demonstrates Esperanto's key value proposition:
the same tool calling code works across all providers.

Environment variables (set the ones for providers you want to test):
- OPENAI_API_KEY: OpenAI API key
- ANTHROPIC_API_KEY: Anthropic API key
- GOOGLE_API_KEY: Google (Gemini) API key
- GROQ_API_KEY: Groq API key
- MISTRAL_API_KEY: Mistral API key
- DEEPSEEK_API_KEY: DeepSeek API key
- XAI_API_KEY: xAI API key

Run with:
    python examples/tool_calling/provider_comparison.py
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction


def create_tools() -> List[Tool]:
    """Create a set of tools that work across all providers."""

    # Simple calculator tool
    calculator = Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Perform a mathematical calculation",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        ),
    )

    # Weather tool
    weather = Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get the current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["city"],
            },
        ),
    )

    return [calculator, weather]


def get_available_providers() -> List[Tuple[str, str, str]]:
    """Get list of available providers based on environment variables.

    Returns:
        List of (provider_name, model_name, env_var_name) tuples.
    """
    providers = [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic", "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
        ("google", "gemini-2.0-flash", "GOOGLE_API_KEY"),
        ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        ("mistral", "mistral-small-latest", "MISTRAL_API_KEY"),
        ("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY"),
        ("xai", "grok-2-latest", "XAI_API_KEY"),
    ]

    available = []
    for provider, model, env_var in providers:
        if os.getenv(env_var):
            available.append((provider, model, env_var))

    return available


def test_provider(
    provider: str, model: str, tools: List[Tool], query: str
) -> Dict[str, Any]:
    """Test tool calling with a specific provider.

    Returns:
        Dict with results including success status, tool calls, and any errors.
    """
    result: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "success": False,
        "tool_calls": None,
        "content": None,
        "error": None,
    }

    try:
        # Create the model - same code for all providers!
        llm = AIFactory.create_language(provider=provider, model_name=model)

        # Make the request - identical across providers
        messages = [{"role": "user", "content": query}]
        response = llm.chat_complete(messages, tools=tools)

        message = response.choices[0].message
        result["success"] = True

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]
        else:
            result["content"] = message.content

    except Exception as e:
        result["error"] = str(e)

    return result


def format_result(result: Dict[str, Any]) -> str:
    """Format a test result for display."""
    lines = [f"  Provider: {result['provider']} ({result['model']})"]

    if result["error"]:
        lines.append(f"  Status: FAILED")
        lines.append(f"  Error: {result['error'][:100]}...")
    elif result["tool_calls"]:
        lines.append(f"  Status: SUCCESS (tool call)")
        for tc in result["tool_calls"]:
            lines.append(f"  Tool: {tc['name']}")
            lines.append(f"  Args: {tc['arguments']}")
    else:
        lines.append(f"  Status: SUCCESS (text response)")
        content = result["content"] or ""
        lines.append(f"  Response: {content[:100]}...")

    return "\n".join(lines)


def main():
    print("=== Provider Comparison: Tool Calling ===\n")

    # Get available providers
    available = get_available_providers()

    if not available:
        print("No provider API keys found!")
        print("\nPlease set at least one of these environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GOOGLE_API_KEY")
        print("  - GROQ_API_KEY")
        print("  - MISTRAL_API_KEY")
        print("  - DEEPSEEK_API_KEY")
        print("  - XAI_API_KEY")
        return

    print(f"Found {len(available)} provider(s) with API keys:\n")
    for provider, model, _ in available:
        print(f"  - {provider}: {model}")
    print()

    # Create tools - same tools work for all providers
    tools = create_tools()
    print(f"Tools defined: {[t.function.name for t in tools]}\n")

    # Test queries
    test_cases = [
        ("What is 15 * 23?", "Should trigger calculate tool"),
        ("What's the weather in Paris?", "Should trigger get_weather tool"),
    ]

    for query, description in test_cases:
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Expected: {description}")
        print("=" * 60)
        print()

        for provider, model, _ in available:
            result = test_provider(provider, model, tools, query)
            print(format_result(result))
            print()

    # Summary
    print("=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
The same tool definition and calling code works across all providers:

    # Define tools once
    tools = [Tool(...), Tool(...)]

    # Works with ANY provider
    model = AIFactory.create_language(provider="openai", ...)
    model = AIFactory.create_language(provider="anthropic", ...)
    model = AIFactory.create_language(provider="google", ...)

    # Identical API call
    response = model.chat_complete(messages, tools=tools)

Esperanto handles all the format conversion internally:
- OpenAI format: {"type": "function", "function": {...}}
- Anthropic format: {"name": ..., "input_schema": {...}}
- Google format: {"function_declarations": [...]}

You write code once, it works everywhere.
""")

    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
