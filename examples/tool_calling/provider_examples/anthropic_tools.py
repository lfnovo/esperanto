#!/usr/bin/env python3
"""Anthropic (Claude) tool calling example.

Demonstrates tool calling with Anthropic Claude models, including:
- Tool definition (automatically converted to Anthropic's input_schema format)
- Multi-turn conversation with tool results
- tool_choice options

Environment variables required:
- ANTHROPIC_API_KEY: Your Anthropic API key

Run with:
    python examples/tool_calling/provider_examples/anthropic_tools.py
"""

import json
import os
from typing import Any, Dict

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction


def create_tools():
    """Create tools for the example."""
    return [
        Tool(
            type="function",
            function=ToolFunction(
                name="get_weather",
                description="Get the current weather in a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="search_database",
                description="Search a database for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution."""
    if name == "get_weather":
        return {
            "location": args.get("location"),
            "temperature": 72,
            "unit": args.get("unit", "fahrenheit"),
            "condition": "sunny",
            "humidity": 45,
        }
    elif name == "search_database":
        return {
            "query": args.get("query"),
            "results": [
                {"id": 1, "title": "Result 1", "relevance": 0.95},
                {"id": 2, "title": "Result 2", "relevance": 0.87},
            ],
            "total": 2,
        }
    return {"error": f"Unknown tool: {name}"}


def main():
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        return

    print("=== Anthropic (Claude) Tool Calling Example ===\n")

    tools = create_tools()

    # Create model - using Claude 3.5 Haiku for fast, cost-effective responses
    model = AIFactory.create_language(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
    )

    # Example 1: Basic tool call
    print("1. Basic Tool Call")
    print("-" * 40)
    messages = [{"role": "user", "content": "What's the weather in Paris, France?"}]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Tool: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
    print()

    # Example 2: Multi-turn conversation with tool results
    print("2. Multi-turn Conversation with Tool Results")
    print("-" * 40)

    # Initial request
    messages = [{"role": "user", "content": "Can you check the weather in London?"}]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        # Add assistant's tool call to messages
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.choices[0].message.tool_calls
                ],
            }
        )

        # Execute tool and add result
        for tc in response.choices[0].message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            print(f"Executed {tc.function.name} -> {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        # Get final response
        final_response = model.chat_complete(messages, tools=tools)
        print(f"\nClaude's response: {final_response.choices[0].message.content}")
    print()

    # Example 3: Using tool_choice="required"
    print("3. Forcing Tool Use (tool_choice='required')")
    print("-" * 40)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = model.chat_complete(
        messages,
        tools=tools,
        tool_choice="required",  # Model must call a tool
    )

    if response.choices[0].message.tool_calls:
        print("Model was forced to call a tool:")
        for tc in response.choices[0].message.tool_calls:
            print(f"  - {tc.function.name}: {json.loads(tc.function.arguments)}")
    print()

    print("=== Example Complete ===")
    print("\nNote: Esperanto automatically converts tools to Anthropic's format:")
    print("  - 'parameters' -> 'input_schema'")
    print("  - 'tool_choice' -> Anthropic's format ({'type': 'auto'/'any'/'tool'})")


if __name__ == "__main__":
    main()
