#!/usr/bin/env python3
"""Multiple tools example with Esperanto.

This example demonstrates using multiple tools:
1. Define several tools for different purposes
2. Let the model choose which tool(s) to call
3. Handle different tool call types

Environment variables required:
- OPENAI_API_KEY: Your OpenAI API key

Run with:
    python examples/tool_calling/multiple_tools.py
"""

import json
import os

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction


def create_tools():
    """Create a set of tools for a hypothetical assistant."""

    # Weather tool
    weather_tool = Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'Paris, France'",
                    }
                },
                "required": ["location"],
            },
        ),
    )

    # Calculator tool
    calculator_tool = Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g., '2 + 2' or '15 * 7'",
                    }
                },
                "required": ["expression"],
            },
        ),
    )

    # Search tool
    search_tool = Tool(
        type="function",
        function=ToolFunction(
            name="search_web",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
    )

    # Time tool
    time_tool = Tool(
        type="function",
        function=ToolFunction(
            name="get_current_time",
            description="Get the current time in a specific timezone",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name, e.g., 'America/New_York' or 'Europe/London'",
                    }
                },
                "required": ["timezone"],
            },
        ),
    )

    return [weather_tool, calculator_tool, search_tool, time_tool]


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("Export it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Create tools
    tools = create_tools()

    # Create model
    model = AIFactory.create_language(
        provider="openai",
        model_name="gpt-4o-mini",
        config={"tools": tools},
    )

    print("=== Multiple Tools Example ===\n")
    print(f"Available tools: {[t.function.name for t in tools]}\n")

    # Test different queries that should trigger different tools
    test_queries = [
        "What's 15 multiplied by 23?",
        "What's the weather in Berlin?",
        "What time is it in Tokyo?",
        "Search for the latest news about AI",
    ]

    for query in test_queries:
        print(f"User: {query}")
        messages = [{"role": "user", "content": query}]

        response = model.chat_complete(messages)
        message = response.choices[0].message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                print(f"  -> Tool: {tool_call.function.name}")
                print(f"     Args: {args}")
        else:
            print(f"  -> Response: {message.content[:100]}...")

        print()

    # Test a query that might trigger multiple tools (parallel tool calls)
    print("--- Testing parallel tool calls ---\n")
    complex_query = "What's the weather in New York and what time is it there?"
    print(f"User: {complex_query}")

    messages = [{"role": "user", "content": complex_query}]

    # Enable parallel tool calls
    response = model.chat_complete(messages, parallel_tool_calls=True)
    message = response.choices[0].message

    if message.tool_calls:
        print(f"Model requested {len(message.tool_calls)} tool call(s):")
        for i, tool_call in enumerate(message.tool_calls, 1):
            args = json.loads(tool_call.function.arguments)
            print(f"  {i}. {tool_call.function.name}: {args}")
    else:
        print(f"  Response: {message.content}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
