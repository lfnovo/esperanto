#!/usr/bin/env python3
"""OpenAI tool calling example.

Demonstrates tool calling with OpenAI models, including:
- Basic tool definition and usage
- tool_choice options
- parallel_tool_calls parameter
- Strict mode for guaranteed schema adherence

Environment variables required:
- OPENAI_API_KEY: Your OpenAI API key

Run with:
    python examples/tool_calling/provider_examples/openai_tools.py
"""

import json
import os

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("=== OpenAI Tool Calling Example ===\n")

    # Define tools - OpenAI supports the 'strict' parameter for guaranteed schema adherence
    tools = [
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
                        },
                    },
                    "required": ["location"],
                    "additionalProperties": False,  # Required for strict mode
                },
                strict=True,  # OpenAI-specific: ensures exact schema adherence
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="get_stock_price",
                description="Get the current stock price for a ticker symbol",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock ticker symbol, e.g. AAPL",
                        }
                    },
                    "required": ["symbol"],
                    "additionalProperties": False,
                },
                strict=True,
            ),
        ),
    ]

    # Create model
    model = AIFactory.create_language(
        provider="openai",
        model_name="gpt-4o-mini",  # Most cost-effective model with tool support
    )

    # Example 1: Basic tool call
    print("1. Basic Tool Call")
    print("-" * 40)
    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Tool: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
    print()

    # Example 2: Forcing a specific tool with tool_choice
    print("2. Forcing Specific Tool (tool_choice)")
    print("-" * 40)
    messages = [{"role": "user", "content": "Tell me about NVIDIA"}]
    response = model.chat_complete(
        messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_stock_price"}},
    )

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Forced tool: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
    print()

    # Example 3: Parallel tool calls
    print("3. Parallel Tool Calls")
    print("-" * 40)
    messages = [
        {
            "role": "user",
            "content": "What's the weather in Tokyo and what's Apple's stock price?",
        }
    ]
    response = model.chat_complete(
        messages,
        tools=tools,
        parallel_tool_calls=True,  # Allow multiple tool calls
    )

    if response.choices[0].message.tool_calls:
        print(f"Number of tool calls: {len(response.choices[0].message.tool_calls)}")
        for i, tc in enumerate(response.choices[0].message.tool_calls, 1):
            print(f"  {i}. {tc.function.name}: {json.loads(tc.function.arguments)}")
    print()

    # Example 4: Preventing tool calls with tool_choice="none"
    print("4. Preventing Tool Calls (tool_choice='none')")
    print("-" * 40)
    messages = [{"role": "user", "content": "What's the weather like today?"}]
    response = model.chat_complete(
        messages,
        tools=tools,
        tool_choice="none",  # Model cannot use tools
    )

    print(f"Response (no tools): {response.choices[0].message.content[:100]}...")
    print()

    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
