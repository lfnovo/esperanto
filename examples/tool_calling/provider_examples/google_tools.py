#!/usr/bin/env python3
"""Google (Gemini) tool calling example.

Demonstrates tool calling with Google Gemini models, including:
- Tool definition (automatically converted to Google's function_declarations format)
- Multi-turn conversation with tool results
- tool_choice options (AUTO, ANY, NONE)

Environment variables required:
- GOOGLE_API_KEY: Your Google AI API key

Run with:
    python examples/tool_calling/provider_examples/google_tools.py
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
                name="get_current_time",
                description="Get the current time in a specific timezone",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone, e.g., 'America/New_York' or 'Europe/London'",
                        }
                    },
                    "required": ["timezone"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="convert_currency",
                description="Convert an amount from one currency to another",
                parameters={
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "The amount to convert",
                        },
                        "from_currency": {
                            "type": "string",
                            "description": "The source currency code (e.g., 'USD')",
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "The target currency code (e.g., 'EUR')",
                        },
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution."""
    if name == "get_current_time":
        # Simulated response
        return {
            "timezone": args.get("timezone"),
            "time": "14:30:00",
            "date": "2024-01-15",
        }
    elif name == "convert_currency":
        # Simulated exchange rate
        rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 148.5}
        amount = args.get("amount", 0)
        from_curr = args.get("from_currency", "USD")
        to_curr = args.get("to_currency", "EUR")

        # Simple conversion (not real rates)
        usd_amount = amount / rates.get(from_curr, 1.0)
        converted = usd_amount * rates.get(to_curr, 1.0)

        return {
            "original_amount": amount,
            "from_currency": from_curr,
            "converted_amount": round(converted, 2),
            "to_currency": to_curr,
            "exchange_rate": round(rates.get(to_curr, 1.0) / rates.get(from_curr, 1.0), 4),
        }
    return {"error": f"Unknown tool: {name}"}


def main():
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY environment variable")
        return

    print("=== Google (Gemini) Tool Calling Example ===\n")

    tools = create_tools()

    # Create model
    model = AIFactory.create_language(
        provider="google",
        model_name="gemini-2.0-flash",
    )

    # Example 1: Basic tool call
    print("1. Basic Tool Call")
    print("-" * 40)
    messages = [{"role": "user", "content": "What time is it in Tokyo?"}]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Tool: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
        print(f"Tool call ID: {tc.id}")  # Note: Esperanto generates IDs for Google
    else:
        print(f"Response: {response.choices[0].message.content}")
    print()

    # Example 2: Multi-turn with tool results
    print("2. Multi-turn Conversation with Tool Results")
    print("-" * 40)

    messages = [{"role": "user", "content": "Convert 100 USD to EUR please"}]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        # Add assistant's tool call
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

        # Execute and add results
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
        print(f"\nGemini's response: {final_response.choices[0].message.content}")
    print()

    # Example 3: tool_choice options
    print("3. Tool Choice Options")
    print("-" * 40)

    # AUTO (default) - model decides
    print("tool_choice='auto' (default): Model decides whether to use tools")

    # REQUIRED - must use a tool
    messages = [{"role": "user", "content": "Hello!"}]
    response = model.chat_complete(messages, tools=tools, tool_choice="required")

    if response.choices[0].message.tool_calls:
        print(f"  -> Model was forced to call: {response.choices[0].message.tool_calls[0].function.name}")
    else:
        print("  -> Model responded with text (tool_choice may not be fully supported)")
    print()

    print("=== Example Complete ===")
    print("\nNote: Esperanto automatically handles Google-specific format:")
    print("  - Converts tools to 'function_declarations' format")
    print("  - Generates tool call IDs (Google doesn't provide them)")
    print("  - Converts 'functionCall' responses to standard ToolCall format")


if __name__ == "__main__":
    main()
