#!/usr/bin/env python3
"""Mistral tool calling example.

Demonstrates tool calling with Mistral AI models.
Mistral uses OpenAI-compatible format for tools.

Environment variables required:
- MISTRAL_API_KEY: Your Mistral API key

Run with:
    python examples/tool_calling/provider_examples/mistral_tools.py
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
                name="get_flight_info",
                description="Get information about a flight",
                parameters={
                    "type": "object",
                    "properties": {
                        "flight_number": {
                            "type": "string",
                            "description": "The flight number, e.g., 'UA123'",
                        },
                        "date": {
                            "type": "string",
                            "description": "The flight date in YYYY-MM-DD format",
                        },
                    },
                    "required": ["flight_number"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="book_hotel",
                description="Book a hotel room",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city for the hotel",
                        },
                        "check_in": {
                            "type": "string",
                            "description": "Check-in date (YYYY-MM-DD)",
                        },
                        "check_out": {
                            "type": "string",
                            "description": "Check-out date (YYYY-MM-DD)",
                        },
                        "guests": {
                            "type": "integer",
                            "description": "Number of guests",
                            "default": 1,
                        },
                    },
                    "required": ["city", "check_in", "check_out"],
                },
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution."""
    if name == "get_flight_info":
        return {
            "flight_number": args.get("flight_number"),
            "date": args.get("date", "2024-01-20"),
            "departure": "SFO",
            "arrival": "JFK",
            "departure_time": "08:00",
            "arrival_time": "16:30",
            "status": "On Time",
        }
    elif name == "book_hotel":
        return {
            "confirmation": "HTL-" + str(hash(args.get("city", "")))[1:7],
            "hotel": f"Grand Hotel {args.get('city')}",
            "city": args.get("city"),
            "check_in": args.get("check_in"),
            "check_out": args.get("check_out"),
            "guests": args.get("guests", 1),
            "price_per_night": 150.00,
        }
    return {"error": f"Unknown tool: {name}"}


def main():
    # Check for API key
    if not os.getenv("MISTRAL_API_KEY"):
        print("Please set MISTRAL_API_KEY environment variable")
        return

    print("=== Mistral Tool Calling Example ===\n")

    tools = create_tools()

    # Create model
    model = AIFactory.create_language(
        provider="mistral",
        model_name="mistral-small-latest",
    )

    # Example 1: Basic tool call
    print("1. Basic Tool Call")
    print("-" * 40)
    messages = [{"role": "user", "content": "What's the status of flight UA456?"}]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Tool: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
    print()

    # Example 2: Multi-turn travel booking
    print("2. Multi-turn Travel Booking")
    print("-" * 40)

    messages = [
        {
            "role": "user",
            "content": "I need to check flight AA100 for tomorrow and book a hotel in New York from Jan 20 to Jan 23",
        }
    ]

    response = model.chat_complete(messages, tools=tools)

    # Process tool calls
    iteration = 0
    while response.choices[0].message.tool_calls and iteration < 5:
        iteration += 1
        tool_calls = response.choices[0].message.tool_calls

        # Add assistant message
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
                    for tc in tool_calls
                ],
            }
        )

        # Execute each tool
        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            print(f"  [{tc.function.name}] {args} -> {json.dumps(result)[:60]}...")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        response = model.chat_complete(messages, tools=tools)

    print(f"\nMistral's summary: {response.choices[0].message.content}")
    print()

    # Example 3: tool_choice options
    print("3. Forcing Specific Tool")
    print("-" * 40)
    messages = [{"role": "user", "content": "I want to stay somewhere nice"}]
    response = model.chat_complete(
        messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "book_hotel"}},
    )

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Forced to call: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
    print()

    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
