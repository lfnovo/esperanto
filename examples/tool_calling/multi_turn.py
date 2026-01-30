#!/usr/bin/env python3
"""Multi-turn tool calling example with Esperanto.

This example demonstrates a complete tool calling workflow:
1. Send a message that triggers tool use
2. Execute the tool and return results
3. Get the final response from the model

Environment variables required:
- OPENAI_API_KEY: Your OpenAI API key

Run with:
    python examples/tool_calling/multi_turn.py
"""

import json
import os
from typing import Any, Dict

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction


def create_tools():
    """Create tools for a restaurant assistant."""

    search_restaurants = Tool(
        type="function",
        function=ToolFunction(
            name="search_restaurants",
            description="Search for restaurants by cuisine type and location",
            parameters={
                "type": "object",
                "properties": {
                    "cuisine": {
                        "type": "string",
                        "description": "Type of cuisine, e.g., 'Italian', 'Japanese', 'Mexican'",
                    },
                    "location": {
                        "type": "string",
                        "description": "City or neighborhood",
                    },
                    "price_range": {
                        "type": "string",
                        "enum": ["$", "$$", "$$$", "$$$$"],
                        "description": "Price range",
                    },
                },
                "required": ["cuisine", "location"],
            },
        ),
    )

    get_restaurant_details = Tool(
        type="function",
        function=ToolFunction(
            name="get_restaurant_details",
            description="Get detailed information about a specific restaurant",
            parameters={
                "type": "object",
                "properties": {
                    "restaurant_id": {
                        "type": "string",
                        "description": "The restaurant ID",
                    }
                },
                "required": ["restaurant_id"],
            },
        ),
    )

    make_reservation = Tool(
        type="function",
        function=ToolFunction(
            name="make_reservation",
            description="Make a reservation at a restaurant",
            parameters={
                "type": "object",
                "properties": {
                    "restaurant_id": {
                        "type": "string",
                        "description": "The restaurant ID",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format",
                    },
                    "time": {
                        "type": "string",
                        "description": "Time in HH:MM format",
                    },
                    "party_size": {
                        "type": "integer",
                        "description": "Number of guests",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["restaurant_id", "date", "time", "party_size"],
            },
        ),
    )

    return [search_restaurants, get_restaurant_details, make_reservation]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution with mock data."""

    if name == "search_restaurants":
        # Simulate restaurant search results
        return {
            "results": [
                {
                    "id": "rest_001",
                    "name": "Bella Italia",
                    "cuisine": args.get("cuisine", "Italian"),
                    "rating": 4.5,
                    "price_range": args.get("price_range", "$$"),
                    "address": f"123 Main St, {args.get('location', 'Unknown')}",
                },
                {
                    "id": "rest_002",
                    "name": "Trattoria Roma",
                    "cuisine": args.get("cuisine", "Italian"),
                    "rating": 4.2,
                    "price_range": args.get("price_range", "$$"),
                    "address": f"456 Oak Ave, {args.get('location', 'Unknown')}",
                },
            ],
            "total": 2,
        }

    elif name == "get_restaurant_details":
        # Simulate restaurant details
        restaurant_id = args.get("restaurant_id", "unknown")
        if restaurant_id == "rest_001":
            return {
                "id": "rest_001",
                "name": "Bella Italia",
                "cuisine": "Italian",
                "rating": 4.5,
                "price_range": "$$",
                "address": "123 Main St, San Francisco",
                "phone": "(555) 123-4567",
                "hours": "11:00 AM - 10:00 PM",
                "description": "Authentic Italian cuisine in a cozy atmosphere",
                "specialties": ["Homemade pasta", "Wood-fired pizza", "Tiramisu"],
            }
        return {"error": f"Restaurant {restaurant_id} not found"}

    elif name == "make_reservation":
        # Simulate making a reservation
        return {
            "confirmation_number": "RES-2024-001234",
            "restaurant": "Bella Italia",
            "date": args.get("date"),
            "time": args.get("time"),
            "party_size": args.get("party_size"),
            "status": "confirmed",
        }

    return {"error": f"Unknown tool: {name}"}


def run_conversation(model, tools, initial_message: str):
    """Run a multi-turn conversation with tool calling."""

    messages = [{"role": "user", "content": initial_message}]
    print(f"\nUser: {initial_message}\n")

    iteration = 0
    max_iterations = 5  # Prevent infinite loops

    while iteration < max_iterations:
        iteration += 1

        # Call the model
        response = model.chat_complete(messages, tools=tools)
        message = response.choices[0].message

        # Check if there are tool calls
        if not message.tool_calls:
            # No tool calls - model is done
            print(f"Assistant: {message.content}\n")
            break

        print(f"[Iteration {iteration}] Model requested {len(message.tool_calls)} tool(s):")

        # Add assistant message with tool calls to history
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
                    for tc in message.tool_calls
                ],
            }
        )

        # Execute each tool and add results
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"  - Executing: {func_name}({func_args})")

            # Execute the tool
            result = execute_tool(func_name, func_args)
            print(f"    Result: {json.dumps(result, indent=2)[:200]}...")

            # Add tool result to messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        print()

    if iteration >= max_iterations:
        print("(Reached maximum iterations)")


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("Export it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Create tools and model
    tools = create_tools()
    model = AIFactory.create_language(
        provider="openai",
        model_name="gpt-4o-mini",
    )

    print("=== Multi-Turn Tool Calling Example ===")
    print(f"Available tools: {[t.function.name for t in tools]}")

    # Example 1: Simple search
    print("\n" + "=" * 50)
    print("Example 1: Restaurant Search")
    print("=" * 50)
    run_conversation(
        model,
        tools,
        "Find me some Italian restaurants in San Francisco.",
    )

    # Example 2: Multi-step interaction (search, get details)
    print("\n" + "=" * 50)
    print("Example 2: Search and Get Details")
    print("=" * 50)
    run_conversation(
        model,
        tools,
        "I'm looking for Italian food in San Francisco. Can you find some options and tell me more about the best one?",
    )

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
