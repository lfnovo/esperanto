#!/usr/bin/env python3
"""Groq tool calling example.

Demonstrates tool calling with Groq's fast inference API.
Groq uses OpenAI-compatible format, so tools work identically.

Environment variables required:
- GROQ_API_KEY: Your Groq API key

Run with:
    python examples/tool_calling/provider_examples/groq_tools.py
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
                name="search_products",
                description="Search for products in the catalog",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for products",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["electronics", "clothing", "home", "sports"],
                            "description": "Product category to filter by",
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Maximum price filter",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="get_product_details",
                description="Get detailed information about a specific product",
                parameters={
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "The unique product identifier",
                        }
                    },
                    "required": ["product_id"],
                },
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution."""
    if name == "search_products":
        return {
            "query": args.get("query"),
            "category": args.get("category"),
            "results": [
                {"id": "prod_001", "name": "Wireless Headphones", "price": 79.99},
                {"id": "prod_002", "name": "Bluetooth Speaker", "price": 49.99},
            ],
            "total": 2,
        }
    elif name == "get_product_details":
        return {
            "id": args.get("product_id"),
            "name": "Wireless Headphones",
            "price": 79.99,
            "description": "High-quality wireless headphones with noise cancellation",
            "in_stock": True,
            "rating": 4.5,
        }
    return {"error": f"Unknown tool: {name}"}


def main():
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("Please set GROQ_API_KEY environment variable")
        return

    print("=== Groq Tool Calling Example ===\n")
    print("Note: Groq provides extremely fast inference with tool support!\n")

    tools = create_tools()

    # Create model - Llama 3.3 70B with tool support
    model = AIFactory.create_language(
        provider="groq",
        model_name="llama-3.3-70b-versatile",
    )

    # Example 1: Basic tool call
    print("1. Basic Tool Call (Fast!)")
    print("-" * 40)
    messages = [{"role": "user", "content": "Find me some wireless headphones"}]

    import time
    start = time.time()
    response = model.chat_complete(messages, tools=tools)
    elapsed = time.time() - start

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Tool: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
        print(f"Response time: {elapsed:.2f}s (Groq is fast!)")
    print()

    # Example 2: Multi-turn conversation
    print("2. Multi-turn with Tool Results")
    print("-" * 40)

    messages = [
        {
            "role": "user",
            "content": "Search for electronics under $100 and tell me about the first result",
        }
    ]

    response = model.chat_complete(messages, tools=tools)

    # Handle tool calls
    while response.choices[0].message.tool_calls:
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
                    for tc in response.choices[0].message.tool_calls
                ],
            }
        )

        # Execute tools
        for tc in response.choices[0].message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            print(f"  -> {tc.function.name}({args}) = {json.dumps(result)[:80]}...")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        # Continue conversation
        response = model.chat_complete(messages, tools=tools)

    print(f"\nFinal response: {response.choices[0].message.content}")
    print()

    # Example 3: Parallel tool calls
    print("3. Parallel Tool Calls")
    print("-" * 40)
    messages = [
        {
            "role": "user",
            "content": "Search for both headphones and speakers, and show me details for product prod_001",
        }
    ]

    response = model.chat_complete(messages, tools=tools, parallel_tool_calls=True)

    if response.choices[0].message.tool_calls:
        print(f"Groq returned {len(response.choices[0].message.tool_calls)} tool call(s):")
        for tc in response.choices[0].message.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  - {tc.function.name}: {args}")
    print()

    print("=== Example Complete ===")
    print("\nGroq benefits:")
    print("  - Extremely fast inference (often <1s for tool calls)")
    print("  - OpenAI-compatible format (same code works)")
    print("  - Supports parallel tool calls")


if __name__ == "__main__":
    main()
