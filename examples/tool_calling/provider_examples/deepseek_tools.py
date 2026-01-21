#!/usr/bin/env python3
"""DeepSeek tool calling example.

Demonstrates tool calling with DeepSeek models.
DeepSeek uses OpenAI-compatible format.

Environment variables required:
- DEEPSEEK_API_KEY: Your DeepSeek API key

Run with:
    python examples/tool_calling/provider_examples/deepseek_tools.py
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
                name="run_code",
                description="Execute Python code and return the result",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Maximum execution time in seconds",
                            "default": 30,
                        },
                    },
                    "required": ["code"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="search_documentation",
                description="Search Python documentation for a topic",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "module": {
                            "type": "string",
                            "description": "Specific module to search in (optional)",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution."""
    if name == "run_code":
        code = args.get("code", "")
        # Mock code execution (don't actually execute arbitrary code!)
        return {
            "code": code[:100] + "..." if len(code) > 100 else code,
            "output": "42",
            "execution_time": 0.05,
            "status": "success",
        }
    elif name == "search_documentation":
        return {
            "query": args.get("query"),
            "module": args.get("module"),
            "results": [
                {
                    "title": f"Python {args.get('query')} documentation",
                    "url": f"https://docs.python.org/3/library/{args.get('query', 'index')}.html",
                    "snippet": f"The {args.get('query')} module provides...",
                }
            ],
        }
    return {"error": f"Unknown tool: {name}"}


def main():
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Please set DEEPSEEK_API_KEY environment variable")
        return

    print("=== DeepSeek Tool Calling Example ===\n")
    print("Note: DeepSeek models are particularly strong at code-related tasks!\n")

    tools = create_tools()

    # Create model
    model = AIFactory.create_language(
        provider="deepseek",
        model_name="deepseek-chat",
    )

    # Example 1: Code-related tool call
    print("1. Code Execution Tool")
    print("-" * 40)
    messages = [
        {
            "role": "user",
            "content": "Can you calculate the factorial of 10 using Python?",
        }
    ]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Tool: {tc.function.name}")
        args = json.loads(tc.function.arguments)
        print(f"Code to execute:\n{args.get('code', 'N/A')}")
    print()

    # Example 2: Multi-turn with tool execution
    print("2. Multi-turn Code Assistant")
    print("-" * 40)

    messages = [
        {
            "role": "user",
            "content": "Write and run a Python function to check if a number is prime",
        }
    ]

    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
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

        for tc in response.choices[0].message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            print(f"Executed {tc.function.name}")
            print(f"Result: {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        final_response = model.chat_complete(messages, tools=tools)
        print(f"\nDeepSeek's response: {final_response.choices[0].message.content}")
    print()

    # Example 3: Documentation search
    print("3. Documentation Search")
    print("-" * 40)
    messages = [{"role": "user", "content": "Find documentation for the asyncio module"}]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Tool: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
    print()

    print("=== Example Complete ===")
    print("\nDeepSeek strengths:")
    print("  - Excellent at code generation and analysis")
    print("  - Cost-effective for development tasks")
    print("  - OpenAI-compatible API format")


if __name__ == "__main__":
    main()
