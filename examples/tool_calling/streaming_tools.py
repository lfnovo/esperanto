#!/usr/bin/env python3
"""Streaming tool calling example with Esperanto.

This example demonstrates streaming responses with tool calls:
1. Stream responses from the model
2. Accumulate tool call chunks
3. Process complete tool calls

Environment variables required:
- OPENAI_API_KEY: Your OpenAI API key

Run with:
    python examples/tool_calling/streaming_tools.py
"""

import json
import os
from typing import Any, Dict, List, Optional

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction


def create_tools():
    """Create tools for demonstration."""

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
                        "description": "City name",
                    }
                },
                "required": ["location"],
            },
        ),
    )

    stock_tool = Tool(
        type="function",
        function=ToolFunction(
            name="get_stock_price",
            description="Get current stock price for a ticker symbol",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g., 'AAPL', 'GOOGL'",
                    }
                },
                "required": ["symbol"],
            },
        ),
    )

    return [weather_tool, stock_tool]


class ToolCallAccumulator:
    """Helper class to accumulate streaming tool call chunks."""

    def __init__(self):
        self.tool_calls: Dict[int, Dict[str, Any]] = {}

    def add_chunk(self, tool_call_chunks: List[Any]) -> None:
        """Add tool call chunks from a streaming delta."""
        if not tool_call_chunks:
            return

        for tc in tool_call_chunks:
            # tc could be a dict or ToolCall object
            if hasattr(tc, "index"):
                index = tc.index if tc.index is not None else 0
                tc_id = tc.id if hasattr(tc, "id") else None
                tc_type = tc.type if hasattr(tc, "type") else "function"
                func = tc.function if hasattr(tc, "function") else None
            else:
                index = tc.get("index", 0)
                tc_id = tc.get("id")
                tc_type = tc.get("type", "function")
                func = tc.get("function", {})

            if index not in self.tool_calls:
                self.tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }

            if tc_id:
                self.tool_calls[index]["id"] = tc_id
            if tc_type:
                self.tool_calls[index]["type"] = tc_type

            if func:
                if hasattr(func, "name") and func.name:
                    self.tool_calls[index]["function"]["name"] += func.name
                elif isinstance(func, dict) and func.get("name"):
                    self.tool_calls[index]["function"]["name"] += func["name"]

                if hasattr(func, "arguments") and func.arguments:
                    self.tool_calls[index]["function"]["arguments"] += func.arguments
                elif isinstance(func, dict) and func.get("arguments"):
                    self.tool_calls[index]["function"]["arguments"] += func["arguments"]

    def get_complete_tool_calls(self) -> List[Dict[str, Any]]:
        """Get list of accumulated tool calls."""
        return [self.tool_calls[i] for i in sorted(self.tool_calls.keys())]


def stream_with_tools(model, messages: List[Dict], tools: List[Tool]):
    """Stream a response and handle tool calls."""

    print("Streaming response...")
    print("-" * 40)

    accumulator = ToolCallAccumulator()
    full_content = ""
    finish_reason: Optional[str] = None

    # Stream the response
    for chunk in model.chat_complete(messages, tools=tools, stream=True):
        choice = chunk.choices[0]
        delta = choice.delta

        # Accumulate text content
        if delta.content:
            print(delta.content, end="", flush=True)
            full_content += delta.content

        # Accumulate tool call chunks
        if delta.tool_calls:
            accumulator.add_chunk(delta.tool_calls)

        # Capture finish reason
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    print()  # Newline after streaming
    print("-" * 40)

    # Get complete tool calls
    tool_calls = accumulator.get_complete_tool_calls()

    return {
        "content": full_content,
        "tool_calls": tool_calls if tool_calls else None,
        "finish_reason": finish_reason,
    }


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
        config={"streaming": True},
    )

    print("=== Streaming Tool Calling Example ===\n")
    print(f"Available tools: {[t.function.name for t in tools]}\n")

    # Example 1: Regular text response (no tool call)
    print("Example 1: Regular response (no tool call)")
    print("=" * 50)
    messages = [{"role": "user", "content": "Hello! How are you today?"}]
    print(f"User: {messages[0]['content']}\n")

    result = stream_with_tools(model, messages, tools)
    print(f"\nFinish reason: {result['finish_reason']}")
    print(f"Tool calls: {result['tool_calls']}")

    # Example 2: Tool call response
    print("\n\nExample 2: Tool call response")
    print("=" * 50)
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    print(f"User: {messages[0]['content']}\n")

    result = stream_with_tools(model, messages, tools)
    print(f"\nFinish reason: {result['finish_reason']}")

    if result["tool_calls"]:
        print("\nAccumulated tool calls:")
        for tc in result["tool_calls"]:
            print(f"  Tool: {tc['function']['name']}")
            print(f"  ID: {tc['id']}")
            args = json.loads(tc["function"]["arguments"])
            print(f"  Arguments: {json.dumps(args, indent=4)}")

    # Example 3: Multiple tool calls
    print("\n\nExample 3: Multiple tool calls (parallel)")
    print("=" * 50)
    messages = [
        {
            "role": "user",
            "content": "What's the weather in London and what's the stock price for AAPL?",
        }
    ]
    print(f"User: {messages[0]['content']}\n")

    result = stream_with_tools(model, messages, tools)
    print(f"\nFinish reason: {result['finish_reason']}")

    if result["tool_calls"]:
        print(f"\nAccumulated {len(result['tool_calls'])} tool call(s):")
        for i, tc in enumerate(result["tool_calls"], 1):
            print(f"\n  {i}. Tool: {tc['function']['name']}")
            print(f"     ID: {tc['id']}")
            try:
                args = json.loads(tc["function"]["arguments"])
                print(f"     Arguments: {args}")
            except json.JSONDecodeError:
                print(f"     Arguments (raw): {tc['function']['arguments']}")

    print("\n\n=== Example Complete ===")


if __name__ == "__main__":
    main()
