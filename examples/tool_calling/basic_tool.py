#!/usr/bin/env python3
"""Basic tool calling example with Esperanto.

This example demonstrates the simplest form of tool calling:
1. Define a single tool
2. Send a message that triggers tool use
3. Process the tool call response

Environment variables required:
- OPENAI_API_KEY: Your OpenAI API key

Run with:
    python examples/tool_calling/basic_tool.py
"""

import json
import os

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("Export it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Define a simple weather tool
    weather_tool = Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get the current weather for a location. "
            "Returns temperature, conditions, and humidity.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country, e.g., 'San Francisco, CA' or 'London, UK'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit preference",
                    },
                },
                "required": ["location"],
            },
        ),
    )

    # Create a model with the tool
    model = AIFactory.create_language(
        provider="openai",
        model_name="gpt-4o-mini",
        config={"tools": [weather_tool]},
    )

    print("=== Basic Tool Calling Example ===\n")

    # Send a message that should trigger tool use
    messages = [{"role": "user", "content": "What's the weather like in Tokyo, Japan?"}]

    print(f"User: {messages[0]['content']}\n")

    # Make the API call
    response = model.chat_complete(messages)
    message = response.choices[0].message

    # Check if the model wants to call a tool
    if message.tool_calls:
        print("Model requested tool call(s):")
        for tool_call in message.tool_calls:
            print(f"  Tool: {tool_call.function.name}")
            print(f"  ID: {tool_call.id}")

            # Parse and display arguments
            args = json.loads(tool_call.function.arguments)
            print(f"  Arguments: {json.dumps(args, indent=4)}")
            print()

            # In a real application, you would execute the tool here
            # For this example, we'll just show what was requested
            print("  (In a real app, you would now call your weather API)")
            print(f"  (with location='{args.get('location')}')")
    else:
        # Model responded with text instead
        print(f"Assistant: {message.content}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
