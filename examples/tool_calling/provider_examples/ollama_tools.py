#!/usr/bin/env python3
"""Ollama tool calling example.

Demonstrates tool calling with local Ollama models.
Note: Tool support varies by model - use models like llama3.1, mistral, etc.

Requirements:
- Ollama installed and running locally
- A model with tool support pulled (e.g., llama3.1)

Environment variables (optional):
- OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)

Run with:
    python examples/tool_calling/provider_examples/ollama_tools.py
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
                name="get_system_info",
                description="Get information about the local system",
                parameters={
                    "type": "object",
                    "properties": {
                        "info_type": {
                            "type": "string",
                            "enum": ["cpu", "memory", "disk", "network"],
                            "description": "Type of system information to retrieve",
                        }
                    },
                    "required": ["info_type"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="read_file",
                description="Read the contents of a local file",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "File encoding (default: utf-8)",
                            "default": "utf-8",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="list_directory",
                description="List contents of a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list",
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "description": "Include hidden files",
                            "default": False,
                        },
                    },
                    "required": ["path"],
                },
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution."""
    if name == "get_system_info":
        info_type = args.get("info_type", "cpu")
        mock_info = {
            "cpu": {"cores": 8, "usage": "25%", "model": "Intel i7"},
            "memory": {"total": "16GB", "used": "8GB", "available": "8GB"},
            "disk": {"total": "500GB", "used": "250GB", "free": "250GB"},
            "network": {"interface": "eth0", "ip": "192.168.1.100", "status": "connected"},
        }
        return mock_info.get(info_type, {"error": "Unknown info type"})

    elif name == "read_file":
        return {
            "file_path": args.get("file_path"),
            "content": "# Example File\n\nThis is mock file content.",
            "size": 45,
            "encoding": args.get("encoding", "utf-8"),
        }

    elif name == "list_directory":
        return {
            "path": args.get("path"),
            "files": ["file1.txt", "file2.py", "README.md"],
            "directories": ["src", "tests", "docs"],
            "total_items": 6,
        }

    return {"error": f"Unknown tool: {name}"}


def main():
    print("=== Ollama Tool Calling Example ===\n")
    print("Note: Requires Ollama running locally with a tool-capable model\n")

    # Get Ollama base URL from environment or use default
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"Using Ollama at: {base_url}\n")

    # Check if Ollama is available
    try:
        import httpx

        response = httpx.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception("Ollama not responding")
        models = response.json().get("models", [])
        print(f"Ollama is running with {len(models)} model(s) available")
    except Exception as e:
        print("Could not connect to Ollama. Please ensure:")
        print("  1. Ollama is installed (https://ollama.ai)")
        print("  2. Ollama is running (ollama serve)")
        print("  3. You have a tool-capable model (ollama pull llama3.1)")
        print(f"  4. OLLAMA_BASE_URL is set correctly (current: {base_url})")
        print(f"\nError: {e}")
        return

    tools = create_tools()

    # Create model - using llama3.1 which has good tool support
    model = AIFactory.create_language(
        provider="ollama",
        model_name="llama3.2:3b",  # or mistral, etc.
        config={"base_url": base_url},
    )

    # Example 1: Basic tool call
    print("\n1. Basic Tool Call")
    print("-" * 40)
    messages = [{"role": "user", "content": "What's the CPU usage on this system?"}]

    try:
        response = model.chat_complete(messages, tools=tools)

        if response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            print(f"Tool: {tc.function.name}")
            print(f"Args: {json.loads(tc.function.arguments)}")
        else:
            print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure the model supports tool calling")
    print()

    # Example 2: Multi-turn with tool results
    print("2. Multi-turn Conversation")
    print("-" * 40)

    messages = [{"role": "user", "content": "List the contents of the current directory"}]

    try:
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
                print(f"Executed {tc.function.name}: {result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

            final_response = model.chat_complete(messages, tools=tools)
            print(f"\nOllama response: {final_response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    print("=== Example Complete ===")
    print("\nOllama benefits:")
    print("  - Runs locally (no API costs, privacy)")
    print("  - OpenAI-compatible format")
    print("  - Tool support depends on the model")
    print("\nRecommended models for tools:")
    print("  - llama3.1 (best tool support)")
    print("  - mistral")
    print("  - mixtral")


if __name__ == "__main__":
    main()
