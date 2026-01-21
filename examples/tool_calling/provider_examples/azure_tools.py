#!/usr/bin/env python3
"""Azure OpenAI tool calling example.

Demonstrates tool calling with Azure OpenAI Service.
Azure uses OpenAI-compatible format.

Environment variables required:
- AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
- AZURE_OPENAI_ENDPOINT: Your Azure endpoint URL
- AZURE_OPENAI_DEPLOYMENT: Your deployment name (optional, defaults to model_name)

Run with:
    python examples/tool_calling/provider_examples/azure_tools.py
"""

import json
import os
from typing import Any, Dict

from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction


def create_tools():
    """Create tools for an enterprise assistant."""
    return [
        Tool(
            type="function",
            function=ToolFunction(
                name="query_database",
                description="Query the enterprise database for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "enum": ["customers", "orders", "products", "employees"],
                            "description": "The table to query",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Key-value pairs for filtering",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["table"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="send_notification",
                description="Send a notification to a user or team",
                parameters={
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "string",
                            "description": "Email address or team name",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Notification subject",
                        },
                        "message": {
                            "type": "string",
                            "description": "Notification message body",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high", "urgent"],
                            "default": "normal",
                        },
                    },
                    "required": ["recipient", "subject", "message"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="create_ticket",
                description="Create a support or task ticket",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Ticket title",
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["bug", "feature", "support", "task"],
                        },
                        "assignee": {
                            "type": "string",
                            "description": "Person to assign the ticket to",
                        },
                    },
                    "required": ["title", "description", "category"],
                },
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution."""
    if name == "query_database":
        table = args.get("table", "customers")
        return {
            "table": table,
            "filters": args.get("filters", {}),
            "results": [
                {"id": 1, "name": f"Sample {table} record 1"},
                {"id": 2, "name": f"Sample {table} record 2"},
            ],
            "total": 2,
        }
    elif name == "send_notification":
        return {
            "status": "sent",
            "recipient": args.get("recipient"),
            "subject": args.get("subject"),
            "message_id": "MSG-" + str(hash(args.get("subject", "")))[1:7],
        }
    elif name == "create_ticket":
        return {
            "ticket_id": "TKT-" + str(hash(args.get("title", "")))[1:7],
            "title": args.get("title"),
            "category": args.get("category"),
            "status": "created",
            "assignee": args.get("assignee", "unassigned"),
        }
    return {"error": f"Unknown tool: {name}"}


def main():
    # Check for required environment variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        print("Please set the following environment variables:")
        print("  - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key")
        print("  - AZURE_OPENAI_ENDPOINT: Your Azure endpoint URL")
        print("    (e.g., https://your-resource.openai.azure.com)")
        print("\nOptionally:")
        print("  - AZURE_OPENAI_DEPLOYMENT: Your deployment name")
        return

    print("=== Azure OpenAI Tool Calling Example ===\n")
    print(f"Endpoint: {endpoint}")

    tools = create_tools()

    # Create model
    # Note: model_name should be your deployment name in Azure
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    model = AIFactory.create_language(
        provider="azure",
        model_name=deployment,
        config={
            "base_url": endpoint,
        },
    )

    # Example 1: Database query
    print("\n1. Enterprise Database Query")
    print("-" * 40)
    messages = [{"role": "user", "content": "Show me all orders from the last week"}]

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
    print()

    # Example 2: Multi-step workflow
    print("2. Multi-step Workflow")
    print("-" * 40)

    messages = [
        {
            "role": "user",
            "content": "Find all high-value customers and create a support ticket to follow up with them",
        }
    ]

    try:
        response = model.chat_complete(messages, tools=tools)

        iteration = 0
        while response.choices[0].message.tool_calls and iteration < 5:
            iteration += 1

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
                print(f"  Step {iteration}: {tc.function.name} -> {json.dumps(result)[:60]}...")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

            response = model.chat_complete(messages, tools=tools)

        print(f"\nFinal response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Example 3: Notification tool
    print("3. Send Notification")
    print("-" * 40)
    messages = [
        {
            "role": "user",
            "content": "Send an urgent notification to the sales team about the quarterly meeting tomorrow",
        }
    ]

    try:
        response = model.chat_complete(messages, tools=tools)

        if response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            print(f"Tool: {tc.function.name}")
            args = json.loads(tc.function.arguments)
            print(f"Recipient: {args.get('recipient')}")
            print(f"Subject: {args.get('subject')}")
            print(f"Priority: {args.get('priority', 'normal')}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    print("=== Example Complete ===")
    print("\nAzure OpenAI benefits:")
    print("  - Enterprise compliance (SOC 2, HIPAA, etc.)")
    print("  - Data stays in your Azure region")
    print("  - Private endpoints available")
    print("  - Same OpenAI models with enterprise security")


if __name__ == "__main__":
    main()
