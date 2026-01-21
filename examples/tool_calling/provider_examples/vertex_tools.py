#!/usr/bin/env python3
"""Vertex AI tool calling example.

Demonstrates tool calling with Google Cloud Vertex AI (Gemini models).
Vertex AI uses Google's function calling format.

Requirements:
- Google Cloud project with Vertex AI enabled
- Authenticated via gcloud CLI or service account

Environment variables:
- GOOGLE_CLOUD_PROJECT: Your Google Cloud project ID
- GOOGLE_CLOUD_LOCATION: The region (default: us-central1)

Run with:
    python examples/tool_calling/provider_examples/vertex_tools.py
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
                name="analyze_sentiment",
                description="Analyze the sentiment of a piece of text",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze",
                        },
                        "language": {
                            "type": "string",
                            "description": "The language of the text (optional)",
                        },
                    },
                    "required": ["text"],
                },
            ),
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="translate_text",
                description="Translate text from one language to another",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to translate",
                        },
                        "source_language": {
                            "type": "string",
                            "description": "Source language code (e.g., 'en', 'es')",
                        },
                        "target_language": {
                            "type": "string",
                            "description": "Target language code",
                        },
                    },
                    "required": ["text", "target_language"],
                },
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution."""
    if name == "analyze_sentiment":
        text = args.get("text", "")
        # Simple mock sentiment analysis
        positive_words = ["good", "great", "happy", "love", "excellent"]
        negative_words = ["bad", "terrible", "sad", "hate", "awful"]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            sentiment = "positive"
            score = 0.7
        elif neg_count > pos_count:
            sentiment = "negative"
            score = -0.7
        else:
            sentiment = "neutral"
            score = 0.0

        return {
            "text": text[:50] + "..." if len(text) > 50 else text,
            "sentiment": sentiment,
            "confidence": abs(score),
            "language": args.get("language", "en"),
        }
    elif name == "translate_text":
        # Mock translation
        return {
            "original": args.get("text"),
            "translated": f"[Translated to {args.get('target_language')}]: {args.get('text')}",
            "source_language": args.get("source_language", "auto-detected"),
            "target_language": args.get("target_language"),
        }
    return {"error": f"Unknown tool: {name}"}


def main():
    # Check for required environment
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        print("Please set GOOGLE_CLOUD_PROJECT environment variable")
        print("\nVertex AI requires:")
        print("  1. A Google Cloud project with Vertex AI API enabled")
        print("  2. Authentication via 'gcloud auth application-default login'")
        print("     or a service account key")
        print("  3. GOOGLE_CLOUD_PROJECT environment variable")
        return

    print("=== Vertex AI Tool Calling Example ===\n")
    print(f"Project: {project}")
    print(f"Location: {os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')}\n")

    tools = create_tools()

    # Create model
    model = AIFactory.create_language(
        provider="vertex",
        model_name="gemini-2.0-flash",  # Vertex AI Gemini model
    )

    # Example 1: Basic tool call
    print("1. Basic Tool Call")
    print("-" * 40)
    messages = [
        {
            "role": "user",
            "content": "Analyze the sentiment of: 'I absolutely love this new product, it's excellent!'",
        }
    ]
    response = model.chat_complete(messages, tools=tools)

    if response.choices[0].message.tool_calls:
        tc = response.choices[0].message.tool_calls[0]
        print(f"Tool: {tc.function.name}")
        print(f"Args: {json.loads(tc.function.arguments)}")
    else:
        print(f"Response: {response.choices[0].message.content}")
    print()

    # Example 2: Multi-turn with tool results
    print("2. Multi-turn Conversation")
    print("-" * 40)

    messages = [
        {
            "role": "user",
            "content": "Translate 'Hello, how are you?' to Spanish",
        }
    ]

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

        # Execute tools
        for tc in response.choices[0].message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            print(f"Executed: {tc.function.name}")
            print(f"Result: {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        # Get final response
        final_response = model.chat_complete(messages, tools=tools)
        print(f"\nVertex AI response: {final_response.choices[0].message.content}")
    print()

    # Example 3: Chained tools
    print("3. Chained Tool Calls")
    print("-" * 40)
    messages = [
        {
            "role": "user",
            "content": "Translate 'The weather is terrible today' to French, then analyze its sentiment",
        }
    ]

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
            print(f"  Step {iteration}: {tc.function.name} -> {json.dumps(result)[:50]}...")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        response = model.chat_complete(messages, tools=tools)

    print(f"\nFinal response: {response.choices[0].message.content}")
    print()

    print("=== Example Complete ===")
    print("\nVertex AI notes:")
    print("  - Uses Google Cloud authentication (not API key)")
    print("  - Esperanto converts to Vertex AI SDK format internally")
    print("  - Tool call IDs are generated by Esperanto")


if __name__ == "__main__":
    main()
