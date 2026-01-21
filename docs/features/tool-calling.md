# Tool/Function Calling

## Overview

Tool calling (also known as function calling) allows language models to request execution of predefined functions during a conversation. This enables models to interact with external systems, retrieve real-time data, or perform actions on behalf of users.

Esperanto provides a **unified tool calling interface** that works consistently across all supported providers. Define your tools once, and use them with OpenAI, Anthropic, Google, Groq, Mistral, and more.

## Quick Start

```python
import json
from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction

# Define a tool
weather_tool = Tool(
    type="function",
    function=ToolFunction(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    )
)

# Create a model with tools
model = AIFactory.create_language(
    provider="openai",
    model_name="gpt-4o",
    config={"tools": [weather_tool]}
)

# Send a message that might trigger tool use
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
response = model.chat_complete(messages)

# Check if the model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Tool: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

## Tool Definition

### Tool Types

Esperanto provides typed classes for tool definitions:

```python
from esperanto.common_types import Tool, ToolFunction

# ToolFunction defines the function signature
function = ToolFunction(
    name="search_database",           # Function name (required)
    description="Search the database", # What the function does (required)
    parameters={                       # JSON Schema for parameters
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    },
    strict=True  # OpenAI-only: Enable strict mode for guaranteed schema adherence
)

# Tool wraps the function
tool = Tool(
    type="function",  # Currently only "function" is supported
    function=function
)
```

### Parameters Schema

The `parameters` field uses [JSON Schema](https://json-schema.org/) to define the function's input:

```python
parameters={
    "type": "object",
    "properties": {
        # String parameter
        "name": {
            "type": "string",
            "description": "The user's name"
        },
        # Number with constraints
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150
        },
        # Enum (fixed choices)
        "status": {
            "type": "string",
            "enum": ["active", "inactive", "pending"]
        },
        # Array
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        # Nested object
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"}
            }
        }
    },
    "required": ["name", "status"]  # Required fields
}
```

## Using Tools

### Passing Tools

Tools can be configured at two levels:

```python
# 1. Instance level - tools available for all calls
model = AIFactory.create_language(
    provider="openai",
    model_name="gpt-4o",
    config={"tools": [weather_tool, search_tool]}
)
response = model.chat_complete(messages)

# 2. Call level - override for specific request
response = model.chat_complete(
    messages,
    tools=[different_tool]  # Overrides instance tools
)
```

### Tool Choice

Control how the model uses tools:

```python
# "auto" (default) - Model decides whether to use tools
response = model.chat_complete(messages, tool_choice="auto")

# "required" - Model must call at least one tool
response = model.chat_complete(messages, tool_choice="required")

# "none" - Model cannot use tools (even if provided)
response = model.chat_complete(messages, tool_choice="none")

# Specific tool - Force calling a particular tool
response = model.chat_complete(
    messages,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)
```

### Parallel Tool Calls

Some providers allow models to request multiple tool calls in a single response:

```python
# Allow parallel tool calls (default for most providers)
response = model.chat_complete(messages, parallel_tool_calls=True)

# Force single tool call per response
response = model.chat_complete(messages, parallel_tool_calls=False)
```

## Processing Tool Calls

### Reading Tool Call Responses

When a model decides to call a tool, the response contains `tool_calls`:

```python
response = model.chat_complete(messages, tools=tools)

message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        # Unique ID for this tool call
        print(f"ID: {tool_call.id}")

        # Function name
        print(f"Function: {tool_call.function.name}")

        # Arguments as JSON string
        print(f"Arguments (raw): {tool_call.function.arguments}")

        # Parse arguments
        args = json.loads(tool_call.function.arguments)
        print(f"Arguments (parsed): {args}")
else:
    # Model responded with text instead
    print(f"Response: {message.content}")
```

### ToolCall Structure

```python
from esperanto.common_types import ToolCall, FunctionCall

# ToolCall has these fields:
tool_call = ToolCall(
    id="call_abc123",           # Unique identifier
    type="function",            # Always "function" currently
    function=FunctionCall(
        name="get_weather",     # Function to call
        arguments='{"location": "Tokyo"}'  # JSON string of arguments
    ),
    index=0                     # Optional: index for streaming
)
```

## Multi-Turn Conversations

To continue a conversation after tool calls, send the tool results back to the model:

```python
import json

# Initial request
messages = [{"role": "user", "content": "What's the weather in Tokyo and London?"}]
response = model.chat_complete(messages, tools=tools)

# Model requests tool calls
assistant_message = response.choices[0].message
if assistant_message.tool_calls:
    # Add assistant's message (with tool_calls) to history
    messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in assistant_message.tool_calls
        ]
    })

    # Execute each tool and add results
    for tool_call in assistant_message.tool_calls:
        # Execute your tool (implement this based on your needs)
        result = execute_tool(
            tool_call.function.name,
            json.loads(tool_call.function.arguments)
        )

        # Add tool result to messages
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)  # Must be a string
        })

    # Get final response with tool results
    final_response = model.chat_complete(messages, tools=tools)
    print(final_response.content)
```

### Complete Multi-Turn Example

```python
import json
from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction

# Define tools
tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        )
    )
]

# Simulated tool execution
def execute_tool(name: str, args: dict) -> dict:
    if name == "get_weather":
        # In reality, call a weather API
        return {"temperature": 22, "condition": "sunny", "city": args["city"]}
    return {"error": f"Unknown tool: {name}"}

# Create model
model = AIFactory.create_language("openai", "gpt-4o")

# Conversation loop
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

while True:
    response = model.chat_complete(messages, tools=tools)
    message = response.choices[0].message

    if not message.tool_calls:
        # No more tool calls - print final response
        print(f"Assistant: {message.content}")
        break

    # Add assistant message with tool calls
    messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in message.tool_calls
        ]
    })

    # Execute tools and add results
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = execute_tool(tool_call.function.name, args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
```

## Validation

Esperanto can validate tool call arguments against the JSON schema:

```python
from esperanto.common_types import ToolCallValidationError

try:
    response = model.chat_complete(
        messages,
        tools=tools,
        validate_tool_calls=True  # Enable validation
    )
except ToolCallValidationError as e:
    print(f"Tool '{e.tool_name}' validation failed:")
    for error in e.errors:
        print(f"  - {error}")
```

Validation requires the `jsonschema` package:

```bash
pip install esperanto[validation]
# or
pip install jsonschema
```

### Manual Validation

You can also validate tool calls manually:

```python
from esperanto.common_types import validate_tool_call, validate_tool_calls, find_tool_by_name

# Validate a single tool call
tool = find_tool_by_name(tools, tool_call.function.name)
if tool:
    validate_tool_call(tool_call, tool)

# Validate all tool calls in a response
if message.tool_calls:
    validate_tool_calls(message.tool_calls, tools)
```

## Streaming with Tools

Tool calls can also be received via streaming:

```python
# Streaming with tools
for chunk in model.chat_complete(messages, tools=tools, stream=True):
    delta = chunk.choices[0].delta

    if delta.content:
        print(delta.content, end="", flush=True)

    # Tool calls come in chunks during streaming
    if delta.tool_calls:
        for tc in delta.tool_calls:
            print(f"Tool call chunk: {tc}")
```

**Note**: When streaming, tool calls arrive incrementally. You'll need to accumulate the chunks to get complete tool call data. The `index` field on tool calls helps track which chunks belong to which tool call.

## Provider Support

| Provider | Tools | tool_choice | Streaming | Parallel Calls |
|----------|-------|-------------|-----------|----------------|
| OpenAI | Yes | Yes | Yes | Yes |
| Anthropic | Yes | Yes | Yes | Yes |
| Google | Yes | Yes | Yes | Varies |
| Groq | Yes | Yes | Yes | Yes |
| Mistral | Yes | Yes | Yes | Yes |
| Azure | Yes | Yes | Yes | Yes |
| Vertex AI | Yes | Yes | Yes | Varies |
| DeepSeek | Yes | Yes | Yes | Varies |
| xAI | Yes | Yes | Yes | Varies |
| OpenRouter | Yes* | Yes* | Yes* | Yes* |
| Ollama | Yes** | Yes** | Yes** | Varies |

\* Depends on underlying model
\** Model-dependent; not all Ollama models support tools

### Provider-Specific Notes

#### OpenAI
- Full support for all tool calling features
- Supports `strict` mode for guaranteed schema adherence
- Reasoning models (o1, o3) have limited tool support

#### Anthropic
- Tools are converted to Anthropic's `input_schema` format automatically
- `tool_choice="required"` maps to `{"type": "any"}`
- `parallel_tool_calls=False` maps to `disable_parallel_tool_use`

#### Google (Gemini)
- Tool IDs are generated by Esperanto (Google doesn't provide them)
- `tool_choice` maps to `function_calling_config.mode`

#### Vertex AI
- Similar to Google but uses the Vertex AI SDK
- Requires Google Cloud authentication

## Examples

See the [examples/tool_calling/](../../examples/tool_calling/) directory for complete examples:

- [basic_tool.py](../../examples/tool_calling/basic_tool.py) - Simple single-tool example
- [multiple_tools.py](../../examples/tool_calling/multiple_tools.py) - Working with multiple tools
- [multi_turn.py](../../examples/tool_calling/multi_turn.py) - Multi-turn conversations with tool results
- [streaming_tools.py](../../examples/tool_calling/streaming_tools.py) - Streaming with tool calls
- [provider_comparison.py](../../examples/tool_calling/provider_comparison.py) - Same code across providers

## Best Practices

### 1. Write Clear Descriptions

Models use descriptions to decide when and how to call tools:

```python
# Good - Clear and specific
ToolFunction(
    name="search_products",
    description="Search the product catalog by name, category, or price range. "
                "Returns up to 10 matching products with name, price, and availability.",
    parameters={...}
)

# Bad - Vague
ToolFunction(
    name="search",
    description="Searches stuff",
    parameters={...}
)
```

### 2. Use Specific Parameter Types

```python
# Good - Specific types with constraints
parameters={
    "type": "object",
    "properties": {
        "price_min": {"type": "number", "minimum": 0},
        "price_max": {"type": "number", "minimum": 0},
        "category": {"type": "string", "enum": ["electronics", "clothing", "food"]}
    }
}

# Bad - Too permissive
parameters={
    "type": "object",
    "properties": {
        "filters": {"type": "object"}  # Too vague
    }
}
```

### 3. Handle Errors Gracefully

```python
def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "get_weather":
            return json.dumps(fetch_weather(args["city"]))
    except Exception as e:
        # Return error as tool result - model can handle it
        return json.dumps({"error": str(e)})
```

### 4. Limit Tool Count

Providing too many tools can confuse the model. Group related functionality:

```python
# Instead of many narrow tools:
# get_user_name, get_user_email, get_user_address, ...

# Use one comprehensive tool:
Tool(
    type="function",
    function=ToolFunction(
        name="get_user_info",
        description="Get user information",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "fields": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["name", "email", "address", "phone"]}
                }
            }
        }
    )
)
```

## See Also

- [Language Models (LLM)](../capabilities/llm.md) - Core LLM documentation
- [Provider Guides](../providers/README.md) - Provider-specific setup and features
- [Examples](../../examples/tool_calling/) - Complete working examples
