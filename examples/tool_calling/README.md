# Tool Calling Examples

This directory contains examples demonstrating Esperanto's unified tool/function calling support across different providers.

## Quick Start

```python
from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction

# Define a tool once
tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        )
    )
]

# Use with any provider
model = AIFactory.create_language("openai", "gpt-4o")
response = model.chat_complete(messages, tools=tools)
```

## Examples

### Core Examples

| File | Description |
|------|-------------|
| [basic_tool.py](basic_tool.py) | Simple single-tool example |
| [multiple_tools.py](multiple_tools.py) | Working with multiple tools |
| [multi_turn.py](multi_turn.py) | Multi-turn conversations with tool results |
| [streaming_tools.py](streaming_tools.py) | Streaming responses with tool calls |
| [provider_comparison.py](provider_comparison.py) | Same code across all providers |

### Provider-Specific Examples

Located in `provider_examples/`:

| File | Provider | Description |
|------|----------|-------------|
| [openai_tools.py](provider_examples/openai_tools.py) | OpenAI | Includes strict mode, parallel calls |
| [anthropic_tools.py](provider_examples/anthropic_tools.py) | Anthropic | Claude tool calling |
| [google_tools.py](provider_examples/google_tools.py) | Google | Gemini function calling |
| [groq_tools.py](provider_examples/groq_tools.py) | Groq | Fast inference with tools |
| [mistral_tools.py](provider_examples/mistral_tools.py) | Mistral | Mistral AI tools |
| [vertex_tools.py](provider_examples/vertex_tools.py) | Vertex AI | Google Cloud Vertex |
| [deepseek_tools.py](provider_examples/deepseek_tools.py) | DeepSeek | Code-focused tool calling |
| [azure_tools.py](provider_examples/azure_tools.py) | Azure OpenAI | Enterprise tool calling |
| [ollama_tools.py](provider_examples/ollama_tools.py) | Ollama | Local models with tools |

## Running Examples

Set the appropriate API key for your provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
python basic_tool.py

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python provider_examples/anthropic_tools.py

# Google
export GOOGLE_API_KEY="..."
python provider_examples/google_tools.py

# Groq
export GROQ_API_KEY="gsk_..."
python provider_examples/groq_tools.py

# Mistral
export MISTRAL_API_KEY="..."
python provider_examples/mistral_tools.py

# Vertex AI (requires gcloud auth)
export GOOGLE_CLOUD_PROJECT="your-project-id"
python provider_examples/vertex_tools.py

# DeepSeek
export DEEPSEEK_API_KEY="..."
python provider_examples/deepseek_tools.py

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
python provider_examples/azure_tools.py

# Ollama (local, no API key needed)
# Requires: ollama serve && ollama pull llama3.1
python provider_examples/ollama_tools.py
```

## Key Concepts

### Tool Definition

```python
from esperanto.common_types import Tool, ToolFunction

tool = Tool(
    type="function",
    function=ToolFunction(
        name="function_name",
        description="What the function does",
        parameters={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
            },
            "required": ["param1"]
        }
    )
)
```

### Tool Calls in Response

```python
response = model.chat_complete(messages, tools=tools)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Tool: {tc.function.name}")
        print(f"Args: {tc.function.arguments}")  # JSON string
```

### Sending Tool Results

```python
# After getting tool calls, execute them and send results:
messages.append({
    "role": "assistant",
    "tool_calls": [...]  # From response
})

messages.append({
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": '{"result": "value"}'  # JSON string
})

# Continue conversation
response = model.chat_complete(messages, tools=tools)
```

## Documentation

For complete documentation, see [docs/features/tool-calling.md](../../docs/features/tool-calling.md).
