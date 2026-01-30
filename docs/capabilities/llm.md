# Language Models (LLM)

## Overview

Language Models are the core text generation capability in Esperanto. They process text prompts and generate human-like responses, supporting various tasks from simple Q&A to complex reasoning and analysis.

## Common Use Cases

- **Conversational AI**: Chatbots, virtual assistants, customer support
- **Content Generation**: Articles, summaries, creative writing, code generation
- **Analysis & Reasoning**: Document analysis, decision support, problem-solving
- **Text Transformation**: Translation, rewriting, formatting, extraction

## Interface

### Creating a Language Model

```python
from esperanto.factory import AIFactory

# Using the factory (recommended)
model = AIFactory.create_language(
    provider="openai",           # Provider name
    model_name="gpt-4",          # Model identifier
    config={                     # Optional configuration
        "temperature": 0.7,      # Creativity (0.0-2.0)
        "max_tokens": 1000,      # Response length limit
        "top_p": 0.9,           # Nucleus sampling
        "streaming": False,      # Enable streaming responses
        "structured": {"type": "json"},  # JSON output mode
        "timeout": 60.0          # Request timeout in seconds
    }
)
```

### Core Methods

#### `chat_complete(messages, **kwargs)`

Synchronous text generation from message history.

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

response = model.chat_complete(messages)
print(response.content)  # Shortcut for response.choices[0].message.content
```

#### `achat_complete(messages, **kwargs)`

Asynchronous text generation (identical interface to `chat_complete`).

```python
response = await model.achat_complete(messages)
print(response.content)
```

### Streaming Responses

Enable streaming to receive responses token by token:

```python
# Enable via config
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"streaming": True}
)

# Or per-request
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

## Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Controls randomness (0.0 = deterministic, 2.0 = very random) |
| `max_tokens` | int | Model default | Maximum tokens in response |
| `top_p` | float | 1.0 | Nucleus sampling threshold (alternative to temperature) |
| `streaming` | bool | False | Enable token-by-token streaming |
| `structured` | dict | None | Enable structured output (e.g., `{"type": "json"}`) |
| `timeout` | float | 60.0 | Request timeout in seconds |

### Message Format

Messages follow the OpenAI chat format:

```python
{
    "role": "system" | "user" | "assistant",
    "content": str  # The message text
}
```

### Parameter Priority

For parameters like `temperature` and `max_tokens`:

1. **Per-request kwargs** (highest priority): `model.chat_complete(messages, temperature=0.5)`
2. **Config dictionary**: `AIFactory.create_language(..., config={"temperature": 0.7})`
3. **Provider default** (lowest priority)

## Response Structure

All LLM providers return standardized `ChatResponse` objects:

```python
response = model.chat_complete(messages)

# Access response content
response.content                              # Shortcut to message content
response.choices[0].message.content          # Full path to content
response.choices[0].message.role             # "assistant"

# Metadata
response.model                                # Model name used
response.usage.total_tokens                   # Total tokens consumed
response.usage.prompt_tokens                  # Input tokens
response.usage.completion_tokens              # Output tokens

# Streaming responses
chunk = next(model.chat_complete(messages, stream=True))
chunk.choices[0].delta.content                # Incremental content
```

### Handling Reasoning Traces

Some models (like Qwen3, DeepSeek R1) include chain-of-thought reasoning in `<think>` tags. The `Message` class provides convenient properties to parse these:

```python
response = model.chat_complete(messages)
msg = response.choices[0].message

# Full response with reasoning
msg.content          # "<think>Let me analyze...</think>\n\n42"

# Just the reasoning trace (returns None if no <think> tags)
msg.thinking         # "Let me analyze..."

# Just the actual answer (with <think> tags removed)
msg.cleaned_content  # "42"
```

Multiple `<think>` blocks are concatenated. If the response has no `<think>` tags, `thinking` returns `None` and `cleaned_content` returns the full content unchanged.

## Structured Output

Request JSON-formatted responses (where supported):

```python
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"structured": {"type": "json"}}
)

messages = [
    {"role": "user", "content": "List three countries in JSON format"}
]

response = model.chat_complete(messages)
# Response content will be valid JSON
```

**Supported Providers**: OpenAI, Anthropic, Google, Groq, OpenAI-Compatible (varies), Mistral, DeepSeek, xAI, OpenRouter, Azure, Perplexity

## Provider Selection

→ **See [Provider Comparison](../providers/README.md)** for detailed comparison and selection guide.

### Quick Provider Guide

- **OpenAI**: Industry standard, best overall quality, extensive model selection
- **Anthropic**: Excellent reasoning, long context support, safety-focused
- **Google (Gemini)**: Strong multimodal, competitive pricing
- **OpenAI-Compatible**: Local deployment (Ollama, LM Studio, vLLM)
- **Groq**: Fastest inference, limited model selection
- **Azure**: Enterprise compliance, private deployment
- **Ollama**: Local models, privacy-focused, no API costs

## Advanced Topics

- **Tool/Function Calling**: [docs/features/tool-calling.md](../features/tool-calling.md) - Let models call functions
- **Timeout Configuration**: [docs/advanced/timeout-configuration.md](../advanced/timeout-configuration.md)
- **LangChain Integration**: [docs/advanced/langchain-integration.md](../advanced/langchain-integration.md)
- **Model Discovery**: [docs/advanced/model-discovery.md](../advanced/model-discovery.md)
- **Resource Management**: [docs/advanced/connection-resource-management.md](../advanced/connection-resource-management.md)

## Examples

### Basic Chat Completion

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("openai", "gpt-4")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in one sentence."}
]

response = model.chat_complete(messages)
print(response.content)
```

### Streaming with Temperature Control

```python
model = AIFactory.create_language(
    "anthropic", "claude-3-5-sonnet-20241022",
    config={"streaming": True, "temperature": 0.3}
)

messages = [{"role": "user", "content": "Write a haiku about coding"}]

for chunk in model.chat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### JSON Output

```python
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "Extract key information as JSON: 'John Smith, age 30, lives in NYC'"
}]

response = model.chat_complete(messages)
print(response.content)  # Valid JSON string
```

### Multi-Turn Conversation

```python
model = AIFactory.create_language("google", "gemini-pro")

messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 15 * 24?"},
    {"role": "assistant", "content": "15 * 24 = 360"},
    {"role": "user", "content": "How did you calculate that?"}
]

response = model.chat_complete(messages)
print(response.content)
```

### Using Context Manager for Resource Management

For long-running applications or explicit resource control, use context managers:

```python
# Synchronous context manager
with AIFactory.create_language("openai", "gpt-4") as model:
    response = model.chat_complete(messages)
    print(response.content)
    # HTTP client automatically closed when exiting context

# Async context manager
import asyncio

async def main():
    async with AIFactory.create_language("openai", "gpt-4") as model:
        response = await model.achat_complete(messages)
        print(response.content)
        # Async HTTP client automatically closed

asyncio.run(main())
```

See [Resource Management](../advanced/connection-resource-management.md) for more details.

### Handling Reasoning Traces

```python
# Works with models that include <think> tags (Qwen3, DeepSeek R1, etc.)
model = AIFactory.create_language(
    "openai-compatible", "qwen/qwen3-4b",
    config={"base_url": "http://localhost:1234/v1"}
)

messages = [{"role": "user", "content": "What is 15 * 24?"}]
response = model.chat_complete(messages)
msg = response.choices[0].message

# Get just the answer, without the reasoning
print(msg.cleaned_content)  # "360"

# Or inspect the reasoning for debugging
if msg.thinking:
    print(f"Model's reasoning: {msg.thinking}")
```

### Tool Calling

```python
from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction

# Define a tool
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

# Use tools with any provider
model = AIFactory.create_language("openai", "gpt-4o")
response = model.chat_complete(
    [{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Tool: {tc.function.name}, Args: {tc.function.arguments}")
```

See [Tool Calling Guide](../features/tool-calling.md) for complete documentation.

## See Also

- [Provider Setup Guides](../providers/README.md)
- [Tool Calling](../features/tool-calling.md)
- [Embedding Models](./embedding.md)
- [Speech-to-Text](./speech-to-text.md)
- [Text-to-Speech](./text-to-speech.md)
