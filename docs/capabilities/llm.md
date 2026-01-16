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

See [Resource Management](../configuration.md#http-client-resource-management) for more details.

## See Also

- [Provider Setup Guides](../providers/README.md)
- [Embedding Models](./embedding.md)
- [Speech-to-Text](./speech-to-text.md)
- [Text-to-Speech](./text-to-speech.md)
