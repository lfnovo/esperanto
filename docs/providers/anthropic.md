# Anthropic

## Overview

Anthropic provides access to the Claude family of large language models, known for their strong performance on reasoning, analysis, and longer-form tasks.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Claude Opus 5.5, Sonnet 5.5, Fable 5.1, Sonnet 5, Haiku 4.5 |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://docs.anthropic.com

## Prerequisites

### Account Requirements
- Anthropic account (sign up at https://console.anthropic.com)
- API key with credits or billing enabled

### Getting API Keys
1. Visit https://console.anthropic.com/settings/keys
2. Click "Create Key"
3. Copy and store the key securely

## Environment Variables

```bash
# Anthropic API key (required)
ANTHROPIC_API_KEY="sk-ant-..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`ANTHROPIC_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Create Claude model
model = AIFactory.create_language("anthropic", "claude-sonnet-5")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Direct Instantiation

```python
from esperanto.providers.llm.anthropic import AnthropicLanguageModel

# Create model instance
model = AnthropicLanguageModel(
    api_key="your-api-key",
    model_name="claude-sonnet-5"
)

# Use the model
messages = [{"role": "user", "content": "Hello!"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Capabilities

### Language Models (LLM)

**Available Models:**

| Model | Context Window | Best For |
|-------|----------------|----------|
| **claude-opus-5-5** | 1M tokens | Current Opus. Complex tasks |
| **claude-sonnet-5-5** | 1M tokens | Current Sonnet. Fast, capable everyday work |
| **claude-fable-5-1** | 1M tokens | Most capable model, for the hardest reasoning tasks |
| **claude-sonnet-5** | 1M tokens | Default. Balanced performance and speed |
| **claude-opus-5** | 1M tokens | Previous Opus |
| **claude-opus-4-5-20251101** | 200K tokens | Pinned Opus 4.5 |
| **claude-sonnet-4-5-20250929** | 1M tokens | Pinned Sonnet 4.5 |
| **claude-haiku-4-5-20251001** | 200K tokens | Fast responses, cost-effective |

Each id names one model generation: `claude-sonnet-5` does not move to Sonnet
5.5. Change the id to use a newer generation.

Claude Opus 5.5 and Fable 5.1 always think, and Sonnet 5.5 thinks by default.
Thinking counts against `max_tokens`, so leave enough room for the answer (see
[Empty Structured Responses](#empty-structured-responses-and-thinking-budgets)).
These models (and Claude Mythos 5.1) also reject forced tool choice
(`tool_choice="required"` or a specific tool). Esperanto sends `tool_choice="auto"` instead and emits a
`UserWarning`, so the same code keeps working, but a tool call is no longer
guaranteed. Name the tool in the prompt to steer the model, or use
`structured={"type": "json_schema", ...}` when you only need JSON back.

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-5",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 1.0)
        "max_tokens": 1024,           # Maximum response length (required)
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode
        "timeout": 60.0               # Request timeout
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create Claude model
model = AIFactory.create_language("anthropic", "claude-sonnet-5")

# Simple chat
messages = [
    {"role": "user", "content": "What's the capital of France?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - With System Message:**

```python
# Claude handles system messages naturally
messages = [
    {"role": "system", "content": "You are a helpful assistant specializing in geography."},
    {"role": "user", "content": "Tell me about the capital of Japan."}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Async Chat:**

```python
async def chat_async():
    model = AIFactory.create_language("anthropic", "claude-sonnet-5")

    messages = [{"role": "user", "content": "Explain quantum computing"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Run async
# await chat_async()
```

**Example - Streaming:**

```python
# Synchronous streaming
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in await model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Example - JSON Mode:**

```python
# Enable JSON output
model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-5",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three programming languages with their typical use cases as JSON"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
# JSON mode is prompt-guided on Anthropic; see "JSON Mode" below
```

**Example - Long Context:**

```python
# Current Claude models have a 1M-token context window (Haiku 4.5: 200K)
long_document = """
[Your long document content here - up to the model's context window]
"""

messages = [
    {"role": "user", "content": f"Summarize this document:\n\n{long_document}"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Multi-turn Conversation:**

```python
# Build conversation history
messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."},
    {"role": "user", "content": "What are its main advantages?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Temperature Control:**

```python
# More creative (higher temperature)
creative_model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-5",
    config={"temperature": 1.0, "max_tokens": 1024}
)

# More focused (lower temperature)
focused_model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-5",
    config={"temperature": 0.2, "max_tokens": 1024}
)

messages = [{"role": "user", "content": "Write a creative story about AI."}]

creative_response = creative_model.chat_complete(messages)
focused_response = focused_model.chat_complete(messages)
```

## Advanced Features

### JSON Mode
`structured={"type": "json"}` is **prompt-guided only** on Anthropic. The Anthropic API has no generic JSON mode: current models reject assistant prefill, and `output_config` only accepts a concrete schema. Esperanto therefore sends no constraint, and the model usually returns JSON because the prompt asks for it. Esperanto emits a `UserWarning` when you use this mode with Anthropic, in `chat_complete()`, `achat_complete()` and `to_langchain()`.

For guaranteed JSON, use [schema-driven structured output](#schema-driven-structured-output-v1) (`{"type": "json_schema", "schema": ...}`), which Anthropic enforces on Claude 4.5 models and newer (Opus 4.5+, Sonnet 4.5+, Haiku 4.5).

```python
model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-5",
    config={"structured": {"type": "json"}}  # warns: prompt-guided only
)

messages = [{
    "role": "user",
    "content": "Create a JSON object with user information"
}]

response = model.chat_complete(messages)
```

### Empty Structured Responses and Thinking Budgets
Claude Opus 5.5 always thinks, and its `max_tokens` covers thinking **plus** the answer. When thinking uses up the whole budget, or the model refuses, the response has no text. In any structured mode, when a response has no text and no tool calls, Esperanto raises `EmptyCompletionError` instead of returning empty content:

```python
from esperanto import AIFactory, EmptyCompletionError

model = AIFactory.create_language(
    "anthropic",
    "claude-opus-5-5",
    config={"structured": {"type": "json"}, "max_tokens": 4096}
)
messages = [{"role": "user", "content": "List three fruits and their colors as JSON"}]

try:
    response = model.chat_complete(messages)
except EmptyCompletionError as e:
    print(e.model, e.finish_reason)  # e.g. "claude-opus-5-5", "length"
```

`finish_reason` is `"length"` when the budget ran out (raise `max_tokens`) and `"content_filter"` when the model refused. Anthropic stop reasons are normalized to the same values the other providers use: `end_turn` becomes `"stop"`, `tool_use` becomes `"tool_calls"`, `max_tokens` becomes `"length"`, and `refusal` becomes `"content_filter"`.

### Schema-Driven Structured Output (v1)
Anthropic also supports schema-constrained outputs via `structured={"type": "json_schema", ...}`:

```python
from pydantic import BaseModel
from esperanto.factory import AIFactory

class TravelPlan(BaseModel):
    summary: str
    next_steps: list[str]

model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-5",
    config={
        "structured": {
            "type": "json_schema",
            "schema": TravelPlan,      # or a JSON schema dict
            "name": "travel_plan",     # optional
            "strict": True             # optional
        }
    }
)

response = model.chat_complete(
    [{"role": "user", "content": "Plan a 3-day trip to Paris"}]
)

print(response.content)      # Raw JSON string
print(response.structured)   # Parsed/validated TravelPlan instance
```

Notes:
- Schema mode is currently non-streaming in Esperanto v1 (`stream=True` raises `ValueError`).
- A response with empty content and no tool calls raises `EmptyCompletionError` (see [Empty Structured Responses](#empty-structured-responses-and-thinking-budgets)).
- Anthropic strict tool-use schema enforcement (`tools[].strict`) is a separate feature and is not part of this v1 schema-output path.

### Temperature and Top-P Priority
Claude prioritizes temperature over top_p when both are specified:

```python
# Temperature takes precedence
model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-5",
    config={
        "temperature": 0.7,  # This will be used
        "top_p": 0.9         # This will be ignored
    }
)
```

### Timeout Configuration
Customize request timeouts:

```python
# Extended timeout for complex tasks
model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-5",
    config={
        "timeout": 120.0,    # 2 minutes
        "max_tokens": 4096
    }
)
```

### LangChain Integration
Convert to LangChain models:

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("anthropic", "claude-sonnet-5")
langchain_model = model.to_langchain()

# Use with LangChain
response = langchain_model.invoke("Hello!")
print(response.content)
```

## Model Selection Guide

### Claude Sonnet 5 (Default) and Sonnet 5.5
**Best for:** Most use cases, balanced performance
- Strong reasoning and analysis at moderate cost
- 1M token context window
- Sonnet 5.5 is the current Sonnet; it thinks by default and does not support forced tool choice (Esperanto falls back to `auto` with a warning)

```python
model = AIFactory.create_language("anthropic", "claude-sonnet-5")    # default
model = AIFactory.create_language("anthropic", "claude-sonnet-5-5")  # current Sonnet
```

### Claude Opus 5.5
**Best for:** Complex reasoning and agentic work
- Current Opus; thinking is always on and counts against `max_tokens`
- 1M token context window
- Does not support forced tool choice (Esperanto falls back to `auto` with a warning)

```python
model = AIFactory.create_language("anthropic", "claude-opus-5-5")
```

### Claude Fable 5.1
**Best for:** The hardest reasoning tasks, when quality matters more than cost
- Anthropic's most capable widely available model
- Thinking is always on; requests can run for minutes on hard tasks
- 1M token context window

```python
model = AIFactory.create_language("anthropic", "claude-fable-5-1")
```

### Claude Haiku 4.5
**Best for:** High-volume, fast responses
- Fastest and most cost-effective Claude model
- 200K token context window

```python
model = AIFactory.create_language("anthropic", "claude-haiku-4-5-20251001")
```

## Performance Characteristics

### Context Window
Current Claude models support a 1M-token context window; Claude Haiku 4.5 and
Claude Opus 4.5 support 200K. See the [models table](#language-models-llm) or
`AIFactory.get_provider_models("anthropic")` for each model's window.

### Response Quality
- **Fable**: Highest quality, for the hardest reasoning tasks
- **Opus**: Very high quality, strong reasoning at a lower cost than Fable
- **Sonnet**: Excellent balance of quality and speed
- **Haiku**: Fast, still maintains good quality

### Speed
- **Haiku**: Fastest (sub-second for short responses)
- **Sonnet**: Fast (1-3 seconds typical)
- **Opus**: Slower but most thorough (3-10 seconds)

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key is correct and active in the Anthropic console.

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic with exponential backoff or contact Anthropic for higher limits.

**Context Length Exceeded:**
```
Error: Prompt is too long
```
**Solution:** Reduce the total tokens in your messages to fit the model's context window (1M tokens on current models, 200K on Haiku 4.5).

**Missing max_tokens:**
```
Error: max_tokens is required
```
**Solution:** Always specify max_tokens in your configuration:
```python
config={"max_tokens": 1024}
```

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase the timeout configuration:
```python
config={"timeout": 120.0, "max_tokens": 1024}
```

### Best Practices

1. **Always Set max_tokens:** Unlike some providers, Anthropic requires max_tokens to be specified.

2. **Use Appropriate Model:** Choose the right model for your use case:
   - Haiku for speed and cost
   - Sonnet for balanced performance
   - Opus for complex reasoning

3. **System Messages:** Claude handles system messages naturally - use them to set context and behavior.

4. **Long Context:** Take advantage of the 1M-token context window (200K on Haiku 4.5) for complex tasks.

5. **Temperature Settings:** Use lower temperatures (0.2-0.5) for factual tasks, higher (0.7-1.0) for creative tasks.

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [OpenAI Provider](./openai.md)
- [Google Provider](./google.md)
- [Groq Provider](./groq.md)
