# PayPerQ (PPQ)

## Overview

[PayPerQ](https://ppq.ai) (PPQ) is a pay-as-you-go AI gateway that exposes hundreds of language models from many labs (OpenAI, Anthropic, Google, xAI, Qwen, DeepSeek, Mistral, and more) through a single OpenAI-compatible API. Instead of subscriptions, usage is billed per request, so a single API key gives access to the full catalog.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Full catalog of chat models via a single key |
| Embeddings | ❌ | Not available via profile (use OpenAI-Compatible) |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://ppq.ai

## Prerequisites

### Account Requirements
- A PayPerQ account with available credits
- An API key from the PayPerQ dashboard

### Getting API Keys
1. Visit https://ppq.ai and sign in
2. Open the API keys section of your account
3. Create a key (it looks like `sk-...`) and copy it

## Environment Variables

```bash
# PayPerQ API key (required)
PPQ_API_KEY="sk-..."

# Custom base URL (optional)
PPQ_BASE_URL="https://api.ppq.ai"
```

**Default base URL:** `https://api.ppq.ai`

## Quick Start

```python
from esperanto.factory import AIFactory

# Create a PayPerQ model
model = AIFactory.create_language("ppq", "claude-sonnet-5")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Available Models

PayPerQ proxies hundreds of models. Pass any model `id` returned by the
`GET https://api.ppq.ai/models` endpoint (or listed in the dashboard) as the
model name. A few examples:

| Model | Provider | Best For |
|-------|----------|----------|
| `auto` | PayPerQ | Automatic routing to a suitable model (default) |
| `gpt-5.4-mini` | OpenAI | Fast, cost-effective tasks |
| `claude-sonnet-5` | Anthropic | Balanced performance, long context |
| `claude-haiku-4.5` | Anthropic | Low-latency, cheap tasks |
| `gemini-3-flash-preview` | Google | Fast multimodal tasks |

> The catalog changes frequently. Use `AIFactory.get_provider_models("ppq")`
> or the `/models` endpoint to discover what is currently available.

## Features

### Streaming

```python
model = AIFactory.create_language("ppq", "claude-sonnet-5")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "ppq", "claude-sonnet-5",
    config={"structured": {"type": "json_object"}}
)
```

> JSON mode / `response_format` support depends on the underlying model that
> PayPerQ routes to. Most OpenAI and Anthropic models support it.

### Tool Calling

```python
from esperanto.common_types import Tool, ToolFunction

tools = [
    Tool(function=ToolFunction(
        name="get_weather",
        description="Get weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    ))
]

response = model.chat_complete(messages, tools=tools)
```

### Async Support

```python
response = await model.achat_complete(messages)
```

## Configuration

```python
# With explicit API key
model = AIFactory.create_language(
    "ppq", "claude-sonnet-5",
    config={"api_key": "your-key"}
)

# With custom base URL
model = AIFactory.create_language(
    "ppq", "claude-sonnet-5",
    config={"base_url": "https://api.ppq.ai"}
)
```

## Notes

- PayPerQ exposes an OpenAI-compatible endpoint (`/chat/completions`, `/models`) at the root of `https://api.ppq.ai` (no `/v1` suffix), so all standard Esperanto LLM features (streaming, tool calling, JSON mode) work with models that support them.
- Because PayPerQ aggregates many providers, feature support (JSON mode, tool calling, reasoning) depends on the specific model you select, not on PayPerQ itself.
- `AIFactory.get_provider_models("ppq")` returns the full catalog, which currently includes non-chat models (e.g. image/video). Filter by the model `id` you intend to use for text generation.
