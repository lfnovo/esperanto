# MiniMax

## Overview

MiniMax provides high-performance language models with large context windows (204K tokens) at competitive pricing through an OpenAI-compatible API.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | MiniMax-M2.5 series |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://platform.minimaxi.com/

## Prerequisites

### Account Requirements
- MiniMax account at https://platform.minimaxi.com/
- API key with credits

### Getting API Keys
1. Visit https://platform.minimaxi.com/
2. Navigate to API Keys
3. Create and copy your API key

## Environment Variables

```bash
# MiniMax API key (required)
MINIMAX_API_KEY="your-api-key"
```

**Default base URL:** `https://api.minimax.io/v1`

## Quick Start

```python
from esperanto.factory import AIFactory

# Create MiniMax model
model = AIFactory.create_language("minimax", "MiniMax-M2.5")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Available Models

| Model | Context Window | Best For |
|-------|---------------|----------|
| `MiniMax-M2.5` | 204K | Flagship model, complex tasks |
| `MiniMax-M2.5-highspeed` | 204K | Faster variant, latency-sensitive tasks |

## Features

### Streaming

```python
model = AIFactory.create_language("minimax", "MiniMax-M2.5")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "minimax", "MiniMax-M2.5",
    config={"structured": {"type": "json_object"}}
)
```

### Async Support

```python
response = await model.achat_complete(messages)
```

## Configuration

```python
# With explicit API key
model = AIFactory.create_language(
    "minimax", "MiniMax-M2.5",
    config={"api_key": "your-key"}
)
```

## Anthropic-compatible endpoint

MiniMax also serves its models over an Anthropic-protocol API. Point Esperanto's
native `anthropic` provider at it by overriding `base_url`:

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "anthropic",
    "MiniMax-M3",
    config={
        "api_key": "your-minimax-key",
        "base_url": "https://api.minimax.io/anthropic",
    },
)
```

Like the OpenAI-compatible endpoint, the Anthropic-compatible one is regional
and keys only work against their own region:

| Region | Anthropic-compatible base URL |
|--------|-------------------------------|
| International | `https://api.minimax.io/anthropic` |
| Mainland China | `https://api.minimax.cn/anthropic` (the legacy `https://api.minimaxi.com/anthropic` host also responds) |
