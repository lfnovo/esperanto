# MiniMax

## Overview

MiniMax provides high-performance language models with context windows up to one million tokens through an OpenAI-compatible API.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | MiniMax-M3, MiniMax-M2.7 |
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

### Regional endpoints

MiniMax keys are region-specific. Override `MINIMAX_BASE_URL` to select the
matching region, otherwise the international endpoint is used.

| Region | OpenAI-compatible | Anthropic-compatible | Docs |
|--------|-------------------|----------------------|------|
| International (`global_en`) | `https://api.minimax.io/v1` | `https://api.minimax.io/anthropic` | https://platform.minimax.io/docs |
| Mainland China (`cn_zh`) | `https://api.minimaxi.com/v1` | `https://api.minimaxi.com/anthropic` | https://platform.minimaxi.com/docs |

```bash
# Mainland China endpoint (keys are region-specific)
MINIMAX_BASE_URL="https://api.minimaxi.com/v1"
```

### Anthropic-compatible endpoint

MiniMax also exposes an Anthropic-protocol API. Route Esperanto's native
Anthropic provider at it by passing `base_url`:

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

The declared regional Anthropic URLs are also available on the MiniMax profile
under `regional_endpoints`.

## Quick Start

```python
from esperanto.factory import AIFactory

# Create MiniMax model
model = AIFactory.create_language("minimax", "MiniMax-M3")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Available Models

| Model | Context Window | Input Modalities | Thinking | Input / Output per 1M tokens |
|-------|----------------|------------------|----------|------------------------------|
| `MiniMax-M3` | 1M | Text, image, video | Adaptive or disabled | $0.60 / $2.40 |
| `MiniMax-M2.7` | 204.8K | Text | Always on | $0.30 / $1.20 |

Cached input costs $0.12 per million tokens for MiniMax-M3 and $0.06 per
million tokens for MiniMax-M2.7. MiniMax-M2.7 cache writes cost $0.375 per
million tokens.

## Features

### Streaming

```python
model = AIFactory.create_language("minimax", "MiniMax-M3")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "minimax", "MiniMax-M3",
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
    "minimax", "MiniMax-M3",
    config={"api_key": "your-key"}
)
```
