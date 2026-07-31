# Z.ai

## Overview

Z.ai (formerly Zhipu AI) provides the GLM family of language models through an OpenAI-compatible API, with strong multilingual and coding capabilities.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | GLM family (glm-5.2, glm-4.5-flash) |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://docs.z.ai/

## Prerequisites

### Account Requirements
- Z.ai account at https://z.ai/
- API key with credits

### Getting API Keys
1. Visit https://z.ai/
2. Navigate to API Keys
3. Create and copy your API key

## Environment Variables

```bash
# Z.ai API key (required)
ZAI_API_KEY="your-api-key"

# Optional base URL override
ZAI_BASE_URL="https://api.z.ai/api/paas/v4"
```

**Default base URL:** `https://api.z.ai/api/paas/v4`

## Quick Start

```python
from esperanto.factory import AIFactory

# Create Z.ai model
model = AIFactory.create_language("zai", "glm-5.2")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Available Models

| Model | Best For |
|-------|----------|
| `glm-5.2` | Flagship model, complex tasks |
| `glm-4.5-flash` | Faster variant, latency-sensitive tasks |

Model discovery via `AIFactory.get_provider_models("zai")` returns only `glm-*` models.

## Features

### Streaming

```python
model = AIFactory.create_language("zai", "glm-5.2")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "zai", "glm-5.2",
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
    "zai", "glm-5.2",
    config={"api_key": "your-key"}
)

# With custom base URL
model = AIFactory.create_language(
    "zai", "glm-5.2",
    config={"api_key": "your-key", "base_url": "https://api.z.ai/api/paas/v4"}
)
```
