# AIHubMix

## Overview

AIHubMix is an OpenAI-compatible gateway that proxies access to models from many upstream vendors (OpenAI, Anthropic, Google, Meta, and others) behind a single API key. Esperanto exposes it as a built-in OpenAI-compatible provider profile.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Access to OpenAI, Anthropic, Google, and other vendors' models through one gateway |
| Embeddings | ❌ | Not exposed by this provider profile |
| Reranking | ❌ | Not exposed by this provider profile |
| Speech-to-Text | ❌ | Not exposed by this provider profile |
| Text-to-Speech | ❌ | Not exposed by this provider profile |

**Official Documentation:** https://aihubmix.com/

## Prerequisites

### Account Requirements
- AIHubMix account
- API key with credits

### Getting API Keys
1. Visit the AIHubMix console.
2. Navigate to API Keys.
3. Create and copy your API key.

## Environment Variables

```bash
# AIHubMix API key (required)
AIHUBMIX_API_KEY="your-api-key"

# Optional: override the default endpoint
AIHUBMIX_BASE_URL="https://aihubmix.com/v1"
```

**Default base URL:** `https://aihubmix.com/v1`

## Quick Start

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("aihubmix", "gpt-4o-mini")

messages = [{"role": "user", "content": "Explain retrieval augmented generation"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Available Models

AIHubMix proxies models from many upstream vendors, so the catalog is large and changes over time. Use model discovery to fetch the current list instead of hardcoding IDs:

```python
from esperanto.factory import AIFactory

models = AIFactory.get_provider_models("aihubmix")
for model in models:
    print(model.id, model.owned_by)
```

## Features

### Streaming

```python
model = AIFactory.create_language("aihubmix", "gpt-4o-mini")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "aihubmix",
    "gpt-4o-mini",
    config={"structured": {"type": "json_object"}},
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
    "aihubmix",
    "gpt-4o-mini",
    config={"api_key": "your-key"},
)

# With explicit API key and a custom endpoint
model = AIFactory.create_language(
    "aihubmix",
    "gpt-4o-mini",
    config={
        "api_key": "your-key",
        "base_url": "https://custom.aihubmix.com/v1",
    },
)
```

### Optional APP-Code Header

AIHubMix supports an optional `APP-Code` header for channel/app attribution (and, for most models other than Claude, a discount). Generate your own code at [aihubmix.com/appstore](https://aihubmix.com/appstore), then pass it via `extra_headers`:

```python
model = AIFactory.create_language(
    "aihubmix",
    "gpt-4o-mini",
    config={
        "api_key": "your-key",
        "extra_headers": {"APP-Code": "your-app-code"},
    },
)
```

This is not required for normal usage. See [Sending Custom Headers](openai-compatible.md#sending-custom-headers) for the general mechanism, which applies to any provider built on `OpenAICompatibleLanguageModel`.
