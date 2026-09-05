# SiliconFlow

## Overview

SiliconFlow provides an OpenAI-compatible API for DeepSeek, Qwen, and other language models. Esperanto exposes it as a built-in OpenAI-compatible provider profile.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | DeepSeek, Qwen, and other SiliconFlow-hosted models |
| Embeddings | ❌ | Not exposed by this provider profile |
| Reranking | ❌ | Not exposed by this provider profile |
| Speech-to-Text | ❌ | Not exposed by this provider profile |
| Text-to-Speech | ❌ | Not exposed by this provider profile |

**Official Documentation:** https://docs.siliconflow.com/ (China: https://docs.siliconflow.cn/)

## Prerequisites

### Account Requirements
- SiliconFlow account
- API key with credits

### Getting API Keys
1. Visit the SiliconFlow console.
2. Navigate to API Keys.
3. Create and copy your API key.

## Environment Variables

```bash
# SiliconFlow API key (required)
SILICONFLOW_API_KEY="your-api-key"

# Optional: override the default global endpoint with the China endpoint
SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"
```

**Default base URL:** `https://api.siliconflow.com/v1`

## Quick Start

```python
from esperanto.factory import AIFactory

# Uses the default global endpoint
model = AIFactory.create_language("siliconflow", "deepseek-ai/DeepSeek-V3.1-Terminus")

messages = [{"role": "user", "content": "Explain retrieval augmented generation"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## China Endpoint

SiliconFlow runs separate platforms for global (`siliconflow.com`) and mainland China (`siliconflow.cn`) accounts, and keys only work on their own platform. If your account is on the China platform, point Esperanto at it by setting `SILICONFLOW_BASE_URL` or by passing `config["base_url"]`.

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "siliconflow",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    config={"base_url": "https://api.siliconflow.cn/v1"},
)
```

## Available Models

Model availability depends on your SiliconFlow account and selected endpoint. Use model discovery to fetch the available model list:

```python
from esperanto.factory import AIFactory

models = AIFactory.get_provider_models("siliconflow")
for model in models:
    print(model.id)
```

## Features

### Streaming

```python
model = AIFactory.create_language("siliconflow", "deepseek-ai/DeepSeek-V3.1-Terminus")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "siliconflow",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
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
    "siliconflow",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    config={"api_key": "your-key"},
)

# With explicit API key and China endpoint
model = AIFactory.create_language(
    "siliconflow",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    config={
        "api_key": "your-key",
        "base_url": "https://api.siliconflow.cn/v1",
    },
)
```
