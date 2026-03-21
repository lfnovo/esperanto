# DashScope (Alibaba Cloud Qwen)

## Overview

DashScope is Alibaba Cloud's AI model service platform, providing access to the Qwen family of models through an OpenAI-compatible API. Qwen models offer strong multilingual performance, especially for Chinese and English tasks.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Qwen series (qwen-turbo, qwen-plus, qwen-max) |
| Embeddings | ❌ | Not available via profile (use OpenAI-Compatible) |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://help.aliyun.com/zh/model-studio/

## Prerequisites

### Account Requirements
- Alibaba Cloud account with DashScope access
- API key from the DashScope console

### Getting API Keys
1. Visit https://dashscope.console.aliyun.com/
2. Navigate to API Keys management
3. Create and copy your API key

## Environment Variables

```bash
# DashScope API key (required)
DASHSCOPE_API_KEY="sk-..."

# Custom base URL (optional, for international endpoint)
DASHSCOPE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

**Default base URL:** `https://dashscope.aliyuncs.com/compatible-mode/v1`

## Quick Start

```python
from esperanto.factory import AIFactory

# Create DashScope model
model = AIFactory.create_language("dashscope", "qwen-plus")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Available Models

| Model | Context Window | Best For |
|-------|---------------|----------|
| `qwen-turbo` | 128K | Fast, cost-effective tasks |
| `qwen-plus` | 128K | Balanced performance and cost |
| `qwen-max` | 32K | Most capable, complex reasoning |
| `qwen-max-longcontext` | 1M | Very long document processing |

## Features

### Streaming

```python
model = AIFactory.create_language("dashscope", "qwen-plus")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "dashscope", "qwen-plus",
    config={"structured": {"type": "json_object"}}
)
```

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
    "dashscope", "qwen-plus",
    config={"api_key": "your-key"}
)

# With custom base URL (e.g., international endpoint)
model = AIFactory.create_language(
    "dashscope", "qwen-plus",
    config={"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"}
)
```

## Notes

- DashScope uses Alibaba Cloud's OpenAI-compatible endpoint, so all standard Esperanto LLM features (streaming, tool calling, JSON mode) are supported.
- Some OpenAI-specific parameters like `logit_bias` may not be supported on all models.
- The `enable_search` DashScope-specific parameter is not exposed through Esperanto's interface. For advanced DashScope features, consider using their native SDK.
