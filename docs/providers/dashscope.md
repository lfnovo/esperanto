# DashScope (Alibaba Cloud Qwen)

## Overview

DashScope is Alibaba Cloud's AI model service platform, providing access to the Qwen family of models through an OpenAI-compatible API. Qwen models offer strong multilingual performance, especially for Chinese and English tasks.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Qwen series (qwen-trubo, qwen-plus, qwen-max) |
| Embeddings | ❌ | Not available |
| Reranking | ⚠️ | qwen3-rerank, gte-rerank-v2, * qwen3-vl-rerank |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

> ⚠️*: Some models are only available in China mainland region. Confirm your account's region before calling.



**Official Documentation:** https://www.alibabacloud.com/help/en/model-studio/

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
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`DASHSCOPE_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Reranker model
reranker = AIFactory.create_reranker("dashscope", "qwen3-rerank")
```

### Direct Instantiation

```python
from esperanto.providers.reranker.dashscope import DashScopeRerankerModel

# Reranker model
reranker = DashScopeRerankerModel(
    api_key="your-api-key",
    model_name="qwen3-rerank"
)
```

## Capabilities

### Reranking

**Available Models:**

| Model | Max Request Token Length | Max Single Query / Document Token Length | Max Document Amount |
|-------|---------|----------|-------|
| **qwen3-rerank** | 30K | 4K | 500 |
| **gte-rerank-v2** | 30K | 4K | 500 |
| **qwen3-vl-rerank** | 800K | 8K | 100 |

**Configuration:**

```python
from esperanto.factory import AIFactory

reranker = AIFactory.create_reranker(
    "dashscope",
    "qwen3-rerank",
    config={
        "timeout": 30.0
    }
)
```

**Example - Basic Reranking:**

```python
from esperanto.factory import AIFactory

# Create reranker
reranker = AIFactory.create_reranker("dashscope", "qwen3-rerank")

query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "The weather forecast shows rain tomorrow.",
    "Python is a popular programming language for machine learning.",
    "Deep learning uses neural networks with multiple layers.",
    "Coffee is best served hot in the morning."
]

# Rerank documents by relevance
results = reranker.rerank(query, documents, top_k=3)

# Results sorted by relevance (highest first)
for i, result in enumerate(results.results):
    print(f"{i+1}. Score: {result.relevance_score:.3f}")
    print(f"   Document: {result.document[:60]}...\n")
```

**Example - Using Instructions (Qwen3):**

The `qwen3-rerank` model supports an `instruct` parameter to guide the reranking behavior.

```python
# Rerank with specific instructions
results = reranker.rerank(
    query="smartphone",
    documents=[
        "A device used for communication and running apps.",
        "A fruit that is yellow and curved.",
        "The latest iPhone 15 features."
    ],
    instruct="Rank based on relevance to consumer electronics products."
)
```

**Example - Async Reranking:**

```python
from esperanto.factory import AIFactory
import asyncio

async def rerank_async():
    reranker = AIFactory.create_reranker("dashscope", "qwen3-rerank")
    results = await reranker.arerank(query, documents, top_k=3)
    return results

# Run async reranking
results = asyncio.run(rerank_async())
```

### Language
**Custom base URL (optional, for international endpoint)**
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

**Available Models**

| Model | Context Window | Best For |
|-------|---------------|----------|
| `qwen-turbo` | 128K | Fast, cost-effective tasks |
| `qwen-plus` | 128K | Balanced performance and cost |
| `qwen-max` | 32K | Most capable, complex reasoning |
| `qwen-max-longcontext` | 1M | Very long document processing |


## Advanced Features

### LangChain Integration

Convert to LangChain models:

```python
from esperanto.factory import AIFactory

# Reranker model
reranker = AIFactory.create_reranker("dashscope", "qwen3-rerank")
langchain_reranker = reranker.to_langchain()

# Use with LangChain compression
from langchain.schema import Document
docs = [Document(page_content=text) for text in documents]
compressed = langchain_reranker.compress_documents(docs, query)
```



### Multi-modal Reranker

Aliyun platform provides `qwen3-vl-rerank` as rerank model accepting multi-modal inputs (**text, image and video**).

This model shares almost the same interface as other rerankers.

```python
from esperanto.factory import AIFactory

reranker = AIFactory.create_reranker("dashscope", "qwen3-vl-rerank")
query = "What is reranking model?"
documents = [
    {"text": "Text ranking models are widely used in search engines and recommendation systems, which rank candidate texts based on their relevance"},
    {"image": "https://img.alicdn.com/imgextra/i3/O1CN01rdstgY1uiZWt8gqSL_!!6000000006071-0-tps-1970-356.jpg"},
    {"video": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250107/lbcemt/new+video.mp4"}
]

responses = reranker.rerank(
	query=query,
    documents=documents,
    top_k=2,
    instruct="Given a web search query, retrieve relevant passages that answer the query.",
    fps=0.5  # Control rate of frame extracted from the uploaded video. Between [0, 1] and default to 1.0
)
```

- The image and video inputs can be given by any public-accessible URL. Image inputs can also be encoded as base-64 format.
- Refer to [Aliyun Reranker Models](https://help.aliyun.com/zh/model-studio/text-rerank-api) for more details.

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
