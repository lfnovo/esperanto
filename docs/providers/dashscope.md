# DashScope

## Overview

DashScope (Alibaba Cloud Model Studio) provides advanced reranking capabilities, including the Qwen and GTE series models. These models offer strong performance for multilingual text reranking and instruction-following capabilities.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ❌ | Not available |
| Embeddings | ❌ | Not available |
| Reranking | ⚠️ | qwen3-rerank, gte-rerank-v2, * qwen3-vl-rerank |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

> ⚠️*: Some models are only available in China mainland region. Confirm your account's region before calling.



**Official Documentation:** https://www.alibabacloud.com/help/en/model-studio/

## Prerequisites

### Account Requirements
- Alibaba Cloud account
- Activated DashScope (Model Studio) service
- API key with credits

### Getting API Keys
1. Log in to the [Alibaba Cloud Console](https://dashscope.console.aliyun.com/)
2. Navigate to the API Key Management section
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
import asyncio

async def rerank_async():
    reranker = AIFactory.create_reranker("dashscope", "qwen3-rerank")
    results = await reranker.arerank(query, documents, top_k=3)
    return results

# Run async reranking
results = asyncio.run(rerank_async())
```



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



## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: DashScope API error (InvalidApiKey): ...
```
**Solution:** Verify your `DASHSCOPE_API_KEY` is correct.

**Rate Limit Error:**
```
Error: DashScope API error (Throttling.RateQuota): ...
```
**Solution:** Check your rate limits in the Alibaba Cloud console.

**Timeout Error:**
```
Error: Request to DashScope API timed out.
```
**Solution:** Increase timeout: `config={"timeout": 120.0}`

## See Also

- [Reranking Guide](../capabilities/reranking.md)
- [Jina Provider](./jina.md)
- [Voyage Provider](./voyage.md)
