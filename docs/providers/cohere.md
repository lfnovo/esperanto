# Cohere

## Overview

Cohere provides enterprise-grade language, embedding, and reranking models. Esperanto integrates Cohere as a **first-class provider** via its native v2 API (`/v2/chat`, `/v2/embed`, `/v2/rerank`), because Cohere's differentiators — `documents`-based RAG, citations, and `input_type` for embeddings — are not exposed through its OpenAI-compatibility endpoint.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | `command-a-03-2025` (default), chat, streaming, tool calling |
| Embeddings | ✅ | `embed-v4.0` (default), `input_type` aware, auto-batched |
| Reranking | ✅ | `rerank-v4.0-pro` (default), `rerank-v3.5` |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://docs.cohere.com

## Environment Variables

```bash
# Cohere API key (required), shared across all three capabilities
COHERE_API_KEY="..."
```

**Variable Priority:**
1. Direct parameter / config (`config={"api_key": "..."}`)
2. Environment variable (`COHERE_API_KEY`)

## Quick Start

```python
from esperanto.factory import AIFactory

llm = AIFactory.create_language("cohere", "command-a-03-2025")
embedder = AIFactory.create_embedding("cohere", "embed-v4.0")
reranker = AIFactory.create_reranker("cohere", "rerank-v4.0-pro")
```

## Language Models

```python
llm = AIFactory.create_language("cohere", "command-a-03-2025")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain RAG in one sentence."},
]

# Standard chat
response = llm.chat_complete(messages)
print(response.choices[0].message.content)

# Streaming
for chunk in llm.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

### Tool Calling

```python
from esperanto.common_types import Tool, ToolFunction

tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    )
]

response = llm.chat_complete(
    [{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)
for tc in response.choices[0].message.tool_calls or []:
    print(tc.function.name, tc.function.arguments)
```

### Cohere-specific RAG (`documents` / `citations`)

Cohere's built-in retrieval features ride along on the request via `config`. They have no analog on other Esperanto providers, so they are passed through without changing the universal `chat_complete` interface:

```python
llm = AIFactory.create_language(
    "cohere",
    "command-a-03-2025",
    config={
        "documents": [{"id": "doc1", "data": {"text": "Paris is the capital of France."}}],
        "citation_options": {"mode": "ACCURATE"},
    },
)
response = llm.chat_complete([{"role": "user", "content": "What is the capital of France?"}])
```

> **Note:** Citations returned by Cohere are **not** surfaced on the universal `ChatCompletion` response in this version (out of scope — see ARCHITECTURE.md). The `documents`/`citation_options`/`connectors` config fields only affect the request.

> Cohere uses `p` for nucleus sampling — Esperanto's `top_p` parameter is mapped to `p` automatically.

## Embeddings

Cohere requires an `input_type`. Esperanto defaults to `search_document`; override via config or per call. Texts are automatically batched to respect Cohere's 96-texts-per-request limit.

```python
# For documents to be indexed
embedder = AIFactory.create_embedding("cohere", "embed-v4.0")
doc_vectors = embedder.embed(["First document.", "Second document."])

# For a search query
query_embedder = AIFactory.create_embedding(
    "cohere", "embed-v4.0", config={"input_type": "search_query"}
)
query_vector = query_embedder.embed(["my search query"])

# Per-call override
vectors = embedder.embed(["classify me"], input_type="classification")
```

Valid `input_type` values: `search_document`, `search_query`, `classification`, `clustering`, `image`.

## Reranking

```python
reranker = AIFactory.create_reranker("cohere", "rerank-v4.0-pro")

result = reranker.rerank(
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of AI.",
        "The weather is nice today.",
        "Neural networks are used in ML.",
    ],
    top_k=2,
)
for r in result.results:
    print(r.index, r.relevance_score, r.document)
```

## Model Discovery

```python
from esperanto import AIFactory

models = AIFactory.get_provider_models("cohere")  # uses COHERE_API_KEY
for m in models:
    print(m.id, m.type)
```

Discovery queries Cohere's live `/v1/models` endpoint and tags each model's type (`language`, `embedding`, `reranker`) from its supported endpoints. Results are cached for 1 hour.

## Notes

- No SDK dependency — integration is pure HTTP via `httpx`.
- `to_langchain()` (LLM and embeddings) requires `langchain_cohere` (`pip install langchain_cohere`).
- Non-float embedding types (`int8`, `binary`, etc.) are out of scope; embeddings default to `float`.
