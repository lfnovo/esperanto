# Eden AI

## Overview

[Eden AI](https://www.edenai.co) is a European AI gateway that exposes language and embedding models from many labs (OpenAI, Anthropic, Google, Mistral, Cohere, DeepSeek, Qwen, and more) through a single OpenAI-compatible API and one key. Eden AI is a French company and the gateway runs on EU infrastructure, which matters for teams with European data-residency requirements.

**Two endpoints, one API surface and one key:**

| Endpoint | What it serves |
|----------|----------------|
| `https://api.edenai.run/v3` | the default, serves the full catalog |
| `https://api.eu.edenai.run/v3` | keeps inference inside the EU, and serves only the subset of the catalog available there, so it is a genuinely smaller list |

Model ids are identical on both, so switching is a base URL change. A model that resolves on the default endpoint may not be reachable through the EU one, so list the models against the endpoint you intend to use before switching.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | OpenAI-compatible `/chat/completions` (default `openai/gpt-5.5`) |
| Embeddings | ✅ | OpenAI-compatible `/embeddings` (default `openai/text-embedding-3-small`) |
| Reranking | ❌ | Not exposed through this profile |
| Speech-to-Text | ❌ | Not exposed through this profile |
| Text-to-Speech | ❌ | Not exposed through this profile |

> Both capabilities resolve under the `edenai` profile, `create_language` and
> `create_embedding`, against `https://api.edenai.run/v3` with `EDENAI_API_KEY`.
> Pass a `model_name` to pick a specific model; omit it to use the defaults above.

**Official Documentation:** https://www.edenai.co/docs

## Prerequisites

### Account Requirements
- An Eden AI account
- An API key from the Eden AI dashboard

### Getting API Keys
1. Visit https://www.edenai.co and sign in
2. Open the API keys section of your account
3. Create a key (it looks like `sk-eden-...`) and copy it

## Environment Variables

```bash
# Eden AI API key (required)
EDENAI_API_KEY="sk-eden-..."

# Custom base URL (optional). Set this to the EU endpoint to keep
# inference inside the EU.
EDENAI_BASE_URL="https://api.eu.edenai.run/v3"
```

**Default base URL:** `https://api.edenai.run/v3`

## Quick Start

```python
from esperanto.factory import AIFactory

# Create an Eden AI model
model = AIFactory.create_language("edenai", "openai/gpt-5.5")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Embeddings

```python
embedder = AIFactory.create_embedding("edenai", "openai/text-embedding-3-small")
vectors = embedder.embed(["Eden AI is a European AI gateway"])
```

Embedding models are listed on their own endpoint, `/v3/embeddings/models`, and
none of them appears in `/v3/models`. Discovery queries both, so filter by
modality rather than assuming one listing holds everything:

```python
llms = AIFactory.get_provider_models("edenai", model_type="language")
embedders = AIFactory.get_provider_models("edenai", model_type="embedding")
```

See https://www.edenai.co/docs/v3/llms/embeddings.

### EU endpoint

```python
model = AIFactory.create_language(
    "edenai", "mistral/mistral-large-latest",
    config={"base_url": "https://api.eu.edenai.run/v3"}
)
```

> The default embedding model, `openai/text-embedding-3-small`, is **not**
> served on the EU endpoint. Pass an EU-eligible id such as
> `mistral/mistral-embed` or `mistral/codestral-embed`, or list what is
> actually available there before switching:
>
> ```python
> AIFactory.get_provider_models(
>     "edenai",
>     model_type="embedding",
>     base_url="https://api.eu.edenai.run/v3",
> )
> ```
>
> See https://www.edenai.co/docs/v3/data-governance/eu-endpoint.

## Available Models

Model ids are namespaced by the upstream vendor, as `<vendor>/<model>`. Pass any
model `id` returned by `GET https://api.edenai.run/v3/models` for chat, or by
`GET https://api.edenai.run/v3/embeddings/models` for embeddings. The two
listings do not overlap. A few examples:

| Model | Vendor | Best For |
|-------|--------|----------|
| `openai/gpt-5.5` | OpenAI | General-purpose chat (default) |
| `openai/gpt-5-mini` | OpenAI | Fast, cost-effective tasks |
| `anthropic/claude-sonnet-latest` | Anthropic | Balanced performance, long context |
| `mistral/mistral-large-latest` | Mistral | Available on the EU endpoint |
| `openai/text-embedding-3-small` | OpenAI | Embeddings (default) |

> The catalog changes frequently. Use `AIFactory.get_provider_models("edenai")`
> to discover what is currently available, optionally with
> `model_type="language"` or `model_type="embedding"` to query a single
> listing. Neither listing requires authentication.

## Features

### Streaming

```python
model = AIFactory.create_language("edenai", "openai/gpt-5.5")

for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "edenai", "openai/gpt-5.5",
    config={"structured": {"type": "json_object"}}
)
```

> Eden AI forwards `response_format` to the selected model, so support depends on
> that model rather than on the gateway.

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
    "edenai", "openai/gpt-5.5",
    config={"api_key": "your-key"}
)

# With custom base URL
model = AIFactory.create_language(
    "edenai", "openai/gpt-5.5",
    config={"base_url": "https://api.eu.edenai.run/v3"}
)
```

## Notes

- Eden AI exposes an OpenAI-compatible endpoint (`/chat/completions`, `/embeddings`, `/models`) under `https://api.edenai.run/v3`, so the standard Esperanto LLM features (streaming, tool calling, JSON mode) work with models that support them.
- Because Eden AI aggregates many vendors, feature support (JSON mode, tool calling, reasoning) depends on the specific model you select, not on Eden AI itself. Some upstream vendors do not report a context window, in which case `context_window` is `None`.
- Model ids carry the vendor as their first segment, so a fully qualified reference has two parts, `<vendor>/<model>`, and some vendors add their own path segments on top of that.
- `AIFactory.get_provider_models("edenai")` lists the catalog. The listing endpoint is public, so it works without a key, which makes it convenient to check availability before configuring anything.
