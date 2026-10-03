# OpenRouter

## Overview

OpenRouter provides unified access to models from many providers through a single API. It acts as a gateway to models from OpenAI, Anthropic, Google, Meta, Mistral and many others, offering flexibility and easy model switching.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Hundreds of models from many providers |
| Embeddings | ✅ | OpenAI-compatible embeddings (`/embeddings`) |
| Reranking | ❌ | Not available |
| Speech-to-Text | ✅ | OpenAI-compatible transcription (`/audio/transcriptions`) |
| Text-to-Speech | ✅ | OpenAI-compatible speech (`/audio/speech`) |

**Official Documentation:** https://openrouter.ai/docs

## Prerequisites

### Account Requirements
- OpenRouter account (sign up at https://openrouter.ai)
- API key with credits or payment method

### Getting API Keys
1. Visit https://openrouter.ai/keys
2. Click "Create Key"
3. Copy and store the key securely

## Environment Variables

```bash
# OpenRouter API key (required)
OPENROUTER_API_KEY="sk-or-v1-..."

# OpenRouter base URL (optional, defaults to https://openrouter.ai/api/v1)
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`, `base_url="..."`)
2. Environment variables (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`)
3. Default base URL (`https://openrouter.ai/api/v1`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Create OpenRouter model
# You can use any model available on OpenRouter
model = AIFactory.create_language("openrouter", "anthropic/claude-sonnet-5.5")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Direct Instantiation

```python
from esperanto.providers.llm.openrouter import OpenRouterLanguageModel

# Create model instance
model = OpenRouterLanguageModel(
    api_key="your-api-key",
    model_name="anthropic/claude-sonnet-5.5"
)

# Use the model
messages = [{"role": "user", "content": "Hello!"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Capabilities

### Language Models (LLM)

**Available Model Categories:**

OpenRouter provides access to hundreds of models, and the lineup changes often. Some popular choices (checked October 2026; see https://openrouter.ai/models for the current list):

**OpenAI Models:**
- `openai/gpt-5.5` - Latest GPT
- `openai/gpt-5.4-mini` - Fast and cost-effective
- `openai/gpt-4o` - Widely used multimodal model

**Anthropic Models:**
- `anthropic/claude-opus-5.5` - Most capable Claude for everyday use
- `anthropic/claude-sonnet-5.5` - Balanced Claude
- `anthropic/claude-haiku-4.5` - Fast Claude

**Google Models:**
- `google/gemini-3.8-flash` - Latest Gemini Flash
- `google/gemini-2.5-pro` - Stable Gemini Pro

**Meta Models:**
- `meta-llama/llama-4-maverick` - Largest Llama 4
- `meta-llama/llama-4-scout` - Efficient Llama 4
- `meta-llama/llama-3.3-70b-instruct` - Llama 3.3

**Mistral Models:**
- `mistralai/mistral-large-2512` - Most capable Mistral
- `mistralai/mistral-small-2603` - Fast Mistral
- `mistralai/codestral-2508` - Code specialist

**Other Popular Models:**
- `perplexity/sonar-pro` - With web search
- `deepseek/deepseek-v4-flash` - Cost-effective
- `qwen/qwen3-235b-a22b-2507` - Multilingual

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "openrouter",
    "anthropic/claude-sonnet-5.5",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 2.0)
        "max_tokens": 1000,           # Maximum response length
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode (model-dependent)
        "timeout": 60.0               # Request timeout
    }
)
```

Supported parameters vary by model; OpenRouter lists them on each model's page
(`supported_parameters` in the model API).

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create OpenRouter model
model = AIFactory.create_language("openrouter", "anthropic/claude-sonnet-5.5")

# Simple chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Switch Models Easily:**

```python
# Try different models with same code
models_to_try = [
    "anthropic/claude-sonnet-5.5",
    "openai/gpt-5.5",
    "google/gemini-3.8-flash",
    "meta-llama/llama-4-maverick"
]

messages = [{"role": "user", "content": "Explain machine learning in simple terms"}]

for model_name in models_to_try:
    model = AIFactory.create_language("openrouter", model_name)
    response = model.chat_complete(messages)
    print(f"\n{model_name}:")
    print(response.choices[0].message.content[:200] + "...")
```

**Example - Streaming:**

```python
model = AIFactory.create_language("openrouter", "anthropic/claude-sonnet-5.5")

messages = [{"role": "user", "content": "Write a short story about AI"}]

# Synchronous streaming
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in await model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Example - JSON Mode:**

```python
# Note: JSON mode support depends on the specific model
model = AIFactory.create_language(
    "openrouter",
    "openai/gpt-5.5",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three programming languages as JSON"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Schema-Driven Structured Output (Model-Dependent):**

```python
from pydantic import BaseModel

class CapitalResponse(BaseModel):
    capital: str

model = AIFactory.create_language(
    "openrouter",
    "openai/gpt-5.5",
    config={
        "structured": {
            "type": "json_schema",
            "schema": CapitalResponse,   # or JSON Schema dict
            "name": "capital_response",  # optional
            "strict": True               # optional
        }
    }
)

response = model.chat_complete(
    [{"role": "user", "content": "Return one European capital"}]
)

print(response.content)      # Raw JSON string
print(response.structured)   # Parsed/validated CapitalResponse
```

Notes:
- Schema mode support depends on the selected OpenRouter model/provider.
- Esperanto passes schema format through and surfaces upstream incompatibility errors directly (fail-fast, no silent downgrade).
- Schema mode is non-streaming in Esperanto v1 (`stream=True` raises `ValueError`).
- In both JSON and schema mode, a response with empty content and no tool calls raises `EmptyCompletionError`.

**Example - Free Models:**

```python
# OpenRouter offers some free models (ids ending in :free)
free_model = AIFactory.create_language("openrouter", "google/gemma-4-31b-it:free")

messages = [{"role": "user", "content": "Hello!"}]
response = free_model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Code Generation:**

```python
# Use a code-specialized model
code_model = AIFactory.create_language("openrouter", "mistralai/codestral-2508")

messages = [{
    "role": "user",
    "content": "Write a Python function to implement quicksort"
}]

response = code_model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Async Chat:**

```python
async def chat_async():
    model = AIFactory.create_language("openrouter", "anthropic/claude-sonnet-5.5")

    messages = [{"role": "user", "content": "Explain quantum computing"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Run async
# await chat_async()
```

**Example - Multi-turn Conversation:**

```python
# Build conversation with context
model = AIFactory.create_language("openrouter", "openai/gpt-5.5")

messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."},
    {"role": "user", "content": "What are its main advantages?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Temperature Control:**

```python
# More creative (higher temperature)
creative_model = AIFactory.create_language(
    "openrouter",
    "google/gemini-3.8-flash",
    config={"temperature": 1.2, "max_tokens": 1024}
)

# More focused (lower temperature)
focused_model = AIFactory.create_language(
    "openrouter",
    "google/gemini-3.8-flash",
    config={"temperature": 0.3, "max_tokens": 1024}
)
```

### Embeddings

OpenRouter exposes an OpenAI-compatible embeddings endpoint (`POST /api/v1/embeddings`).
The default model is `openai/text-embedding-3-small`.

**Popular models:**
- `openai/text-embedding-3-small` (default), `openai/text-embedding-3-large`
- `google/gemini-embedding-001`
- `voyageai/voyage-4`, `voyageai/voyage-code-4`
- `qwen/qwen3-embedding-8b`
- `mistralai/mistral-embed-2312`

```python
from esperanto.factory import AIFactory

embedder = AIFactory.create_embedding("openrouter", "openai/text-embedding-3-small")

vectors = embedder.embed(["Hello world", "Esperanto makes provider swaps easy"])
print(len(vectors), len(vectors[0]))

# Async
vectors = await embedder.aembed(["Hello world"])
```

Large inputs are split automatically into requests of up to 96 texts each and
the results are returned in input order. Set `config={"embed_batch_size": N}` to
use smaller batches.

### Text-to-Speech (TTS)

OpenRouter exposes an OpenAI-compatible speech endpoint (`POST /api/v1/audio/speech`)
that accepts `model`, `input`, `voice`, and `response_format`. Model names follow
OpenRouter's `vendor/model` convention.

**Available models** — OpenRouter lists dedicated speech models under its
`?output_modalities=speech` filter (they do **not** appear in the unfiltered
`/models` list that `AIFactory.get_provider_models("openrouter")` returns). Some choices:
- `microsoft/mai-voice-2` (default), `microsoft/mai-voice-2.1`, `microsoft/mai-voice-2.1-flash`
- `google/gemini-3.8-flash-tts`, `google/gemini-3.1-flash-tts-preview`
- `x-ai/grok-voice-tts-1.0`
- `mistralai/voxtral-mini-tts-2603`
- `deepgram/aura-2`, `minimax/speech-2.8-hd`, `fish-audio/s2.1-pro`
- `hexgrad/kokoro-82m`, `sesame/csm-1b`, `canopylabs/orpheus-3b-0.1-ft`

> **Voices are model-specific.** There is currently no OpenAI TTS model on
> OpenRouter, so OpenAI's `alloy`/`nova` voice names do **not** apply. The default
> `microsoft/mai-voice-2` uses Microsoft neural voice names (e.g.
> `en-US-AvaNeural`, `en-US-AndrewNeural`). When picking a different model, pass a
> voice listed on that model's page. `response_format` supports `mp3` (default)
> and `pcm` only.

```python
from esperanto.factory import AIFactory

# Zero-config: default model (microsoft/mai-voice-2) + default voice (en-US-AvaNeural)
tts = AIFactory.create_text_to_speech("openrouter")
response = tts.generate_speech("Hello from OpenRouter")
with open("speech.mp3", "wb") as f:
    f.write(response.audio_data)

# Explicit model + a voice that model supports
tts = AIFactory.create_text_to_speech("openrouter", "microsoft/mai-voice-2")
response = tts.generate_speech("Hello", voice="en-US-AndrewNeural")

# Async
async def synth():
    return (await tts.agenerate_speech("Hello", voice="en-US-EmmaNeural")).audio_data
```

### Speech-to-Text (STT)

OpenRouter's transcription endpoint (`POST /api/v1/audio/transcriptions`) accepts a
JSON body with base64-encoded audio (not OpenAI's multipart upload). Esperanto handles
this encoding for you — pass a file path or a binary stream exactly like other providers.

**Popular models** (OpenRouter lists them under `?output_modalities=transcription`):
- `openai/whisper-1` (default), `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo`
- `openai/gpt-4o-transcribe`, `openai/gpt-4o-mini-transcribe`
- `deepgram/nova-3`, `mistralai/voxtral-mini-transcribe`, `google/gemini-3.5-transcribe`

```python
from esperanto.factory import AIFactory

stt = AIFactory.create_speech_to_text("openrouter", "openai/whisper-1")

# From a file path
result = stt.transcribe("audio.mp3", language="en")
print(result.text)

# From a binary stream
with open("audio.mp3", "rb") as f:
    result = stt.transcribe(f)

# Usage stats (when reported by OpenRouter)
if result.usage:
    print(result.usage.input_seconds, result.usage.total_tokens)
```

The audio codec is inferred from the file extension (wav, mp3, flac, m4a, ogg, webm, aac).
OpenRouter's transcription endpoint does not document a `prompt` parameter, so `prompt`
is accepted for interface parity but not sent, and segment-level timestamps are not
returned (`segments` stays `None`).

## Advanced Features

### Model Discovery

Browse available models at https://openrouter.ai/models, or list them from Esperanto:

```python
from esperanto.factory import AIFactory

models = AIFactory.get_provider_models("openrouter", api_key="your-api-key")
for model in models[:10]:  # Show first 10
    print(model.id)
```

The result is cached for an hour. It lists chat models only; speech and
transcription models are listed on OpenRouter under their own filters (see above).

### Free Models

OpenRouter offers free access to some models. Free ids end in `:free`, and the
set changes often; filter https://openrouter.ai/models by price to see the
current ones. Free models have stricter rate limits.

```python
free_models = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3.8-27b:free",
]

model = AIFactory.create_language("openrouter", free_models[0])
```

### Cost Optimization

Choose models based on your budget:

```python
# Highest quality
premium_model = AIFactory.create_language("openrouter", "anthropic/claude-opus-5.5")

# Balanced cost/performance
balanced_model = AIFactory.create_language("openrouter", "openai/gpt-5.4-mini")

# Budget-friendly
budget_model = AIFactory.create_language("openrouter", "deepseek/deepseek-v4-flash")
```

### Timeout Configuration

Customize request timeouts:

```python
# Extended timeout for complex tasks
model = AIFactory.create_language(
    "openrouter",
    "anthropic/claude-opus-5.5",
    config={
        "timeout": 120.0,    # 2 minutes
        "max_tokens": 4096
    }
)
```

### LangChain Integration

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("openrouter", "anthropic/claude-sonnet-5.5")
langchain_model = model.to_langchain()  # a ChatOpenAI pointed at OpenRouter

# Use with LangChain
response = langchain_model.invoke("Explain quantum computing in one sentence")
print(response.content)
```

`to_langchain()` requires `langchain-openai` (`pip install langchain-openai`).

## Model Selection Guide

### For Quality
**Best:** Claude Opus 5.5, GPT-5.5, Gemini 2.5 Pro
```python
model = AIFactory.create_language("openrouter", "anthropic/claude-opus-5.5")
```

### For Speed
**Best:** Gemini Flash, Claude Haiku 4.5, GPT-5.4 mini
```python
model = AIFactory.create_language("openrouter", "google/gemini-3.8-flash")
```

### For Coding
**Best:** Claude Sonnet 5.5, GPT-5.5, Codestral
```python
model = AIFactory.create_language("openrouter", "mistralai/codestral-2508")
```

### For Cost
**Best:** Free models, DeepSeek V4 Flash, GPT-5.4 mini
```python
model = AIFactory.create_language("openrouter", "deepseek/deepseek-v4-flash")
```

### For Long Context
**Best:** Claude Sonnet 5.5, GPT-5.5, Gemini (about 1M tokens each)
```python
model = AIFactory.create_language("openrouter", "google/gemini-2.5-pro")
```

### For Multilingual
**Best:** Qwen 3, Mistral models, Gemini
```python
model = AIFactory.create_language("openrouter", "qwen/qwen3-235b-a22b-2507")
```

## Use Cases

### When to Choose OpenRouter

**Perfect for:**
- Model comparison and benchmarking
- Flexibility to switch providers easily
- Access to models not directly available
- Fallback strategies (try multiple models)
- Cost optimization across providers
- Single API for multiple providers
- Avoiding vendor lock-in

**Consider alternatives if:**
- Using only one provider consistently
- Need provider-specific features
- Want direct provider billing
- Require lowest possible latency

### Common Applications

**1. Model Comparison:**
```python
def compare_models(question, models):
    results = {}
    for model_name in models:
        model = AIFactory.create_language("openrouter", model_name)
        messages = [{"role": "user", "content": question}]
        response = model.chat_complete(messages)
        results[model_name] = response.choices[0].message.content
    return results

models = [
    "anthropic/claude-sonnet-5.5",
    "openai/gpt-5.5",
    "google/gemini-3.8-flash"
]

results = compare_models("Explain quantum computing", models)
```

**2. Fallback Strategy:**
```python
async def chat_with_fallback(messages):
    # Try models in order of preference
    models = [
        "anthropic/claude-sonnet-5.5",
        "openai/gpt-5.5",
        "meta-llama/llama-4-maverick"
    ]

    for model_name in models:
        try:
            model = AIFactory.create_language("openrouter", model_name)
            response = await model.achat_complete(messages)
            return response
        except Exception as e:
            print(f"Failed with {model_name}: {e}")
            continue

    raise Exception("All models failed")
```

**3. Cost-Optimized Pipeline:**
```python
# Use cheap model for simple tasks, premium for complex
def smart_completion(question, complexity="low"):
    if complexity == "low":
        model = AIFactory.create_language("openrouter", "deepseek/deepseek-v4-flash")
    elif complexity == "medium":
        model = AIFactory.create_language("openrouter", "openai/gpt-5.4-mini")
    else:
        model = AIFactory.create_language("openrouter", "anthropic/claude-opus-5.5")

    messages = [{"role": "user", "content": question}]
    return model.chat_complete(messages)
```

**4. Specialized Tasks:**
```python
# Use best model for each task type
def get_specialized_model(task_type):
    models = {
        "code": "mistralai/codestral-2508",
        "creative": "anthropic/claude-sonnet-5.5",
        "analysis": "openai/gpt-5.5",
        "chat": "meta-llama/llama-4-maverick"
    }
    return AIFactory.create_language("openrouter", models[task_type])

code_model = get_specialized_model("code")
creative_model = get_specialized_model("creative")
```

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key at https://openrouter.ai/keys

**Insufficient Credits:**
```
Error: Insufficient credits
```
**Solution:** Add credits at https://openrouter.ai/settings/credits

**Model Not Available:**
```
Error: Model not found
```
**Solution:**
- Check the model ID at https://openrouter.ai/models (models are retired regularly)
- Ensure correct format: `vendor/model-name`

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic or add credits (free models have stricter limits)

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase timeout:
```python
config={"timeout": 120.0}
```

### Best Practices

1. **Use Full Model IDs:** Always include the vendor prefix (e.g., `anthropic/claude-sonnet-5.5`)

2. **Monitor Costs:** Different models have different pricing - check https://openrouter.ai/models

3. **Free Models:** Ids ending in `:free` are free but rate-limited, and the set changes often

4. **Model Selection:** Choose based on your specific needs (quality, speed, cost)

5. **Fallback Strategy:** Implement fallbacks for production applications

6. **Check Capabilities:** Not all models support all features (JSON mode, tool calling, sampling parameters)

7. **Credits:** Keep credits topped up for uninterrupted service

## Performance Characteristics

### Response Times
Varies by model, provider load and output length. Small and "flash"/"mini"
models respond fastest; large reasoning models take longer, especially when
they think before answering.

### Context Windows
Varies by model, from a few thousand tokens on small open models to about 1M
tokens on current Claude, GPT and Gemini models. OpenRouter reports each
model's `context_length` on its model page and in the model API.

### Pricing
Check current pricing at https://openrouter.ai/models
- Ranges from free to premium
- Pay only for what you use
- No subscription required

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [Embedding Guide](../capabilities/embedding.md)
- [OpenAI Provider](./openai.md)
- [Anthropic Provider](./anthropic.md)
- [Google Provider](./google.md)
- [Mistral Provider](./mistral.md)
