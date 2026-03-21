# Architecture & Design Principles

This document defines the core design principles that guide Esperanto's development. **Please read this before contributing** — it will save you time and help us review your work faster.

## Core Principle: Provider Parity

Esperanto's entire value proposition is a **consistent, provider-agnostic interface**. Users can switch providers by changing one parameter, with identical code otherwise:

```python
# Same code, different provider — this is the promise
model = AIFactory.create_language("openai", "gpt-4o")
model = AIFactory.create_language("anthropic", "claude-sonnet-4-20250514")
model = AIFactory.create_language("google", "gemini-2.0-flash")

response = model.chat_complete(messages)  # identical API for all
```

This means:

1. **New features must work across all (or most) providers.** If a feature only makes sense for one provider, it probably doesn't belong in the public interface. Open an issue to discuss the cross-provider design before implementing.

2. **Interface consistency > feature count.** We'd rather ship a feature later with full provider support than ship it early for one provider. Partial implementations create an inconsistent API surface that breaks the core promise.

3. **Graceful degradation when needed.** Some providers genuinely can't support certain features. In those rare cases, raise a clear error — never silently ignore or return unexpected results. The user should always know what to expect.

## Provider Tiers

Not all providers require the same level of implementation effort. We classify them into tiers:

### First-class Providers

Providers with a fundamentally different API or SDK that requires unique implementation logic:

- **OpenAI** — the reference implementation
- **Anthropic** — different message format, tool format, content blocks
- **Google / Vertex AI** — parts-based format, GCP authentication
- **Azure** — deployment-based naming, Azure-specific auth
- **Ollama** — local execution, no auth, options dict
- **Mistral** — different tool format

These justify their own provider class with custom logic.

### Extended Providers

Providers that are OpenAI-compatible but add meaningful unique capabilities that require code:

- **Perplexity** — web search options, custom streaming behavior

### Profile-based Providers (OpenAI-Compatible)

Providers that are OpenAI-compatible with a different `base_url` and API key. These are implemented as **profiles** — declarative config objects — not Python classes:

- **DeepSeek** — just changes `base_url` and API key
- **xAI** — changes `base_url`, disables `response_format`, filters models to `grok-*`
- **DashScope (Qwen)** — Alibaba Cloud's OpenAI-compatible endpoint
- **MiniMax** — OpenAI-compatible endpoint

Profiles are defined in `src/esperanto/providers/llm/profiles.py` and resolved by the factory at runtime. Adding a new OpenAI-compatible provider is a **6-line config change**, not a new class:

```python
"minimax": OpenAICompatibleProfile(
    name="minimax",
    base_url="https://api.minimax.io/v1",
    api_key_env="MINIMAX_API_KEY",
    default_model="MiniMax-M2.5",
    owned_by="MiniMax",
    display_name="MiniMax",
),
```

Users can also register their own profiles at runtime:

```python
from esperanto import AIFactory, OpenAICompatibleProfile

AIFactory.register_openai_compatible_profile(
    OpenAICompatibleProfile(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        default_model="meta-llama/Llama-3-70b-chat-hf",
    )
)
model = AIFactory.create_language("together", "meta-llama/Llama-3-70b-chat-hf")
```

### Extended Providers (OpenAI-Compatible with Custom Code)

Providers that are OpenAI-compatible but add unique behavior that can't be expressed as config:

- **OpenRouter** — custom HTTP headers, selective `response_format` by model, custom HTTP request format
- **Perplexity** — web search parameters, custom streaming behavior

These keep their own Python classes because their customizations go beyond what a profile can express.

### When to Add a New Provider

Before implementing a new provider, ask yourself:

1. **Is it OpenAI-compatible?** If yes, add a profile in `profiles.py` — don't create a new class. This covers the vast majority of new provider requests.
2. **Does it need custom behavior beyond base_url/api_key/model filtering?** Custom headers, unique parameters, special error handling? If yes, it may need a class that extends `OpenAICompatibleLanguageModel`.
3. **Does it have a fundamentally different API?** Different message format, auth mechanism, or response structure? Then it needs a first-class provider class.
4. **Can it support the full interface?** A provider that can only do `chat_complete` but not streaming, tools, or structured output may not be ready for first-class support.

When in doubt, open an issue. We'd rather discuss the design upfront than review a PR that doesn't align with these principles.

## Architecture Overview

### Provider Pattern

All provider types follow the same structure:

```
Base class (abstract interface)
  -> Provider implementation (API integration)
    -> Factory registration (discoverability)
      -> Common response types (consistency)
```

### Configuration Priority

Three-tier configuration system (highest to lowest priority):

1. **Constructor args / config dict**: `config={"timeout": 120}`
2. **Environment variables**: `ESPERANTO_LLM_TIMEOUT=90`
3. **Provider defaults**

### Provider Composition

Providers inherit functionality via mixins:

- `TimeoutMixin` — configurable HTTP timeouts
- `SSLMixin` — configurable SSL verification
- Base class (e.g., `LanguageModel`) — provider-specific interface
- Provider implementation — actual API integration

### Response Types

All providers convert API-specific responses into Esperanto's common types:

| Type | Response |
|------|----------|
| Language | `ChatCompletion` / `ChatCompletionChunk` |
| Embedding | `List[List[float]]` |
| Reranker | `RerankResponse` |
| Speech-to-Text | `TranscriptionResponse` |
| Text-to-Speech | `AudioResponse` |

This normalization is what enables the provider-agnostic interface.

## Testing Philosophy

With 40+ provider implementations across 5 provider types, manual testing is impractical. We rely heavily on automated tests:

- **Every provider must have unit tests** that mock API responses and verify the Esperanto response format.
- **Every feature must be tested across all providers that support it.** A feature that works for OpenAI but breaks on Anthropic is a bug, not a partial implementation.
- **Test the interface, not the internals.** Tests should verify that `chat_complete()` returns the right `ChatCompletion` regardless of provider — not test provider-specific parsing logic in isolation.

Run the full test suite before submitting:

```bash
uv run pytest -v
```

## Key Design Decisions

### Why Not Just Wrap LangChain/LiteLLM?

Esperanto is a lightweight, focused library. We control the interface, the response types, and the provider implementations. This gives us:

- Predictable behavior across providers
- Minimal dependencies (only install SDKs for providers you use)
- First-class async support without adapter layers
- Direct control over streaming, tool calling, and structured output formats

### Why Optional Dependencies?

Each provider SDK is an optional dependency. Users only install what they need:

```bash
pip install esperanto[openai,anthropic]  # only these two
```

This keeps the base install small and avoids dependency conflicts between provider SDKs.
