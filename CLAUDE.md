# Esperanto - Claude Agent Instructions

## Project Overview

Esperanto is a unified interface library for multiple AI model providers. The core value proposition is **consistency across all providers** - same interface, same behavior, regardless of the underlying provider.

**Packages:**
- `esperanto` - Core library (15+ LLM providers, embeddings, rerankers, TTS, STT)
- `brio_ext` - BrioDocs extensions (local model support, chat adapters, metrics)

## Critical Guidelines

### Consistency is Everything

When building or modifying providers:
1. **Always check base classes first** - Look at `src/esperanto/providers/{type}/base.py`
2. **Study sibling implementations** - Check 2-3 existing providers for patterns
3. **Normalize responses** - All providers must return standard `ChatCompletion`, `Message`, `Usage` types
4. **Follow naming conventions** - `{Provider}LanguageModel`, `{Provider}EmbeddingModel`, etc.

### Provider Architecture

**Registry Location:** `src/esperanto/factory.py`

```python
_provider_modules = {
    "language": {
        "openai": "esperanto.providers.llm.openai:OpenAILanguageModel",
        # ...
    }
}
```

**Adding a new provider:**
1. Create class in `src/esperanto/providers/{type}/{provider}.py`
2. Register in `_provider_modules` dictionary in `factory.py`
3. Write tests in `tests/providers/{type}/test_{provider}_provider.py`
4. Run tests: `uv run pytest -v`

**Removing a provider:**
1. Remove from `_provider_modules` in `factory.py`
2. Delete provider file
3. Delete tests

### Adapter System (brio_ext)

For local models needing chat templates:

**Registry Location:** `src/brio_ext/registry.py`

```python
ADAPTERS = (
    QwenAdapter(),
    LlamaAdapter(),
    # ...
)
```

**Adding an adapter:**
1. Create adapter in `src/brio_ext/adapters/{model}_adapter.py`
2. Add to `ADAPTERS` tuple in `registry.py`

**Removing an adapter:**
1. Remove from `ADAPTERS` tuple
2. Delete adapter file

### Key Files

| Purpose | Location |
|---------|----------|
| Main factory | `src/esperanto/factory.py` |
| BrioDocs factory | `src/brio_ext/factory.py` |
| Language base class | `src/esperanto/providers/llm/base.py` |
| Embedding base class | `src/esperanto/providers/embedding/base.py` |
| Response types | `src/esperanto/common_types/` |
| Adapter registry | `src/brio_ext/registry.py` |
| Metrics logger | `src/brio_ext/metrics/logger.py` |

### Configuration Priority

1. Direct constructor parameters
2. `config={}` dictionary
3. Environment variables
4. Default values

### HTTP Pattern

All providers use `httpx` (no vendor SDKs):

```python
def __post_init__(self):
    super().__post_init__()
    self.api_key = self.api_key or os.getenv("PROVIDER_API_KEY")
    self._create_http_clients()  # Creates self.client and self.async_client
```

## Testing

Always run tests after changes:

```bash
uv run pytest -v
```

For specific provider:

```bash
uv run pytest -v tests/providers/llm/test_{provider}_provider.py
```

## Documentation

- Developer guide: `docs/2025-12-20_Developer_Guide.md`
- Changelog entries: `docs/YYYY-MM-DD_*.md`

## Common Mistakes to Avoid

1. **Don't use vendor SDKs** - Use httpx directly
2. **Don't skip normalization** - Always convert to standard types
3. **Don't forget async** - Implement both `chat_complete()` and `achat_complete()`
4. **Don't hardcode timeouts** - Use `TimeoutMixin` pattern
5. **Don't mix brio_ext with esperanto** - Keep extensions separate from core
