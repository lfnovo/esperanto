# Esperanto

Unified interface library for working with multiple AI models (LLM, embedding, reranking, speech-to-text, text-to-speech) from different providers.

## Core Value Proposition

Esperanto provides a **consistent, provider-agnostic interface** for AI models. Users can switch providers by changing one parameter, with identical code otherwise.

**Key principle**: Consistency across providers is the main value proposition. When adding features, maintain interface uniformity.

## Project Structure

```
esperanto/
├── src/esperanto/
│   ├── __init__.py                 # Public API exports
│   ├── factory.py                  # AIFactory for creating provider instances
│   ├── model_discovery.py          # Static model discovery system
│   ├── providers/                  # Provider implementations
│   │   ├── llm/                    # Language model providers
│   │   ├── embedding/              # Embedding providers
│   │   ├── reranker/               # Reranker providers
│   │   ├── stt/                    # Speech-to-text providers
│   │   └── tts/                    # Text-to-speech providers
│   ├── common_types/               # Shared type definitions
│   └── utils/                      # Cross-cutting utilities
└── tests/                          # Test suite

See detailed documentation in subdirectory CLAUDE.md files.
```

## Module Documentation

- **[src/esperanto/providers/](src/esperanto/providers/CLAUDE.md)**: All provider implementations
  - **[llm/](src/esperanto/providers/llm/CLAUDE.md)**: Language models (OpenAI, Anthropic, Google, etc.)
  - **[embedding/](src/esperanto/providers/embedding/CLAUDE.md)**: Embedding models (OpenAI, Jina, Voyage, etc.)
  - **[reranker/](src/esperanto/providers/reranker/CLAUDE.md)**: Reranking models (Jina, Voyage, Transformers)
  - **[stt/](src/esperanto/providers/stt/CLAUDE.md)**: Speech-to-text (OpenAI, Groq, Google, Azure)
  - **[tts/](src/esperanto/providers/tts/CLAUDE.md)**: Text-to-speech (OpenAI, ElevenLabs, Google, Azure)
- **[src/esperanto/common_types/](src/esperanto/common_types/CLAUDE.md)**: Response types and models
- **[src/esperanto/utils/](src/esperanto/utils/CLAUDE.md)**: Timeout, SSL, caching utilities

## Key Files

### factory.py

Central factory for creating provider instances:

- `AIFactory.create_language()`: Create LLM provider
- `AIFactory.create_embedding()`: Create embedding provider
- `AIFactory.create_reranker()`: Create reranker provider
- `AIFactory.create_speech_to_text()`: Create STT provider
- `AIFactory.create_text_to_speech()`: Create TTS provider
- `AIFactory.get_provider_models()`: Static model discovery (no instance needed)

**Registration**: All providers registered in `_provider_modules` dict by type and name.

### model_discovery.py

Static model discovery system:

- `PROVIDER_MODELS_REGISTRY`: Maps provider names to discovery functions
- Discovery functions return `List[Model]` without creating provider instances
- Results cached with `ModelCache` (1 hour TTL)

### __init__.py

Public API surface:

- Exports `AIFactory` (primary interface)
- Exports base classes (`LanguageModel`, `EmbeddingModel`, etc.)
- Conditionally imports provider classes (handles missing dependencies)

## Architecture Patterns

### Provider Pattern

All providers follow consistent architecture:

1. **Base class** defines interface (abstract methods)
2. **Provider implementations** inherit and implement interface
3. **Factory registration** makes provider discoverable
4. **Common response types** ensure consistency

### Configuration Priority

Three-tier configuration system (highest to lowest):

1. **Config dict**: `config={"timeout": 120}`
2. **Environment variables**: `ESPERANTO_LLM_TIMEOUT=90`
3. **Defaults**: Provider type defaults

### Mixin Composition

Providers inherit functionality via mixins:

- `TimeoutMixin`: Configurable HTTP timeouts
- `SSLMixin`: Configurable SSL verification
- Base class (e.g., `LanguageModel`): Provider-specific interface
- Provider implementation: Actual API integration

## Adding a New Provider

When building a provider:

1. **Research existing implementations**: Check base class and 2-3 sibling providers for patterns
2. **Follow the interface exactly**: Consistency is critical for user experience
3. **Implement all abstract methods**: Don't skip required methods
4. **Register in factory**: Add to `factory._provider_modules["{type}"]`
5. **Add optional import**: In `src/esperanto/__init__.py` with try/except
6. **Write tests**: Use `uv run pytest -v` to verify

**Critical pattern** - `__post_init__()`:

```python
def __post_init__(self):
    super().__post_init__()  # ALWAYS call first
    self.api_key = self.api_key or os.getenv("PROVIDER_API_KEY")
    self.base_url = self.base_url or "https://api.provider.com/v1"
    self._create_http_clients()  # ALWAYS call last
```

### Steps for New Provider

1. Identify provider type (language, embedding, reranker, stt, tts)
2. Create `{provider_name}.py` in appropriate `src/esperanto/providers/{type}/` directory
3. Import base class from `.base`
4. Implement all abstract methods
5. Follow `__post_init__()` pattern above
6. Add to `factory._provider_modules["{type}"]["{provider}"]`
7. Add optional import to `src/esperanto/__init__.py`
8. Write tests in `tests/providers/{type}/test_{provider}.py`
9. Run tests: `uv run pytest -v`
10. Add docs in `docs/providers/{provider}.md`

## Integration Points

### AIFactory ↔ Providers

Factory imports providers dynamically via `_import_provider_class()`:

- Avoids loading all providers at import time
- Handles missing dependencies gracefully
- Raises helpful errors for missing packages

### Providers ↔ Common Types

All providers convert API responses to Esperanto's common types:

- Language: `ChatCompletion` / `ChatCompletionChunk`
- Language (tools): `Tool`, `ToolFunction`, `ToolCall`, `FunctionCall`
- Embedding: `List[List[float]]`
- Reranker: `RerankResponse`
- STT: `TranscriptionResponse`
- TTS: `AudioResponse`

### Tool Calling

Esperanto provides unified tool/function calling across all LLM providers:

```python
from esperanto import AIFactory
from esperanto.common_types import Tool, ToolFunction

# Define tools once - works with any provider
tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        )
    )
]

# Use with any provider - identical code
model = AIFactory.create_language("openai", "gpt-4o")  # or "anthropic", "google", etc.
response = model.chat_complete(messages, tools=tools)

# Tool calls in response
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"{tc.function.name}: {tc.function.arguments}")
```

See [docs/features/tool-calling.md](docs/features/tool-calling.md) for full documentation.

### Providers ↔ Utils

All providers use utility mixins:

- `TimeoutMixin._get_timeout()` for HTTP timeout configuration
- `SSLMixin._get_ssl_verify()` for SSL verification settings
- `ModelCache` for caching model lists (via model_discovery)

## Gotchas

### Adding Providers

- **Consistency is key**: Look at existing providers before implementing
- **Base class inspection**: Always check the base class for the provider type
- **Super call order**: `super().__post_init__()` must be called **first**
- **Client creation timing**: `_create_http_clients()` must be called **last** (needs api_key, base_url)
- **Factory registration**: Provider won't work until added to `factory._provider_modules`
- **Optional dependencies**: Don't make Esperanto depend on all provider SDKs - handle ImportError

### Interface Consistency

- **Method signatures**: Must match base class exactly (don't add required params)
- **Return types**: Must use common types (don't return provider-specific objects)
- **Error handling**: Raise `RuntimeError` for API errors, `ValueError` for validation
- **Response normalization**: Always convert provider responses to Esperanto types

### Testing

- **Test after writing**: `uv run pytest -v` to verify functionality
- **Check all providers**: Changes to base classes affect all providers
- **Integration tests**: Test provider switching (same code, different provider)

### API Keys

- **Environment variables**: Follow pattern `{PROVIDER}_API_KEY` (all caps)
- **Validation**: Always check for None in `__post_init__` and raise helpful ValueError
- **Security**: Never log API keys or include in error messages

### Documentation

- **User docs**: Update `docs/providers/{provider}.md` for human users
- **AI docs**: Keep CLAUDE.md files updated for AI context (this file structure)
- **Consistency**: Documentation should reflect actual implementation

## Development Workflow

1. **Before implementing**: Read relevant base class + 2-3 provider examples
2. **During implementation**: Follow patterns exactly, check tests frequently
3. **After implementation**: Run full test suite, update docs
4. **Before committing**: Ensure tests pass, check consistency with sibling providers

## Common Commands

- **Run all tests**: `uv run pytest -v`
- **Run specific test**: `uv run pytest tests/providers/llm/test_openai.py -v`
- **Run integration tests**: `uv run pytest tests/integration/ -v`
- **Check types**: `uv run mypy src/esperanto`
- **Format code**: `uv run black src/ tests/`

## For Automated Agents

If you are an automated coding agent (harny, Claude Code in headless mode, etc.) running against this repo, read this section before deciding what to validate.

### Validator command

The single command to confirm a change is acceptable:

```bash
uv sync --all-extras && uv run pytest tests/providers tests/unit tests/common_types -q --no-cov
```

This runs ~895 tests (mocked, no real API calls) in roughly 70 seconds. Pass = exit 0. The same scope is gated in CI via `.github/workflows/test.yml`.

For a stricter local check that mirrors CI, also run:

```bash
uv run ruff check .
uv run mypy src/esperanto
```

Both are clean on `main` and gated on every PR by `.github/workflows/lint.yml`. If you introduce a new ruff or mypy error, fix it before opening the PR.

### Known-broken tests (out of validator scope)

A few top-level test files have pre-existing failures unrelated to typical changes:

- `tests/test_deprecation_warnings.py` — several tests construct the warning context but never trigger the deprecated property inside it, so the assertions always fail.

These files are intentionally excluded from the validator command. Don't attempt to fix them as a side-effect of unrelated work.

### Integration tests

`tests/integration/` requires real provider API keys and external network. Always exclude from automated runs unless the user has explicitly set up credentials and asked for it.

### Other notes

- The `notebooks/` directory is local-only (gitignored). If you see modifications there, leave them alone — they aren't part of the project.
- Do not commit `.env`, `google-credentials.json`, or any other credential file. The `.gitignore` covers the common cases but always double-check before staging.

## Critical Principles

See @ARCHITECTURE.md for the full design principles. The key rules:

1. **Provider Parity**: New features MUST work across all (or most) providers. Partial implementations are not acceptable — they break the core promise of a provider-agnostic interface.
2. **Consistency > Features**: If a feature can't be consistent across providers, reconsider. We'd rather ship later with full support than early with partial support.
3. **Interface First**: Design interfaces before implementing providers.
4. **Provider Tiers**: Not every provider needs its own class. OpenAI-compatible providers should use `OpenAICompatibleLanguageModel` unless they have fundamentally different APIs. See ARCHITECTURE.md for tier definitions.
5. **Test Driven**: Write tests as you implement, run frequently. Every feature must be tested across all affected providers.
6. **Documentation**: Keep both human and AI docs in sync with code.
