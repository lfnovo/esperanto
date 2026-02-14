# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.19.2] - 2026-02-14

### Fixed

- **TTS Factory Config Support** - Aligned `create_text_to_speech()` signature with other factory methods by adding `config` dict parameter. Previously, `config={"api_key": "..."}` was silently ignored, causing authentication failures for callers using the standard config pattern.

### Deprecated

- Direct `api_key` and `base_url` parameters on `create_text_to_speech()` — use `config={"api_key": "...", "base_url": "..."}` instead.
- Removed `api_key` parameter from `create_tts()` deprecated alias.

## [2.19.1] - 2026-02-12

### Fixed

- **Config Dict API Key Resolution in Embedding Providers** - Fixed embedding providers ignoring `api_key` passed via config dict
  - Affected providers: OpenAI, Google, Jina, Voyage (embedding)
  - Providers now correctly honor API keys resolved from config before env fallback
  - Added regression tests to ensure `config={"api_key": "..."}` works consistently

## [2.19.0] - 2026-02-07

### Added

- **Vertex AI Credentials Support** - Added `credentials_file` parameter and full credentials priority chain
  - Supports explicit service account JSON file via `credentials_file` parameter
  - Supports `GOOGLE_APPLICATION_CREDENTIALS` environment variable
  - Supports Application Default Credentials (ADC) via `google.auth.default()`
  - Falls back to `gcloud` CLI when `google-auth` is not installed

### Changed

- **Vertex AI LangChain Migration** - Migrated from deprecated `ChatVertexAI` to `ChatGoogleGenerativeAI`
  - Uses `langchain-google-genai` (already a dependency) instead of `langchain-google-vertexai`
  - Removed `langchain-google-vertexai` from dev dependencies
  - LangChain model instances are now cached (matching Google provider pattern)

- **Proxy Configuration** - Simplified proxy handling by delegating entirely to httpx
  - Esperanto now uses standard environment variables: `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`
  - **BREAKING**: Removed `ESPERANTO_PROXY` environment variable support
  - **BREAKING**: Removed `config={"proxy": "..."}` parameter support
  - Migration: Replace `ESPERANTO_PROXY` with `HTTP_PROXY` and `HTTPS_PROXY`

## [2.17.2] - 2026-01-24

### Fixed

- **LangChain Connection Error from Garbage Collection** - Fixed "Connection error" when using `to_langchain()` (#73)
  - When Esperanto model was garbage collected, shared httpx clients were closed, breaking LangChain
  - Now creates fresh httpx clients for LangChain with same configuration (timeout, SSL, proxy)
  - Affected providers: OpenAI, Groq, Perplexity, OpenAI-compatible, Azure
  - Fixes: lfnovo/open-notebook#460

## [2.17.1] - 2026-01-24

### Fixed

- **Config Dict API Key Not Unpacked** - Fixed providers ignoring `api_key` passed via config dict (#68)
  - Affected providers: OpenRouter, DeepSeek, xAI (LLM), Groq (STT)
  - These providers inherit from OpenAI-compatible parent classes and were checking for `api_key` before the config dict was unpacked
  - Now correctly extracts `api_key` and `base_url` from config dict before setting provider defaults
  - Example that now works:
    ```python
    model = AIFactory.create_language(
        "openrouter",
        "anthropic/claude-3.5-sonnet",
        config={"api_key": "sk-or-v1-xxxxx"}
    )
    ```

## [2.17.0] - 2026-01-23

### Added

- **Unified Tool Calling** - Added tool/function calling support across all LLM providers (#67)
  - Define tools once using `Tool` and `ToolFunction` types, use with any provider
  - Consistent interface: `chat_complete(messages, tools=tools)`
  - Support for `tool_choice` parameter: `"auto"`, `"required"`, `"none"`, or specific tool
  - Support for `parallel_tool_calls` parameter
  - Multi-turn conversations with tool results (`role="tool"` messages)
  - Tool call validation with `validate_tool_calls=True` parameter
  - New types: `Tool`, `ToolFunction`, `ToolCall`, `FunctionCall`, `ToolCallValidationError`
  - Validation utilities: `validate_tool_call()`, `validate_tool_calls()`, `find_tool_by_name()`
  - Tested providers: OpenAI, Anthropic, Google, Groq, Mistral, DeepSeek, xAI, OpenRouter, Azure, Ollama
  - Full documentation at `docs/features/tool-calling.md`
  - Examples at `examples/tool_calling/`

- **Real Integration Tests for Tool Calling** - Added tests that call actual APIs (#71)
  - Validates tool calling works correctly across 10 providers
  - Tests both basic tool calls and multi-turn conversations
  - Perplexity skipped (doesn't support tool calling)

### Fixed

- **Streaming Validation Warning** - Added warning when `validate_tool_calls=True` is used with streaming (#71)
  - Tool call validation requires the complete response
  - Now emits `UserWarning` instead of silently ignoring the parameter
  - Affects all providers consistently

### Changed

- Moved mocked tool calling tests from `tests/integration/` to `tests/unit/`

## [2.16.0] - 2026-01-21

### Added

- **Ollama Context Window Configuration** - Added `num_ctx` support for Ollama provider
  - Default context window increased to 128,000 tokens (Ollama's default of 2,048 was causing context truncation)
  - Configurable via `config={"num_ctx": 32768}`
  - Passed to LangChain's ChatOllama via `to_langchain()`

- **Ollama Keep Alive Configuration** - Added `keep_alive` support for Ollama provider
  - Controls how long models stay loaded in memory
  - No default set (doesn't force memory usage on users)
  - Examples: `"5m"` (5 minutes), `"0"` (unload immediately), `"-1"` (keep indefinitely)
  - Configurable via `config={"keep_alive": "10m"}`

## [2.15.0] - 2026-01-16

### Added

- **Message Thinking Properties** - Added `thinking` and `cleaned_content` properties to `Message` class
  - `thinking`: Extracts content inside `<think>` tags (reasoning trace from models like Qwen3, DeepSeek R1)
  - `cleaned_content`: Returns content with `<think>` tags removed (actual response)
  - Multiple `<think>` blocks are concatenated
  - Returns `None` for `thinking` if no tags present

## [2.14.1] - 2026-01-16

### Fixed

- **LM Studio Compatibility** - Fixed `response_format` parameter rejection by LM Studio (#46)
  - LM Studio only supports `json_schema` response format, not `json_object`
  - Added automatic detection for LM Studio (port 1234 heuristic)
  - Added graceful degradation: retries without `response_format` if endpoint rejects it
  - Affects both direct API calls and LangChain integration
  - See also: [lmstudio-ai/lmstudio-bug-tracker#189](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/189)

## [2.14.0] - 2026-01-16

### Added

- **Proxy Configuration** - Added HTTP proxy support for all provider connections
  - Configuration via environment variable: `ESPERANTO_PROXY`
  - Configuration via config dict: `config={"proxy": "http://proxy:8080"}`
  - Supports HTTP, HTTPS, and authenticated proxies
  - Priority order: config dict > environment variable > none
  - Example:
    ```python
    # Via environment variable
    export ESPERANTO_PROXY="http://proxy.example.com:8080"

    # Via config dict
    model = AIFactory.create_language(
        "openai", "gpt-4",
        config={"proxy": "http://proxy.example.com:8080"}
    )
    ```

### Changed

- Updated dependencies to latest versions
- General codebase cleanup and maintenance

### Fixed

- **HTTP Client Resource Management** - Fixed async client not being properly closed when deleting model instances (#60)
  - Added `HttpConnectionMixin` consolidating timeout, SSL, and client lifecycle management
  - Providers now support context managers (`with model:` and `async with model:`)
  - Added explicit `close()` and `aclose()` methods for resource cleanup
  - Destructor now properly cleans up sync clients

## [2.13.0] - 2026-01-04

### Changed

- **Dependency Updates** - Updated all dependencies to latest major versions (#55)

### Fixed

- Fixed CONTRIBUTING.md to use correct `uv sync` command for PEP735 dependency groups (#54)
- Fixed unused `openai` import that could cause errors in tests (#53)

## [2.12.1] - 2025-12-15

### Fixed

- **Ollama LangChain JSON Format** - Fixed `OllamaLanguageModel.to_langchain()` to correctly pass `format="json"` when `structured={"type": "json"}` is configured (#49)

## [2.12.0] - 2025-12-14

### Fixed

- **SSL Configuration for LangChain** - Extended SSL verification configuration to all LangChain integrations (#47)
  - Anthropic, Groq, Mistral, and Ollama providers now pass SSL settings to LangChain

## [2.11.0] - 2025-12-14

### Added

- **SSL Verification Configuration** - Added ability to disable SSL verification or use custom CA bundles (#45)
  - See [2.9.1] for full feature documentation (feature was backported)

## [2.10.0] - 2025-11-27

### Added

- **OpenRouter Embedding Provider** - Access 60+ embedding providers through OpenRouter's unified API (#44)
  - Inherits from `OpenAIEmbeddingModel` following existing patterns
  - Supports `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` environment variables
  - Model discovery filters embedding models from OpenRouter's `/models` endpoint
  - Example:
    ```python
    model = AIFactory.create_embedding(
        "openrouter",
        "openai/text-embedding-3-small"
    )
    embeddings = model.embed(["Hello", "World"])
    ```

- **Google Speech-to-Text** - Added STT support using Gemini API's audio transcription (#43)
  - Supports `gemini-2.5-flash` and `gemini-2.0-flash` models
  - Base64 audio encoding with automatic MIME type detection
  - Supports MP3, WAV, AIFF, AAC, OGG, FLAC formats
  - Language hints and custom prompts for improved accuracy
  - Example:
    ```python
    model = AIFactory.create_speech_to_text("google", "gemini-2.5-flash")
    response = model.transcribe("audio.mp3", language="en")
    ```

### Fixed

- **Anthropic Temperature/Top-p Conflict** - Fixed issue where passing both `temperature` and `top_p` to Anthropic LangChain integration caused errors (#42, #39)
  - Now prioritizes `temperature` over `top_p` when both are set

## [2.9.2] - 2025-11-11

### Fixed

- **Ollama Embeddings API** - Updated to use newer `/api/embed` endpoint for compatibility with recent Ollama versions (#37)

## [2.9.1] - 2025-11-27

### Added

- **SSL Verification Configuration** - Added ability to disable SSL verification or use custom CA bundles for local providers with self-signed certificates (Ollama, LM Studio, etc.)
  - Configuration priority hierarchy: config dict > environment variables > default (True)
  - Config parameter `verify_ssl` (boolean) to disable SSL verification
  - Config parameter `ssl_ca_bundle` (string path) for custom CA certificates
  - Environment variables `ESPERANTO_SSL_VERIFY` and `ESPERANTO_SSL_CA_BUNDLE`
  - Security warning emitted when SSL verification is disabled
  - Type validation for `verify_ssl` accepts booleans, integers, and common string representations ("true", "false", "yes", "no", "0", "1")
  - Available across all provider types: LLM, Embedding, STT, TTS, Reranker
  - Example:
    ```python
    # Disable SSL verification (development only)
    model = AIFactory.create_language(
        "ollama",
        "llama3",
        config={"verify_ssl": False}
    )

    # Use custom CA bundle (recommended for self-signed certs)
    model = AIFactory.create_language(
        "ollama",
        "llama3",
        config={"ssl_ca_bundle": "/path/to/ca-bundle.pem"}
    )
    ```

## [2.8.0] - 2025-10-25

### Added

- **Azure OpenAI Speech-to-Text Support** - Added Whisper model support via Azure deployments
  - Direct HTTP implementation using httpx (no SDK dependencies)
  - Modality-specific environment variables: `AZURE_OPENAI_API_KEY_STT`, `AZURE_OPENAI_ENDPOINT_STT`, `AZURE_OPENAI_API_VERSION_STT`
  - Fallback to generic Azure environment variables
  - Full async support with `transcribe()` and `atranscribe()` methods
  - Example:
    ```python
    model = AIFactory.create_speech_to_text("azure", "whisper-deployment")
    response = model.transcribe("audio.mp3")
    ```

- **Azure OpenAI Text-to-Speech Support** - Added TTS model support via Azure deployments
  - Direct HTTP implementation using httpx (no SDK dependencies)
  - Modality-specific environment variables: `AZURE_OPENAI_API_KEY_TTS`, `AZURE_OPENAI_ENDPOINT_TTS`, `AZURE_OPENAI_API_VERSION_TTS`
  - Fallback to generic Azure environment variables
  - Supports all OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
  - Full async support with `generate_speech()` and `agenerate_speech()` methods
  - Example:
    ```python
    model = AIFactory.create_text_to_speech("azure", "tts-deployment")
    response = model.generate_speech("Hello!", voice="alloy")
    ```

- **Static Model Discovery** - New `AIFactory.get_provider_models()` method for discovering available models without creating provider instances
  - Supports all 15 providers with intelligent caching (1-hour TTL)
  - Type filtering for multi-model providers (OpenAI)
  - Pass provider-specific configuration (API keys, base URLs, etc.)
  - Example:
    ```python
    # Discover models without creating instances
    models = AIFactory.get_provider_models("openai", api_key="...")

    # Filter by type
    language_models = AIFactory.get_provider_models(
        "openai",
        api_key="...",
        model_type="language"
    )
    ```

- **Model Discovery Functions** - 16 provider-specific discovery functions in `model_discovery.py`:
  - API-based: OpenAI, OpenAI-Compatible, Google/Gemini, Vertex AI, Mistral, Groq, xAI, OpenRouter, Ollama
  - Hardcoded lists: Anthropic, DeepSeek, Perplexity, Jina, Voyage
  - Special cases: Azure (deployments), Transformers (local models)
  - OpenAI-Compatible supports any endpoint implementing the OpenAI API specification (LM Studio, vLLM, custom endpoints)

- **ModelCache Utility** - Thread-safe caching system for model discovery with configurable TTL

### Deprecated

- **`.models` property** on all provider instances (LanguageModel, EmbeddingModel, RerankerModel, SpeechToTextModel, TextToSpeechModel)
  - Will be removed in version 3.0.0
  - Emits `DeprecationWarning` with migration guidance
  - Use `AIFactory.get_provider_models()` instead

### Migration Guide - Static Model Discovery

#### Why This Change?

The new static discovery approach provides several benefits:
1. **No Instance Creation Required** - List models without creating provider instances
2. **Performance** - Cached results (1 hour TTL) reduce unnecessary API calls
3. **Consistency** - Unified interface across all providers
4. **Flexibility** - Pass configuration parameters as needed

#### Quick Migration Examples

**Before (Deprecated):**
```python
# ❌ Creating instance just to list models
model = AIFactory.create_language("openai", "gpt-4", config={"api_key": "..."})
available_models = model.models  # Deprecated
```

**After (Recommended):**
```python
# ✅ Static discovery without creating instances
available_models = AIFactory.get_provider_models("openai", api_key="...")
```

#### Detailed Migration Examples

**Basic Model Discovery:**
```python
# Before
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"api_key": "sk-..."}
)
models = model.models

# After
models = AIFactory.get_provider_models("openai", api_key="sk-...")
```

**Listing Embedding Models:**
```python
# Before
embedder = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-small",
    config={"api_key": "sk-..."}
)
embedding_models = embedder.models

# After
embedding_models = AIFactory.get_provider_models(
    "openai",
    api_key="sk-...",
    model_type="embedding"
)
```

**Provider-Specific Classes:**
```python
# Before
from esperanto.providers.llm.anthropic import AnthropicLanguageModel

model = AnthropicLanguageModel(api_key="sk-ant-...")
claude_models = model.models

# After
from esperanto import AIFactory

claude_models = AIFactory.get_provider_models("anthropic")
```

**Type Filtering (OpenAI):**
```python
# Before
model = OpenAILanguageModel(api_key="sk-...")
all_models = model.models
language_models = [m for m in all_models if m.id.startswith("gpt")]

# After
language_models = AIFactory.get_provider_models(
    "openai",
    api_key="sk-...",
    model_type="language"  # or 'embedding', 'speech_to_text', 'text_to_speech'
)
```

#### Timeline

- **Version 2.8.0** (Current) - `.models` property deprecated, warnings emitted
- **Version 3.0.0** (Future) - `.models` property will be removed

#### Suppressing Warnings (Temporary)

If you need time to migrate but want to suppress warnings temporarily:

```python
import warnings

# Suppress only Esperanto deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='esperanto')
```

**Note**: Suppressing warnings is not a long-term solution. Plan to migrate your code before version 3.0.0.

#### Provider-Specific Notes

**OpenAI:**
- Supports `model_type` parameter for filtering
- Requires API key (or `OPENAI_API_KEY` environment variable)
- Results are cached for 1 hour

**Anthropic:**
- Returns hardcoded list of Claude models
- No API key required for discovery
- Includes context window information

**Google/Gemini:**
- Fetches models via API
- Requires API key (or `GOOGLE_API_KEY`/`GEMINI_API_KEY` environment variable)
- Results are cached for 1 hour

**OpenAI-Compatible:**
- Fetches models from any OpenAI-compatible endpoint
- Requires `base_url` parameter (e.g., `http://localhost:1234/v1` for LM Studio)
- Optional `api_key` if the endpoint requires authentication
- Supports type filtering
- Results are cached for 1 hour
- Example:
  ```python
  models = AIFactory.get_provider_models(
      "openai-compatible",
      base_url="http://localhost:1234/v1"
  )
  ```

**Ollama:**
- Lists locally available models
- Requires Ollama to be running locally
- Default base URL: `http://localhost:11434`

**Transformers:**
- Currently returns empty list (local models are not auto-discovered)
- You need to specify model names explicitly when creating instances

**Azure:**
- Returns empty list (Azure uses deployments, not discoverable models)
- You need to specify your deployment names explicitly

#### API Reference

**New Method:**
```python
AIFactory.get_provider_models(
    provider: str,                    # Provider name (e.g., "openai", "anthropic")
    model_type: Optional[str] = None, # Filter by type (OpenAI only)
    **config                          # Provider-specific configuration (api_key, base_url, etc.)
) -> List[Model]
```

**Model Object:**
```python
@dataclass(frozen=True)
class Model:
    id: str                           # Model identifier (e.g., "gpt-4")
    owned_by: str                     # Owner organization (e.g., "openai")
    context_window: Optional[int]     # Max context size in tokens
    type: Optional[str] = None        # Model type (optional)
```

#### Migration Checklist

- [ ] Replace all uses of `.models` with `AIFactory.get_provider_models()`
- [ ] Update API key passing from instance creation to discovery method
- [ ] Add `model_type` parameter where needed (OpenAI)
- [ ] Test that model discovery works with your provider configuration
- [ ] Remove any manual filtering code (use `model_type` instead)
- [ ] Update documentation/comments in your codebase
- [ ] Verify no deprecation warnings are emitted

#### Code Search Tips

To find all uses of the deprecated API in your codebase:

```bash
# Search for .models property access
grep -r "\.models" --include="*.py" .

# Search for specific providers
grep -r "LanguageModel.*\.models" --include="*.py" .
grep -r "EmbeddingModel.*\.models" --include="*.py" .
```

### Internal Changes

- Renamed internal `models()` method to `_get_models()` across all 34 provider implementations
- Added comprehensive test coverage (37 new tests)
- Improved code coverage to 68%

---

### Planned for 3.0.0
- Remove deprecated `.models` property from all provider base classes

---

**For older versions, see Git history.**
