# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- **Model Discovery Functions** - 15 provider-specific discovery functions in `model_discovery.py`:
  - API-based: OpenAI, Google/Gemini, Vertex AI, Mistral, Groq, xAI, OpenRouter, Ollama
  - Hardcoded lists: Anthropic, DeepSeek, Perplexity, Jina, Voyage
  - Special cases: Azure (deployments), Transformers (local models)

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

## [Unreleased]

### Planned for 3.0.0
- Remove deprecated `.models` property from all provider base classes

---

**For older versions, see Git history.**
