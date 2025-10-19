# Architecture: Provider-Specific Environment Variables for OpenAI-Compatible Providers

**Feature**: Provider-Specific Environment Variables for OpenAI-Compatible Providers
**Date**: 2025-10-18
**Branch**: `openai-compatible-env-vars`
**Specs**: [spec.md](./spec.md)

## Summary

We're implementing provider-specific environment variables for all four OpenAI-compatible providers (LLM, Embedding, STT, TTS) to enable users to configure different endpoints for each provider type. The solution maintains backward compatibility through a fallback mechanism where provider-specific variables take precedence over generic ones.

## Technical Context

**Language/Stack**: Python 3.10+
**Key Dependencies**:
- `httpx` - HTTP client library (already in use)
- `os` - Environment variable access (standard library)
**Storage**: N/A - Configuration only
**Testing**: pytest with monkeypatch for environment variable testing
**Platform**: Cross-platform Python library

## Technical Decisions

### Decision 1: Environment Variable Fallback Pattern

**What**: Implement a hierarchical fallback mechanism for environment variables: Provider-specific → Generic → Default

**Why**:
- Maintains backward compatibility with existing configurations
- Provides maximum flexibility without breaking existing code
- Follows established patterns in the codebase (similar to timeout configuration)

**Alternatives Considered**:
- Replace generic variables entirely - Rejected: Would break existing deployments
- Use different precedence order - Rejected: Would create confusion and inconsistency

**Trade-offs**:
- Gain: Seamless migration path, no breaking changes
- Lose: Slightly more complex lookup logic (negligible performance impact)

### Decision 2: Consistent Pattern Across All Providers

**What**: Apply the exact same implementation pattern to all four provider types

**Why**:
- Ensures consistency and predictability for users
- Easier to maintain and test
- Reduces cognitive load when working with different provider types

**Alternatives Considered**:
- Provider-specific implementations - Rejected: Would create maintenance burden and inconsistency

**Trade-offs**:
- Gain: Consistency, maintainability, easier documentation
- Lose: None - this is the optimal approach

### Decision 3: Update Error Messages to Show Both Options

**What**: Error messages will display both provider-specific and generic variable names

**Why**:
- Helps users understand the fallback mechanism
- Provides clear guidance on configuration options
- Reduces support burden

**Alternatives Considered**:
- Show only provider-specific variable - Rejected: Users might not know about generic fallback
- Show full precedence chain - Rejected: Too verbose and overwhelming

**Trade-offs**:
- Gain: Clear user guidance, self-documenting behavior
- Lose: Slightly longer error messages (acceptable)

### Decision 4: Environment Variable Naming Convention

**What**: Use suffix pattern with provider type abbreviations
- LLM: `OPENAI_COMPATIBLE_BASE_URL_LLM`, `OPENAI_COMPATIBLE_API_KEY_LLM`
- Embedding: `OPENAI_COMPATIBLE_BASE_URL_EMBEDDING`, `OPENAI_COMPATIBLE_API_KEY_EMBEDDING`
- STT: `OPENAI_COMPATIBLE_BASE_URL_STT`, `OPENAI_COMPATIBLE_API_KEY_STT`
- TTS: `OPENAI_COMPATIBLE_BASE_URL_TTS`, `OPENAI_COMPATIBLE_API_KEY_TTS`

**Why**:
- Groups related variables together alphabetically
- Uses commonly understood abbreviations
- Matches user's original suggestion
- Follows existing timeout configuration pattern

**Alternatives Considered**:
- Prefix pattern - Rejected: Doesn't group as well alphabetically
- Full names instead of abbreviations - Rejected: Too verbose

**Trade-offs**:
- Gain: Alphabetical grouping, familiar abbreviations
- Lose: None - this matches established patterns

## Architecture Overview

Currently, all four OpenAI-compatible providers share the same environment variables (`OPENAI_COMPATIBLE_BASE_URL` and `OPENAI_COMPATIBLE_API_KEY`). This creates a limitation where users can only configure one endpoint for all provider types.

After this change, each provider will check for its specific environment variables first, then fall back to the generic ones if not found. The configuration precedence will be:

1. Direct parameters (highest priority)
2. Config dictionary
3. Provider-specific environment variables ← **NEW**
4. Generic environment variables
5. Default values (lowest priority)

### Component Structure

**Modified Files:**
```
src/esperanto/providers/llm/openai_compatible.py - Update env var lookup for LLM
src/esperanto/providers/embedding/openai_compatible.py - Update env var lookup for Embedding
src/esperanto/providers/stt/openai_compatible.py - Update env var lookup for STT
src/esperanto/providers/tts/openai_compatible.py - Update env var lookup for TTS
README.md - Document new environment variables
.env.example - Add examples for new variables
```

**Modified Test Files:**
```
tests/providers/llm/test_openai_compatible_provider.py - Add provider-specific env var tests
tests/providers/embedding/test_openai_compatible.py - Add provider-specific env var tests
tests/providers/stt/test_openai_compatible.py - Add provider-specific env var tests
tests/providers/tts/test_openai_compatible.py - Add provider-specific env var tests
```

### Component Relationships

Each provider operates independently with its own environment variable lookup:
- `OpenAICompatibleLanguageModel` → checks `OPENAI_COMPATIBLE_BASE_URL_LLM` → falls back to `OPENAI_COMPATIBLE_BASE_URL`
- `OpenAICompatibleEmbeddingModel` → checks `OPENAI_COMPATIBLE_BASE_URL_EMBEDDING` → falls back to `OPENAI_COMPATIBLE_BASE_URL`
- `OpenAICompatibleSpeechToTextModel` → checks `OPENAI_COMPATIBLE_BASE_URL_STT` → falls back to `OPENAI_COMPATIBLE_BASE_URL`
- `OpenAICompatibleTextToSpeechModel` → checks `OPENAI_COMPATIBLE_BASE_URL_TTS` → falls back to `OPENAI_COMPATIBLE_BASE_URL`

### Data Flow

**Configuration Resolution Flow:**
```
1. User creates provider via AIFactory or direct instantiation
2. Provider __init__ or __post_init__ executes
3. For base_url:
   a. Check direct parameter
   b. Check config dict
   c. Check provider-specific env var (e.g., OPENAI_COMPATIBLE_BASE_URL_LLM)
   d. Check generic env var (OPENAI_COMPATIBLE_BASE_URL)
   e. Raise error if not found
4. For api_key:
   a. Check direct parameter
   b. Check config dict
   c. Check provider-specific env var (e.g., OPENAI_COMPATIBLE_API_KEY_LLM)
   d. Check generic env var (OPENAI_COMPATIBLE_API_KEY)
   e. Use "not-required" default if not found
```

## Implementation Approach

### User Story Mapping

**US-001: Environment variable hierarchy implementation for all OpenAI-compatible providers**
- Files involved:
  - `src/esperanto/providers/llm/openai_compatible.py`
  - `src/esperanto/providers/embedding/openai_compatible.py`
  - `src/esperanto/providers/stt/openai_compatible.py`
  - `src/esperanto/providers/tts/openai_compatible.py`
- Key components: Environment variable lookup logic in `__init__` or `__post_init__`
- Dependencies: None
- Testing approach: Unit tests with monkeypatch to set environment variables

**US-002: Update test suite to validate provider-specific environment variables**
- Files involved:
  - `tests/providers/llm/test_openai_compatible_provider.py`
  - `tests/providers/embedding/test_openai_compatible.py`
  - `tests/providers/stt/test_openai_compatible.py`
  - `tests/providers/tts/test_openai_compatible.py`
- Key components: New test cases for provider-specific env vars and fallback behavior
- Dependencies: US-001 must be completed first
- Testing approach:
  - Test provider-specific env var takes precedence
  - Test fallback to generic env var
  - Test backward compatibility (existing tests should pass)

**US-003: Update documentation to explain the new configuration options**
- Files involved:
  - `README.md`
  - `.env.example`
- Key components: Environment variable documentation, examples
- Dependencies: US-001 completed for accurate documentation
- Testing approach: Manual review of documentation clarity

**US-004: Ensure backward compatibility with existing configurations**
- Files involved: All provider files (validation phase)
- Key components: Existing test suite
- Dependencies: US-001, US-002
- Testing approach: Run full test suite, verify no breaking changes

### File Structure

```
esperanto/
├── src/esperanto/providers/
│   ├── llm/
│   │   └── openai_compatible.py    # MODIFIED - Add LLM-specific env vars
│   ├── embedding/
│   │   └── openai_compatible.py    # MODIFIED - Add EMBEDDING-specific env vars
│   ├── stt/
│   │   └── openai_compatible.py    # MODIFIED - Add STT-specific env vars
│   └── tts/
│       └── openai_compatible.py    # MODIFIED - Add TTS-specific env vars
├── tests/providers/
│   ├── llm/
│   │   └── test_openai_compatible_provider.py  # MODIFIED - Add new tests
│   ├── embedding/
│   │   └── test_openai_compatible.py           # MODIFIED - Add new tests
│   ├── stt/
│   │   └── test_openai_compatible.py           # MODIFIED - Add new tests
│   └── tts/
│       └── test_openai_compatible.py           # MODIFIED - Add new tests
├── README.md           # MODIFIED - Document new env vars
└── .env.example        # MODIFIED - Add examples
```

## Integration Points

**Integration with existing configuration system:**
- The new provider-specific environment variables integrate into the existing configuration precedence chain
- No changes to the AIFactory or base classes required
- Pattern follows the established timeout configuration approach in `utils/timeout.py`

**New Environment Variables Exposed:**
- `OPENAI_COMPATIBLE_BASE_URL_LLM`
- `OPENAI_COMPATIBLE_API_KEY_LLM`
- `OPENAI_COMPATIBLE_BASE_URL_EMBEDDING`
- `OPENAI_COMPATIBLE_API_KEY_EMBEDDING`
- `OPENAI_COMPATIBLE_BASE_URL_STT`
- `OPENAI_COMPATIBLE_API_KEY_STT`
- `OPENAI_COMPATIBLE_BASE_URL_TTS`
- `OPENAI_COMPATIBLE_API_KEY_TTS`

**Dependencies:**
- No new external dependencies required
- Relies on existing `os.getenv()` functionality
- Compatible with all existing provider initialization paths

## Technical Constraints

- **Must maintain backward compatibility**: Existing configurations using generic environment variables must continue to work without any changes
- **Must follow established patterns**: Implementation should align with existing timeout configuration pattern
- **Must be consistent across providers**: All four providers must implement the same pattern
- **Must not affect performance**: Environment variable lookup is already used; adding one more check is negligible

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes to existing deployments | High | Implement fallback mechanism; maintain generic env vars with lower priority |
| Inconsistent implementation across providers | Medium | Use identical pattern for all four providers; comprehensive testing |
| User confusion about precedence | Low | Clear documentation with examples; informative error messages |
| Test coverage gaps | Medium | Add specific tests for each provider validating both specific and generic env vars |

## Implementation Pattern

Each provider will use this pattern:

```python
# For base_url
self.base_url = (
    base_url or                                          # Direct parameter (highest)
    config.get("base_url") or                           # Config dict
    os.getenv("OPENAI_COMPATIBLE_BASE_URL_<TYPE>") or  # Provider-specific env var
    os.getenv("OPENAI_COMPATIBLE_BASE_URL")            # Generic env var (fallback)
)

# For api_key
self.api_key = (
    api_key or                                          # Direct parameter (highest)
    config.get("api_key") or                           # Config dict
    os.getenv("OPENAI_COMPATIBLE_API_KEY_<TYPE>") or  # Provider-specific env var
    os.getenv("OPENAI_COMPATIBLE_API_KEY")            # Generic env var (fallback)
)

# Updated error message
if not self.base_url:
    raise ValueError(
        "OpenAI-compatible base URL is required. "
        f"Set OPENAI_COMPATIBLE_BASE_URL_<TYPE> or OPENAI_COMPATIBLE_BASE_URL "
        "environment variable or provide base_url in config."
    )
```

Where `<TYPE>` is:
- `LLM` for language models
- `EMBEDDING` for embedding models
- `STT` for speech-to-text models
- `TTS` for text-to-speech models

## Test Strategy

### Test Cases Per Provider

Each provider needs these test cases:

1. **Provider-specific env var takes precedence over generic**
   - Set both `OPENAI_COMPATIBLE_BASE_URL_<TYPE>` and `OPENAI_COMPATIBLE_BASE_URL`
   - Verify provider uses the specific one

2. **Fallback to generic env var**
   - Set only `OPENAI_COMPATIBLE_BASE_URL`
   - Verify provider uses the generic one

3. **Config overrides env vars**
   - Set env var and pass config
   - Verify config takes precedence

4. **Direct parameter overrides everything**
   - Set env var, config, and direct parameter
   - Verify direct parameter wins

5. **Backward compatibility**
   - Run existing tests to ensure no breakage
   - Verify generic env vars still work as before

## Open Questions

None - all clarifications have been resolved in the spec.

## Next Steps

1. Review this architecture with stakeholders
2. Confirm implementation approach aligns with project standards
3. Run `/por:plan` to break down into detailed implementation tasks
4. Begin implementation starting with one provider as a reference
