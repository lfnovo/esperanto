# Feature Specification: Provider-Specific Environment Variables for OpenAI-Compatible Providers

**Feature Branch**: `openai-compatible-env-vars`
**Input**: User description: "We have several openai compatible providers:

@src/esperanto/providers/stt/openai_compatible.py
@src/esperanto/providers/tts/openai_compatible.py
@src/esperanto/providers/llm/openai_compatible.py
@src/esperanto/providers/embedding/openai_compatible.py

I noticed an issue with the way we configured this. All of them are using the same BASE_URL and API_KEY environment variables. So, if the user has different services for each of them, it won't be possible to use them.

Here is what I want to do:

Each provider with have its own env variable

For instance: OPENAI_COMPATIBLE_BASE_URL_EMBEDDING and OPENAI_COMPATIBLE_API_KEY_EMBEDDING.

So, if the specific variable exists, it should be used and we should fallback to the generic one if that one is not present.

We need to change this on all 4 providers, update the tests and documentation"

## Context and Understanding

Currently, all four OpenAI-compatible providers (LLM, Embedding, Speech-to-Text, and Text-to-Speech) share the same environment variables: `OPENAI_COMPATIBLE_BASE_URL` and `OPENAI_COMPATIBLE_API_KEY`. This creates a limitation where users cannot configure different OpenAI-compatible endpoints for different provider types. For example, a user might want to use a local LM Studio instance for LLM operations, a different service for embeddings, and yet another for speech processing. The current implementation forces all providers to use the same endpoint, which is not flexible enough for real-world scenarios where users might have specialized services for different AI tasks.

## Feature Description

This feature enhances the OpenAI-compatible provider configuration system by introducing provider-specific environment variables while maintaining backward compatibility through a fallback mechanism. Each provider type (LLM, Embedding, STT, TTS) will have its own dedicated environment variables that take precedence over the generic ones, allowing users to configure different endpoints for different AI capabilities without code changes.

## Requirements

### Proposed Solution

- **US-001**: Environment variable hierarchy implementation for all OpenAI-compatible providers
- **US-002**: Update test suite to validate provider-specific environment variables
- **US-003**: Update documentation to explain the new configuration options
- **US-004**: Ensure backward compatibility with existing configurations

### Functional Requirements

- **FR-001**: System MUST support provider-specific environment variables for each OpenAI-compatible provider type (LLM, EMBEDDING, STT, TTS)
- **FR-002**: System MUST implement a fallback mechanism where provider-specific variables take precedence over generic ones
- **FR-003**: System MUST maintain backward compatibility - existing configurations using generic variables must continue to work
- **FR-004**: Each provider MUST check for its specific environment variable first, then fall back to the generic one if not found
- **FR-005**: Error messages MUST be updated to mention both the specific and generic environment variable options
- **FR-006**: The configuration precedence MUST be: Direct parameters > Config dict > Provider-specific env vars > Generic env vars > Default values

## Success Criteria

### Measurable Outcomes

- **SC-001**: Users can configure different OpenAI-compatible endpoints for each provider type using environment variables
- **SC-002**: Existing configurations using generic environment variables continue to work without any changes
- **SC-003**: All four providers (LLM, Embedding, STT, TTS) support their respective provider-specific variables
- **SC-004**: Documentation clearly explains the new configuration options and precedence rules
- **SC-005**: Test coverage validates both provider-specific and generic fallback scenarios for all providers
- **SC-006**: No breaking changes - existing code using the library continues to work as before

## Clarification Needed

### 1. Environment Variable Naming Convention ✅ RESOLVED
**Decision**: Suffix format: `OPENAI_COMPATIBLE_BASE_URL_EMBEDDING`

This format groups related variables together when sorted alphabetically and matches the original suggestion.

### 2. Provider Type Identifiers ✅ RESOLVED
**Decision**: Mixed (common abbreviations): `_LLM`, `_EMBEDDING`, `_STT`, `_TTS`

Using commonly understood abbreviations that align with industry standards.

**Final environment variable names:**
- LLM: `OPENAI_COMPATIBLE_BASE_URL_LLM`, `OPENAI_COMPATIBLE_API_KEY_LLM`
- Embedding: `OPENAI_COMPATIBLE_BASE_URL_EMBEDDING`, `OPENAI_COMPATIBLE_API_KEY_EMBEDDING`
- STT: `OPENAI_COMPATIBLE_BASE_URL_STT`, `OPENAI_COMPATIBLE_API_KEY_STT`
- TTS: `OPENAI_COMPATIBLE_BASE_URL_TTS`, `OPENAI_COMPATIBLE_API_KEY_TTS`

### 3. Error Message Updates ✅ RESOLVED
**Decision**: Show both options

Error messages will indicate both the provider-specific and generic variables: "Set OPENAI_COMPATIBLE_BASE_URL_LLM or OPENAI_COMPATIBLE_BASE_URL environment variable"

This approach is informative without being overwhelming and helps users understand the fallback mechanism.

## Notes

- This change improves the flexibility of the Esperanto library significantly for users with diverse AI infrastructure
- The backward compatibility ensures this is a non-breaking change that can be released as a minor version update
- Consider adding a debug log that shows which configuration source was ultimately used for better troubleshooting
- Future consideration: Could extend this pattern to other provider types if they ever need OpenAI-compatible variants