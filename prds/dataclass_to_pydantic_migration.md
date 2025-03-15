# Dataclass to Pydantic Migration Plan

## Overview
This document outlines the plan to migrate all dataclass implementations in the Esperanto project to Pydantic models. This migration will provide better validation, serialization, and documentation capabilities.

We'll use Pydantic 2.0.0 as our minimum version requirement to ensure maximum compatibility while still leveraging the improved features and performance of Pydantic 2.x. This version provides a good balance between modern features and broad compatibility with other libraries.

## Files Requiring Updates

### 1. Base Provider Classes
- `src/esperanto/providers/embedding/base.py`:
  - `EmbeddingModel` class
  - `PoolingConfig` class (in transformers.py)

- `src/esperanto/providers/llm/base.py`:
  - `LanguageModel` class

- `src/esperanto/providers/stt/base.py`:
  - `SpeechToTextModel` class

- `src/esperanto/providers/tts/base.py`:
  - `TextToSpeechModel` class

### 2. Type Definitions
- `src/esperanto/types/model.py`:
  - `Model` class

- `src/esperanto/types/response.py`:
  - `Usage` class
  - `Message` class
  - `ChatCompletionMessage` class
  - `DeltaMessage` class
  - `Choice` class
  - `ChatCompletionChoice` class
  - `StreamChoice` class
  - `ChatCompletion` class
  - `ChatCompletionChunk` class

- `src/esperanto/types/stt.py`:
  - `TranscriptionResponse` class

- `src/esperanto/types/tts.py`:
  - `AudioResponse` class
  - `Voice` class

## Required Changes

1. Add Pydantic dependency to pyproject.toml:
```toml
dependencies = [
    "pydantic>=2.0.0",
    ...
]
```

2. For each dataclass:
   - Replace `@dataclass` decorator with Pydantic model inheritance
   - Convert type hints to use Pydantic's Field when needed
   - Add model_config for customization
   - Add field validations where appropriate
   - Update docstrings to include field descriptions

3. Example conversion:
```python
# Before
@dataclass
class Model:
    id: str
    owned_by: str
    context_window: Optional[int] = None
    type: Literal["language", "embedding", "text_to_speech", "speech_to_text"] = "language"

# After
class Model(BaseModel):
    id: str = Field(description="The model identifier")
    owned_by: str = Field(description="The organization that owns the model")
    context_window: Optional[int] = Field(default=None, description="Maximum context window size")
    type: Literal["language", "embedding", "text_to_speech", "speech_to_text"] = Field(
        default="language",
        description="The type of model"
    )

    model_config = ConfigDict(frozen=True)
```

4. Special Considerations:
   - Base classes need to maintain their ABC inheritance while adding Pydantic
   - Ensure all validation methods are preserved
   - Add JSON schema validation where needed
   - Maintain backwards compatibility for any external code
   - Update tests to handle Pydantic models

5. Testing Requirements:
   - Add validation tests for each model
   - Test JSON serialization/deserialization
   - Verify all providers still work with new models
   - Test error cases with invalid data

## Implementation Steps

1. Add Pydantic dependency
2. Create new Pydantic models alongside existing dataclasses
3. Update one file at a time, starting with base types
4. Update dependent files
5. Run tests after each file update
6. Update documentation
7. Final integration testing

## Benefits

1. Better Validation:
   - Type checking at runtime
   - Custom validation rules
   - Clear error messages

2. Improved Serialization:
   - Built-in JSON serialization
   - Schema generation
   - Better handling of nested models

3. Enhanced Documentation:
   - Automatic schema documentation
   - Field descriptions in code
   - Better IDE support

## Risks and Mitigations

1. Risk: Breaking changes in external code
   - Mitigation: Maintain compatibility layer initially
   - Plan: Phase out compatibility layer in next major version

2. Risk: Performance impact
   - Mitigation: Benchmark before and after
   - Plan: Optimize if needed

3. Risk: Learning curve for contributors
   - Mitigation: Add examples in CONTRIBUTING.md
   - Plan: Update documentation with Pydantic best practices
