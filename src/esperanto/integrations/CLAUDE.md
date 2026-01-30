# Integrations Module

Third-party framework integrations for Esperanto models.

## Files

- **`__init__.py`**: Module exports (EsperantoPydanticModel)
- **`pydantic_ai.py`**: Pydantic AI integration adapter

## Pydantic AI Integration

### Overview

The `pydantic_ai.py` module provides an adapter that allows any Esperanto `LanguageModel` to be used with Pydantic AI agents. This enables users to:

- Configure models once through Esperanto's unified interface
- Use those models with Pydantic AI's agent framework
- Switch providers without changing agent code
- Maintain a single source of truth for model configuration

### Key Classes

#### `EsperantoPydanticModel`

Main adapter class that implements Pydantic AI's `Model` interface.

```python
from esperanto import AIFactory
from pydantic_ai import Agent

model = AIFactory.create_language("openai", "gpt-4o")
pydantic_model = model.to_pydantic_ai()  # Returns EsperantoPydanticModel

agent = Agent(pydantic_model)
```

**Key Methods:**
- `request()`: Non-streaming request to the model
- `request_stream()`: Streaming request returning `EsperantoStreamedResponse`
- `_convert_messages()`: Converts Pydantic AI messages to Esperanto format
- `_convert_response()`: Converts Esperanto `ChatCompletion` to Pydantic AI `ModelResponse`
- `_convert_tools()`: Converts Pydantic AI `ToolDefinition` to Esperanto `Tool` objects
- `_apply_settings()`: Extracts settings from `ModelSettings` (TypedDict)

**Properties:**
- `model_name`: Returns the model identifier
- `system`: Returns the provider name (used for telemetry)

#### `EsperantoStreamedResponse`

Streaming response handler that implements Pydantic AI's `StreamedResponse` interface.

**Key Methods:**
- `_get_event_iterator()`: Abstract method implementation that yields `ModelResponseStreamEvent`
- Uses `ModelResponsePartsManager` for proper event handling
- Supports text streaming and tool call streaming

### Message Conversion

Pydantic AI uses a different message format than Esperanto's OpenAI-compatible format:

| Pydantic AI | Esperanto |
|-------------|-----------|
| `SystemPromptPart` | `{"role": "system", "content": ...}` |
| `UserPromptPart` | `{"role": "user", "content": ...}` |
| `TextPart` in `ModelResponse` | `{"role": "assistant", "content": ...}` |
| `ToolCallPart` | `{"role": "assistant", "tool_calls": [...]}` |
| `ToolReturnPart` | `{"role": "tool", "tool_call_id": ..., "content": ...}` |
| `RetryPromptPart` | `{"role": "user", "content": "Error: ..."}` |

### Tool Conversion

Tools are converted from Pydantic AI's `ToolDefinition` to Esperanto's `Tool` objects:

```python
# Pydantic AI format
ToolDefinition(
    name="get_weather",
    description="Get weather for a city",
    parameters_json_schema={...}
)

# Converted to Esperanto format
Tool(
    type="function",
    function=ToolFunction(
        name="get_weather",
        description="Get weather for a city",
        parameters={...}
    )
)
```

### ModelSettings Handling

Pydantic AI's `ModelSettings` is a `TypedDict`, not a class with attributes. Access values using dict-style syntax:

```python
# Correct
settings.get("temperature")
settings.get("max_tokens")

# Incorrect (won't work)
settings.temperature
settings.max_tokens
```

### Dependency Handling

The module handles optional pydantic-ai dependency gracefully:

```python
try:
    import pydantic_ai
    PYDANTIC_AI_INSTALLED = True
except ImportError:
    PYDANTIC_AI_INSTALLED = False
```

The `to_pydantic_ai()` method on `LanguageModel` raises a helpful `ImportError` if pydantic-ai is not installed.

### Provider-Specific Considerations

Some providers require special handling:

1. **Google**: JSON schemas must not contain `additionalProperties` - handled in the Google provider's `_clean_schema_for_google()` method

2. **Google tool responses**: Must be dict/Struct, not primitives - handled in `_convert_messages()`

3. **Ollama**: Expects `Tool` objects, not dicts - `_convert_tools()` returns `Tool` objects

## Adding New Integrations

When adding a new framework integration:

1. Create a new file (e.g., `new_framework.py`)
2. Implement the framework's model interface
3. Add conversion methods for messages, responses, and tools
4. Handle streaming if supported
5. Add to `__init__.py` exports with try/except for optional dependencies
6. Add `to_new_framework()` method to `LanguageModel` base class
7. Write tests in `tests/unit/test_new_framework_adapter.py`
8. Write integration tests in `tests/integration/test_new_framework_integration.py`
9. Add documentation in `docs/advanced/new-framework-integration.md`

## Testing

- **Unit tests**: `tests/unit/test_pydantic_ai_adapter.py` - Test conversion methods with mocks
- **Integration tests**: `tests/integration/test_pydantic_ai_integration.py` - Test with real Pydantic AI but mocked LLM responses

## Gotchas

- `ModelSettings` is a TypedDict - use dict access, not attribute access
- `StreamedResponse` requires implementing `_get_event_iterator()` abstract method
- `ModelResponsePartsManager` methods use keyword arguments with `vendor_part_id`
- Some providers (Google) don't support all JSON Schema fields - clean schemas as needed
- Tool response content must be serializable and match provider expectations
