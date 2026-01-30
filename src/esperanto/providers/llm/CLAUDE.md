# Language Model Providers

Language model (LLM) provider implementations for chat completion APIs.

## Files

- **`base.py`**: Abstract base class `LanguageModel` defining the interface all providers must implement
- **`openai.py`**: OpenAI GPT models (GPT-4, GPT-3.5, etc.)
- **`anthropic.py`**: Anthropic Claude models (Claude 3.7 Sonnet, etc.)
- **`google.py`**: Google Gemini models
- **`azure.py`**: Azure OpenAI Service integration
- **`ollama.py`**: Local Ollama models
- **`groq.py`**: Groq inference API
- **`mistral.py`**: Mistral AI models
- **`deepseek.py`**: DeepSeek models
- **`xai.py`**: xAI (X.AI) models
- **`perplexity.py`**: Perplexity AI models
- **`openrouter.py`**: OpenRouter unified API
- **`vertex.py`**: Google Vertex AI
- **`openai_compatible.py`**: Generic OpenAI-compatible API provider (for custom endpoints)

## Patterns

### Base Class Contract

All providers inherit from `LanguageModel` (base.py:14) and must:

1. **Implement abstract methods**:
   - `chat_complete()`: Synchronous chat completion
   - `achat_complete()`: Async chat completion
   - `_get_models()`: Return list of available models
   - `_get_default_model()`: Return default model name
   - `to_langchain()`: Convert to LangChain chat model
   - `provider` property: Return provider name string

2. **Override `__post_init__()`**:
   - Call `super().__post_init__()` first
   - Set `api_key` from parameter or environment variable
   - Set `base_url` (use default if not provided)
   - Call `self._create_http_clients()` last to initialize httpx clients with timeout/SSL settings

3. **Error Handling**:
   - Implement `_handle_error()` to parse provider-specific error responses
   - Raise `RuntimeError` with descriptive messages

### Response Normalization

Providers convert API responses to Esperanto's common types:

- **Non-streaming**: Return `ChatCompletion` object
- **Streaming**: Yield `ChatCompletionChunk` objects
- Use helper methods like `_normalize_response()` and `_normalize_stream_chunk()`

### Configuration Management

- Providers use `self._config` dict (initialized in base class)
- Call `get_model_name()` to retrieve model from config or use default
- Use `get_completion_kwargs()` for common parameters (max_tokens, temperature, etc.)

### HTTP Client Pattern

All providers use httpx for HTTP requests:

```python
def __post_init__(self):
    super().__post_init__()
    self.api_key = self.api_key or os.getenv("PROVIDER_API_KEY")
    self.base_url = self.base_url or "https://api.provider.com/v1"
    self._create_http_clients()  # Creates self.client and self.async_client
```

The `_create_http_clients()` method from base class handles timeout and SSL configuration via mixins.

### Streaming Implementation

Streaming is handled differently by provider:

- **OpenAI-style**: SSE (Server-Sent Events) with `data: {...}` format
- **Anthropic**: Custom streaming format with event types
- All convert to unified `ChatCompletionChunk` format

### Structured Output

Providers support structured output via `self.structured` parameter:

- OpenAI: Uses `response_format` parameter
- Google: Uses `generation_config` with schema
- Anthropic: Not natively supported (returns JSON as string)

## Integration

- Imported by `factory.py` via `AIFactory._provider_modules["language"]`
- Registered with provider name (e.g., "openai", "anthropic")
- Uses types from `esperanto.common_types`
- Inherits mixins from `esperanto.utils.timeout` and `esperanto.utils.ssl`

## Gotchas

- **Always call `super().__post_init__()` first** - base class initializes `_config` dict
- **Call `_create_http_clients()` last** in `__post_init__` - requires api_key and base_url to be set
- **API key validation**: Check for None and raise ValueError with helpful message
- **Model filtering**: When implementing `_get_models()`, filter out non-language models (e.g., OpenAI filters to only `gpt` models, not embeddings/TTS/etc)
- **Streaming detection**: Use `stream = stream if stream is not None else self.streaming` to allow per-request override
- **LangChain conversion**: Must handle optional LangChain dependencies gracefully, raise ImportError if not installed
- **Default models**: `_get_default_model()` should return a widely available, stable model (not the latest/most expensive)
- **Base URL normalization**: Strip trailing slashes from base_url to avoid double-slash issues
- **Environment variable naming**: Follow pattern `{PROVIDER}_API_KEY` (all caps)
- **Deprecation warnings**: Use `_get_models()` internally (not `.models` property which is deprecated)

## When Adding a New Provider

1. Create new file `provider_name.py`
2. Import `LanguageModel` from `esperanto.providers.llm.base`
3. Define class inheriting from `LanguageModel`
4. Implement all abstract methods
5. Add `__post_init__()` following the pattern above
6. Create response normalization helpers
7. Add provider to `factory.py` in `_provider_modules["language"]` dict
8. Add optional import in `src/esperanto/__init__.py`
9. Write tests in `tests/providers/llm/test_provider_name.py`
10. Add documentation in `docs/providers/provider_name.md`

## Common Patterns Across Providers

### Message Format Transformation

Most providers expect different message formats:

- **OpenAI**: `{"role": "user", "content": "..."}`
- **Anthropic**: Requires system messages separate from message list
- **Google**: Uses `{"role": "user", "parts": [{"text": "..."}]}`

Implement `_convert_messages()` or similar to transform Esperanto's standard format.

### Tool/Function Calling

Esperanto provides unified tool calling across all providers. The base class defines tool-related parameters:

- `tools: Optional[List[Tool]]` - List of tools the model can call
- `tool_choice: Optional[Union[str, Dict[str, Any]]]` - Controls tool usage ("auto", "required", "none", or specific tool)
- `parallel_tool_calls: Optional[bool]` - Allow multiple tool calls per response

**Provider implementations must:**

1. Add `tools`, `tool_choice`, `parallel_tool_calls` parameters to `chat_complete()` and `achat_complete()`
2. Implement `_convert_tools_to_*()` to convert Esperanto's `Tool` format to provider-specific format
3. Update `_normalize_response()` to extract `ToolCall` objects from responses
4. Handle tool result messages (role="tool" with tool_call_id)

**Format conversion examples:**

```python
# Esperanto unified format (input)
Tool(
    type="function",
    function=ToolFunction(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {...}}
    )
)

# OpenAI format
{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

# Anthropic format
{"name": "...", "description": "...", "input_schema": {...}}

# Google format
{"function_declarations": [{"name": "...", "description": "...", "parameters": {...}}]}
```

**Tool call response normalization:**

```python
# All providers normalize to ToolCall objects
tool_calls = [
    ToolCall(
        id="call_abc123",
        type="function",
        function=FunctionCall(
            name="get_weather",
            arguments='{"location": "Tokyo"}'  # Always JSON string
        )
    )
]
```

**Provider-specific notes:**

- **OpenAI**: Native support, format matches closely
- **Anthropic**: Tool calls in content blocks as `tool_use`, tool results as `tool_result` in user message
- **Google**: No tool call IDs (Esperanto generates them), uses `functionCall` in parts
- **Groq/Mistral/Azure/etc**: OpenAI-compatible format

### Reasoning Models

Some providers (OpenAI o1, DeepSeek R1) have reasoning models with special constraints:

- Limited/no temperature control
- Different max_tokens behavior
- Reasoning traces in responses

Handle these models separately or with conditional logic based on model name.
