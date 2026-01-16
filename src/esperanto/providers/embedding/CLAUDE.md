# Embedding Model Providers

Embedding provider implementations for text-to-vector conversion.

## Files

- **`base.py`**: Abstract base class `EmbeddingModel` defining the interface
- **`openai.py`**: OpenAI embedding models (text-embedding-3-small/large, etc.)
- **`google.py`**: Google embedding models (text-embedding-004, etc.)
- **`azure.py`**: Azure OpenAI embedding models
- **`ollama.py`**: Local Ollama embedding models
- **`mistral.py`**: Mistral embedding models
- **`jina.py`**: Jina AI embedding models
- **`voyage.py`**: Voyage AI embedding models
- **`openrouter.py`**: OpenRouter embedding API
- **`vertex.py`**: Google Vertex AI embeddings
- **`transformers.py`**: Local HuggingFace transformers models (BERT, sentence-transformers, etc.)
- **`openai_compatible.py`**: Generic OpenAI-compatible embedding API

## Patterns

### Base Class Contract

All providers inherit from `EmbeddingModel` (base.py:16) and must:

1. **Implement abstract methods**:
   - `embed()`: Synchronous embedding generation
   - `aembed()`: Async embedding generation
   - `_get_models()`: Return list of available models
   - `_get_default_model()`: Return default model name
   - `provider` property: Return provider name string

2. **Override `__post_init__()`**:
   - Call `super().__post_init__()` first (extracts task-aware settings)
   - Set `api_key` from parameter or environment variable
   - Set `base_url` (use default if not provided)
   - Call `self._create_http_clients()` last (for API-based providers)

3. **Handle Advanced Features**:
   - Task-aware embeddings via `self.task_type` (EmbeddingTaskType enum)
   - Late chunking via `self.late_chunking` boolean
   - Output dimensions via `self.output_dimensions`
   - Truncation control via `self.truncate_at_max_length`

### Task-Aware Embeddings

Providers handle `task_type` differently:

- **Native support** (Jina, Voyage, Google): Pass task directly to API
- **Prefix-based** (others): Use `_apply_task_optimization()` to add prefixes like "query: " or "passage: "
- **No support**: Override `_apply_task_optimization()` to return texts unchanged

Set `SUPPORTED_FEATURES` class attribute to list which features are supported:

```python
class JinaEmbeddingModel(EmbeddingModel):
    SUPPORTED_FEATURES = ["task_type", "late_chunking"]
```

### Task Type Conversion

Google uses different task names. Implement `_map_task_to_google_task()` or similar:

- Esperanto `RETRIEVAL_QUERY` → Google `RETRIEVAL_QUERY`
- Esperanto `RETRIEVAL_DOCUMENT` → Google `RETRIEVAL_DOCUMENT`
- Esperanto `SIMILARITY` → Google `SEMANTIC_SIMILARITY`

### Text Preprocessing

Base class provides:

- `_clean_text()`: Normalize spacing, remove extra punctuation (base.py:101)
- `_apply_task_optimization()`: Add task-specific prefixes (base.py:128)
- `_apply_late_chunking()`: Simple sentence-based chunking (base.py:161)

Providers with native support should override to skip preprocessing.

### Configuration Serialization

When passing config to APIs:

- Use `_serialize_config_for_api()` to convert enums to strings (base.py:218)
- Use `_filter_unsupported_params()` to remove features the provider doesn't support (base.py:238)
- Use `_get_api_kwargs()` for clean kwargs dict (base.py:263)

### HTTP Client Pattern

Same as LLM providers:

```python
def __post_init__(self):
    super().__post_init__()  # Extracts task_type, late_chunking, etc.
    self.api_key = self.api_key or os.getenv("PROVIDER_API_KEY")
    self.base_url = self.base_url or "https://api.provider.com/v1"
    self._create_http_clients()
```

### Local vs API Providers

- **API providers** (OpenAI, Jina, Voyage): Use httpx clients, make HTTP requests
- **Local providers** (Transformers, Ollama local): May use local libraries instead
  - Transformers: Uses HuggingFace `sentence-transformers` library
  - Ollama: Can use HTTP (if remote) or local client

### Batch Processing

Most embedding APIs have batch limits:

- OpenAI: 2048 texts per request
- Google: 100-250 texts per request
- Jina: 8192 texts per request

Implement batching in `embed()` and `aembed()`:

```python
def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
    batch_size = 2048
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Process batch...
        all_embeddings.extend(batch_embeddings)
    return all_embeddings
```

## Integration

- Imported by `factory.py` via `AIFactory._provider_modules["embedding"]`
- Uses types from `esperanto.common_types` (Model)
- Uses `EmbeddingTaskType` enum from `esperanto.common_types.task_type`
- Inherits mixins from `esperanto.utils.timeout` and `esperanto.utils.ssl`

## Gotchas

- **Task type extraction**: Base `__post_init__()` automatically extracts `task_type` from config and converts strings to enum
- **Task type string conversion**: Config may have `"retrieval.query"` (with dot) or `"retrieval_query"` (underscore) - base class handles both
- **Feature filtering**: If a provider doesn't declare `SUPPORTED_FEATURES`, all advanced features are removed by `_filter_unsupported_params()`
- **Text cleaning**: Only apply if provider doesn't handle it - avoid double-processing
- **Empty texts**: Handle empty string lists gracefully (return empty list)
- **Dimension validation**: If `output_dimensions` is set but provider doesn't support it, ignore or raise error
- **API key optional**: Local providers (Transformers, local Ollama) don't need API keys
- **Model loading**: Transformers provider needs to download models on first use - handle this gracefully
- **Async implementation**: For local models (Transformers), don't spawn threads - use asyncio properly or run in executor
- **Deprecation warnings**: Use `_get_models()` internally (not `.models` property)

## When Adding a New Provider

1. Create new file `provider_name.py`
2. Import `EmbeddingModel` from `esperanto.providers.embedding.base`
3. Define class inheriting from `EmbeddingModel`
4. Declare `SUPPORTED_FEATURES` class attribute (if provider supports advanced features)
5. Implement all abstract methods
6. Add `__post_init__()` following the pattern
7. Implement batching for API efficiency
8. Add provider to `factory.py` in `_provider_modules["embedding"]` dict
9. Add optional import in `src/esperanto/__init__.py`
10. Write tests in `tests/providers/embedding/test_provider_name.py`
11. Add documentation in `docs/providers/provider_name.md`

## Special Cases

### Transformers Provider

- Uses HuggingFace `sentence-transformers` library
- Downloads models to cache on first use
- Supports advanced features like late chunking natively
- Can run on GPU if available (check `device` parameter)
- Model names are HuggingFace model IDs (e.g., "sentence-transformers/all-MiniLM-L6-v2")

### OpenAI-Compatible Provider

- Generic provider for any OpenAI-compatible API
- Requires explicit `base_url` parameter
- Useful for self-hosted models (vLLM, text-embeddings-inference, etc.)
- Configure via environment variables with `OPENAI_COMPATIBLE_` prefix

### Task Type Support Matrix

| Provider | task_type | late_chunking | output_dimensions |
|----------|-----------|---------------|-------------------|
| Jina | ✓ | ✓ | ✗ |
| Voyage | ✓ | ✗ | ✗ |
| Google | ✓ | ✗ | ✗ |
| Transformers | ✓ | ✓ | ✗ |
| OpenAI | ✗ | ✗ | ✓ |
| Azure | ✗ | ✗ | ✓ |
| Mistral | ✗ | ✗ | ✗ |
| Others | ✗ | ✗ | ✗ |

Providers without native support use prefix-based task optimization from base class.
