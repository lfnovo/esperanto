# Utilities

Utility modules providing cross-cutting functionality for all providers.

## Files

- **`connect.py`**: `HttpConnectionMixin` combining timeout, SSL, and proxy configuration for HTTP clients
- **`timeout.py`**: `TimeoutMixin` for configurable HTTP request timeouts
- **`ssl.py`**: `SSLMixin` for configurable SSL verification
- **`model_cache.py`**: `ModelCache` for caching provider model lists with TTL
- **`logging.py`**: Logger configuration for Esperanto

## Patterns

### Mixin Architecture

`TimeoutMixin`, `SSLMixin`, and `HttpConnectionMixin` use the mixin pattern:

- Inherit alongside provider base class
- Provide configuration methods called by base class
- Implement priority-based configuration (config > env var > default)

Example inheritance:

```python
class LanguageModel(TimeoutMixin, SSLMixin, ABC):
    pass
```

### Three-Tier Priority System

Both mixins implement configuration priority hierarchy:

1. **Config dict** (highest priority): `config={"timeout": 120}`
2. **Environment variable**: `ESPERANTO_LLM_TIMEOUT=90`
3. **Default value** (lowest priority): Provider type default

### TimeoutMixin

Provides timeout configuration for HTTP requests (timeout.py:26).

**Provider Integration:**

Providers must:

1. Inherit from `TimeoutMixin`
2. Implement `_get_provider_type()` returning provider type string
3. Call `self._get_timeout()` when creating HTTP clients

**Provider Types and Defaults:**

- `"language"`: 60 seconds
- `"embedding"`: 60 seconds
- `"reranker"`: 60 seconds
- `"speech_to_text"`: 300 seconds (file processing)
- `"text_to_speech"`: 300 seconds (file generation)

**Environment Variables:**

- `ESPERANTO_LLM_TIMEOUT`: Language model timeout
- `ESPERANTO_EMBEDDING_TIMEOUT`: Embedding model timeout
- `ESPERANTO_RERANKER_TIMEOUT`: Reranker timeout
- `ESPERANTO_STT_TIMEOUT`: Speech-to-text timeout
- `ESPERANTO_TTS_TIMEOUT`: Text-to-speech timeout

**Usage in Providers:**

```python
def _create_http_clients(self):
    import httpx
    timeout = self._get_timeout()  # From TimeoutMixin
    self.client = httpx.Client(timeout=timeout)
    self.async_client = httpx.AsyncClient(timeout=timeout)
```

**Validation:**

- Must be numeric (int or float)
- Must be positive (> 0)
- Must be finite (not inf/nan)
- Maximum: 600 seconds (10 minutes)

### SSLMixin

Provides SSL verification configuration (ssl.py:12).

**Configuration Options:**

1. **Enable verification** (default): `verify_ssl=True`
2. **Disable verification** (insecure): `verify_ssl=False`
3. **Custom CA bundle**: `ssl_ca_bundle="/path/to/ca.pem"`

**Priority Hierarchy:**

1. Config dict `ssl_ca_bundle` (highest)
2. Config dict `verify_ssl`
3. Env var `ESPERANTO_SSL_CA_BUNDLE`
4. Env var `ESPERANTO_SSL_VERIFY`
5. Default: `True` (lowest)

**Usage in Providers:**

```python
def _create_http_clients(self):
    import httpx
    verify = self._get_ssl_verify()  # From SSLMixin
    self.client = httpx.Client(verify=verify)
    self.async_client = httpx.AsyncClient(verify=verify)
```

**Return Types:**

- `True`: SSL verification enabled with default CA bundle
- `False`: SSL verification disabled (emits warning)
- `str`: Path to custom CA bundle file

**Warnings:**

Disabling SSL verification emits a warning:

```
UserWarning: SSL verification is disabled. This is insecure and should only be used for development.
```

### HttpConnectionMixin (connect.py)

Central mixin that combines `TimeoutMixin` and `SSLMixin`, adding proxy support and HTTP client lifecycle management.

**Provides:**

- `_get_proxy()`: Get proxy URL from config or environment
- `_create_http_clients()`: Create httpx clients with timeout, SSL, and proxy settings
- `close()` / `aclose()`: Explicit client cleanup
- Context manager support (`with` / `async with`)
- Destructor cleanup

**Proxy Configuration:**

Priority order:
1. Config dict `proxy` (highest)
2. Env var `ESPERANTO_PROXY`
3. Default: `None` (no proxy)

**Usage:**

```python
# Via environment variable
os.environ["ESPERANTO_PROXY"] = "http://proxy.example.com:8080"
model = AIFactory.create_language("openai", "gpt-4")

# Via config dict
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"proxy": "http://proxy.example.com:8080"}
)
```

**Proxy URL Formats:**

```python
# HTTP proxy
"http://proxy.example.com:8080"

# HTTPS proxy
"https://secure-proxy.example.com:443"

# With authentication
"http://user:pass@proxy.example.com:8080"
```

### ModelCache

Thread-safe cache for provider model lists (model_cache.py:32).

**Purpose:**

- Reduce unnecessary API calls to model listing endpoints
- Cache model lists per provider with configurable TTL
- Thread-safe for concurrent access

**Usage:**

```python
from esperanto.utils.model_cache import ModelCache

cache = ModelCache()

# Try to get from cache
models = cache.get("openai:api_key_hash")
if models is None:
    # Cache miss - fetch from API
    models = fetch_models_from_api()
    cache.set("openai:api_key_hash", models, ttl=3600)

return models
```

**Cache Key Format:**

Typically: `"{provider}:{api_key_hash}"`

Different API keys may have access to different models (e.g., organization-specific models).

**TTL Behavior:**

- Default: 3600 seconds (1 hour)
- Expired entries automatically removed on access
- Manual invalidation via `invalidate(key)` or `clear()`

**Thread Safety:**

Uses `threading.Lock()` for all operations:

- `get()`: Read with automatic expiration cleanup
- `set()`: Write with TTL
- `clear()`: Remove all entries
- `invalidate()`: Remove specific entry

## Integration

### Mixin Usage in Base Classes

All provider base classes inherit both mixins:

```python
from esperanto.utils.ssl import SSLMixin
from esperanto.utils.timeout import TimeoutMixin

@dataclass
class LanguageModel(TimeoutMixin, SSLMixin, ABC):
    def _get_provider_type(self) -> str:
        return "language"

    def _create_http_clients(self):
        import httpx
        timeout = self._get_timeout()
        verify = self._get_ssl_verify()
        self.client = httpx.Client(timeout=timeout, verify=verify)
        self.async_client = httpx.AsyncClient(timeout=timeout, verify=verify)
```

### Model Discovery Integration

Model discovery functions use `ModelCache` to avoid redundant API calls:

```python
from esperanto.utils.model_cache import ModelCache

_model_cache = ModelCache()

def discover_openai_models(api_key: str = None) -> List[Model]:
    cache_key = f"openai:{hash(api_key)}"

    # Try cache first
    cached = _model_cache.get(cache_key)
    if cached:
        return cached

    # Fetch from API
    models = fetch_from_openai_api(api_key)

    # Cache for 1 hour
    _model_cache.set(cache_key, models, ttl=3600)

    return models
```

## Gotchas

### TimeoutMixin

- **Must implement `_get_provider_type()`**: Abstract method - raises error if not implemented
- **Timeout applies to entire request**: Including connection + read time
- **No infinite timeout**: Maximum 600 seconds enforced
- **Environment variable format**: Must be numeric string (e.g., "120" not "2m")
- **Provider type mismatch**: Using wrong type returns wrong default timeout
- **Validation errors**: Invalid timeouts raise ValueError with clear message

### SSLMixin

- **Security warning**: Disabling SSL emits UserWarning - intentional, don't suppress
- **CA bundle validation**: Path must exist - checked on configuration
- **Development only**: SSL disabled should never be used in production
- **Custom CA bundles**: For self-signed certs - recommended over disabling SSL
- **Environment variable format**: Use "false"/"0"/"no" to disable (case-insensitive)
- **Return type variation**: Can be bool OR str - handle both when using

### ModelCache

- **Not persistent**: In-memory only - cleared on process restart
- **Thread-safe only**: Not process-safe - don't share across processes
- **TTL precision**: Uses `time.time()` - second precision
- **Memory growth**: No max size - can grow unbounded if many providers/keys used
- **Expiration check**: Happens on access - expired entries stay in memory until accessed
- **Cache key collisions**: Use unique keys (include API key hash)

## When Adding New Mixins

If creating a new mixin:

1. Create new file in this directory
2. Define mixin class with descriptive name
3. Document priority hierarchy clearly
4. Implement validation for configuration values
5. Add to all relevant base classes
6. Update this CLAUDE.md file
7. Add tests for priority hierarchy
8. Document in main docs if user-facing

## Configuration Examples

### Setting Timeouts

```python
# Via config dict
model = AIFactory.create_language("openai", "gpt-4", config={"timeout": 120})

# Via environment variable
import os
os.environ["ESPERANTO_LLM_TIMEOUT"] = "90"
model = AIFactory.create_language("openai", "gpt-4")

# Default (60 seconds for LLM)
model = AIFactory.create_language("openai", "gpt-4")
```

### Configuring SSL

```python
# Disable SSL (development only)
model = AIFactory.create_language(
    "ollama",
    "llama3",
    config={"verify_ssl": False}
)

# Custom CA bundle (recommended for self-signed certs)
model = AIFactory.create_language(
    "ollama",
    "llama3",
    config={"ssl_ca_bundle": "/path/to/ca-bundle.pem"}
)

# Via environment variable
import os
os.environ["ESPERANTO_SSL_VERIFY"] = "false"
model = AIFactory.create_language("ollama", "llama3")
```

### Configuring Proxy

```python
# Via environment variable (recommended)
import os
os.environ["ESPERANTO_PROXY"] = "http://proxy.example.com:8080"
model = AIFactory.create_language("openai", "gpt-4")

# Via config dict
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"proxy": "http://proxy.example.com:8080"}
)

# With authentication
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"proxy": "http://user:pass@proxy.example.com:8080"}
)
```

### Using Model Cache

```python
from esperanto.utils.model_cache import ModelCache

# Global cache instance (typically in model_discovery.py)
_cache = ModelCache()

def get_models(provider: str, api_key: str) -> List[Model]:
    cache_key = f"{provider}:{hash(api_key)}"

    # Try cache
    models = _cache.get(cache_key)
    if models:
        return models

    # Fetch and cache
    models = api_fetch(provider, api_key)
    _cache.set(cache_key, models, ttl=3600)
    return models
```
