# Model Discovery

## Overview

Esperanto provides a convenient way to discover available models from providers without creating instances. This allows you to explore what models are available, check their capabilities, and make informed decisions about which models to use in your applications.

The model discovery feature is available through the static `AIFactory.get_provider_models()` method, which queries providers for their available models and returns structured information about each one.

## Quick Start

```python
from esperanto.factory import AIFactory

# Discover available models from OpenAI
models = AIFactory.get_provider_models("openai", api_key="your-api-key")

for model in models:
    print(f"{model.id} - owned by {model.owned_by}")
```

## Static vs Instance-Based Discovery

### Static Discovery (Recommended)

The recommended approach uses the static factory method:

```python
# ✅ Recommended - Static discovery
models = AIFactory.get_provider_models("openai", api_key="your-api-key")
```

**Benefits:**
- No instance creation required
- Cleaner, more straightforward API
- Consistent across all providers
- Results are cached for performance

### Instance-Based Discovery (Deprecated)

The older approach used the `.models` property on provider instances:

```python
# ❌ Deprecated - Will be removed in version 3.0
model = AIFactory.create_language("openai", "gpt-4", api_key="your-api-key")
models = model.models
```

**Note:** The `.models` property on provider instances is deprecated and will be removed in version 3.0. Please migrate to the static `get_provider_models()` method.

## Benefits of Static Discovery

### 1. No Instance Creation Required

Query models without setting up providers:

```python
# No need to create a model instance first
models = AIFactory.get_provider_models("openai", api_key="your-api-key")

# vs the old way (deprecated)
model_instance = AIFactory.create_language("openai", "gpt-4", api_key="your-api-key")
models = model_instance.models  # Deprecated
```

### 2. Cached Results

Model lists are cached for 1 hour to reduce API calls:

```python
# First call - queries the API
models1 = AIFactory.get_provider_models("openai", api_key="your-api-key")

# Second call within 1 hour - returns cached result (instant!)
models2 = AIFactory.get_provider_models("openai", api_key="your-api-key")
```

### 3. Flexible Configuration

Pass provider-specific configuration:

```python
# OpenAI with API key
openai_models = AIFactory.get_provider_models(
    "openai",
    api_key="your-api-key"
)

# OpenAI-compatible with custom base URL
local_models = AIFactory.get_provider_models(
    "openai-compatible",
    base_url="http://localhost:1234/v1"
)

# Google with API key
google_models = AIFactory.get_provider_models(
    "google",
    api_key="your-api-key"
)
```

### 4. Type Filtering

Filter models by type for multi-capability providers:

```python
# Get only language models
language_models = AIFactory.get_provider_models(
    "openai",
    api_key="your-api-key",
    model_type="language"
)

# Get only embedding models
embedding_models = AIFactory.get_provider_models(
    "openai",
    api_key="your-api-key",
    model_type="embedding"
)

# Get only speech-to-text models
stt_models = AIFactory.get_provider_models(
    "openai",
    api_key="your-api-key",
    model_type="speech_to_text"
)

# Get only text-to-speech models
tts_models = AIFactory.get_provider_models(
    "openai",
    api_key="your-api-key",
    model_type="text_to_speech"
)
```

## Supported Providers

### OpenAI

Fetches models via API with type filtering support:

```python
models = AIFactory.get_provider_models("openai", api_key="your-api-key")

# Example output:
# gpt-4-turbo - owned by openai
# gpt-4 - owned by openai
# gpt-3.5-turbo - owned by openai
# text-embedding-3-small - owned by openai
# whisper-1 - owned by openai

# Filter by type
language_models = AIFactory.get_provider_models(
    "openai",
    api_key="your-api-key",
    model_type="language"
)
# Only returns: gpt-4-turbo, gpt-4, gpt-3.5-turbo, etc.
```

### OpenAI-Compatible Endpoints

Fetches models from any OpenAI-compatible endpoint:

```python
# LM Studio
lm_studio_models = AIFactory.get_provider_models(
    "openai-compatible",
    base_url="http://localhost:1234/v1"
)

# vLLM
vllm_models = AIFactory.get_provider_models(
    "openai-compatible",
    base_url="http://localhost:8000/v1"
)

# LocalAI
localai_models = AIFactory.get_provider_models(
    "openai-compatible",
    base_url="http://localhost:8080/v1"
)

for model in lm_studio_models:
    print(f"{model.id} - {model.owned_by}")
```

### Anthropic (Claude)

Returns hardcoded list of Claude models:

```python
models = AIFactory.get_provider_models("anthropic")

for model in models:
    print(f"{model.id} - Context: {model.context_window} tokens")

# Example output:
# claude-3-5-sonnet-20241022 - Context: 200000 tokens
# claude-3-5-haiku-20241022 - Context: 200000 tokens
# claude-3-opus-20240229 - Context: 200000 tokens
```

### Google/Gemini

Fetches models via API:

```python
models = AIFactory.get_provider_models("google", api_key="your-api-key")

for model in models:
    print(f"{model.id}")

# Example output:
# gemini-1.5-pro
# gemini-1.5-flash
# text-embedding-004
```

### Groq

Fetches models via API:

```python
models = AIFactory.get_provider_models("groq", api_key="your-api-key")

for model in models:
    print(f"{model.id}")

# Example output:
# mixtral-8x7b-32768
# llama3-70b-8192
# llama3-8b-8192
```

### Mistral

Fetches models via API:

```python
models = AIFactory.get_provider_models("mistral", api_key="your-api-key")

for model in models:
    print(f"{model.id}")

# Example output:
# mistral-large-latest
# mistral-medium-latest
# mistral-small-latest
```

### Ollama

Fetches locally available models:

```python
models = AIFactory.get_provider_models(
    "ollama",
    base_url="http://localhost:11434"  # Optional, defaults to localhost:11434
)

for model in models:
    print(f"{model.id}")

# Example output (your local models):
# llama3.2
# codellama
# mistral
```

### Jina

Returns hardcoded list of embedding and reranking models:

```python
models = AIFactory.get_provider_models("jina")

for model in models:
    print(f"{model.id}")

# Example output:
# jina-embeddings-v3
# jina-embeddings-v2-base-en
# jina-reranker-v2-base-multilingual
```

### Voyage

Returns hardcoded list of embedding and reranking models:

```python
models = AIFactory.get_provider_models("voyage")

for model in models:
    print(f"{model.id}")

# Example output:
# voyage-3
# voyage-3-lite
# voyage-code-3
# rerank-2
# rerank-2-lite
```

### Cohere

Fetches models via API (language and embedding):

```python
models = AIFactory.get_provider_models("cohere", api_key="your-api-key")

for model in models:
    print(f"{model.id}")
```

### Together

Fetches models via API:

```python
models = AIFactory.get_provider_models("together", api_key="your-api-key")

for model in models:
    print(f"{model.id}")
```

## Provider Support Matrix

| Provider | Discovery Method | Type Filtering | Authentication Required |
|----------|------------------|----------------|-------------------------|
| **OpenAI** | API | ✅ Yes | ✅ API Key |
| **OpenAI-Compatible** | API | ❌ No | ⚠️ Optional |
| **Anthropic** | Hardcoded | ❌ No | ❌ No |
| **Google** | API | ❌ No | ✅ API Key |
| **Groq** | API | ❌ No | ✅ API Key |
| **Mistral** | API | ❌ No | ✅ API Key |
| **Ollama** | API (Local) | ❌ No | ❌ No |
| **Jina** | Hardcoded | ❌ No | ❌ No |
| **Voyage** | Hardcoded | ❌ No | ❌ No |
| **Cohere** | API | ❌ No | ✅ API Key |
| **Together** | API | ❌ No | ✅ API Key |

## Use Cases

### Model Selection UI

Build a UI for users to select models:

```python
def get_available_models(provider, api_key=None):
    """Get available models for UI dropdown"""
    try:
        models = AIFactory.get_provider_models(
            provider,
            api_key=api_key
        )
        return [{"id": m.id, "name": m.id} for m in models]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

# Usage in a web app
@app.route("/api/models")
def list_models():
    provider = request.args.get("provider")
    api_key = request.headers.get("X-API-Key")

    models = get_available_models(provider, api_key)
    return jsonify(models)
```

### Feature Detection

Check if a provider supports specific models:

```python
def supports_model(provider, model_id, api_key=None):
    """Check if provider supports a specific model"""
    models = AIFactory.get_provider_models(provider, api_key=api_key)
    return any(m.id == model_id for m in models)

# Usage
if supports_model("openai", "gpt-4", api_key="your-key"):
    print("GPT-4 is available!")
else:
    print("GPT-4 is not available, falling back...")
```

### Model Comparison

Compare available models across providers:

```python
def compare_providers(providers, api_keys):
    """Compare models across multiple providers"""
    results = {}

    for provider in providers:
        api_key = api_keys.get(provider)
        try:
            models = AIFactory.get_provider_models(provider, api_key=api_key)
            results[provider] = [m.id for m in models]
        except Exception as e:
            results[provider] = f"Error: {e}"

    return results

# Usage
providers = ["openai", "anthropic", "google"]
api_keys = {
    "openai": "sk-...",
    "google": "AIza...",
    # anthropic doesn't need a key for model discovery
}

comparison = compare_providers(providers, api_keys)
for provider, models in comparison.items():
    print(f"\n{provider}:")
    if isinstance(models, str):
        print(f"  {models}")
    else:
        for model in models:
            print(f"  - {model}")
```

### Automated Model Selection

Automatically select the best available model:

```python
def get_best_model(provider, preferred_models, api_key=None):
    """
    Get the best available model from a list of preferences

    Args:
        provider: Provider name
        preferred_models: List of model IDs in order of preference
        api_key: Optional API key
    """
    available_models = AIFactory.get_provider_models(provider, api_key=api_key)
    available_ids = {m.id for m in available_models}

    for preferred in preferred_models:
        if preferred in available_ids:
            return preferred

    # Fallback to first available model
    return available_models[0].id if available_models else None

# Usage
preferred = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
model_id = get_best_model("openai", preferred, api_key="your-key")

if model_id:
    model = AIFactory.create_language("openai", model_id)
    print(f"Using model: {model_id}")
else:
    print("No models available!")
```

### Local Model Detection

Check what models are installed locally:

```python
def get_local_ollama_models():
    """Get list of locally installed Ollama models"""
    try:
        models = AIFactory.get_provider_models("ollama")
        return [m.id for m in models]
    except Exception as e:
        print(f"Ollama not running or error: {e}")
        return []

# Usage
local_models = get_local_ollama_models()
if local_models:
    print("Available local models:")
    for model in local_models:
        print(f"  - {model}")
else:
    print("No local models found. Please pull some models with 'ollama pull'")
```

### Configuration Validation

Validate configuration before deployment:

```python
def validate_deployment_config(config):
    """Validate that all configured models are available"""
    errors = []

    for service, settings in config.items():
        provider = settings["provider"]
        model_id = settings["model"]
        api_key = settings.get("api_key")

        try:
            models = AIFactory.get_provider_models(provider, api_key=api_key)
            available_ids = {m.id for m in models}

            if model_id not in available_ids:
                errors.append(
                    f"{service}: Model '{model_id}' not available in {provider}"
                )
        except Exception as e:
            errors.append(f"{service}: Error checking {provider} - {e}")

    return errors

# Usage
deployment_config = {
    "chat_service": {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "sk-..."
    },
    "embedding_service": {
        "provider": "voyage",
        "model": "voyage-3"
    }
}

errors = validate_deployment_config(deployment_config)
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration validated successfully!")
```

## Advanced Features

### Model Metadata

Access detailed model information (when available):

```python
models = AIFactory.get_provider_models("openai", api_key="your-api-key")

for model in models:
    print(f"ID: {model.id}")
    if hasattr(model, 'owned_by'):
        print(f"  Owner: {model.owned_by}")
    if hasattr(model, 'context_window'):
        print(f"  Context: {model.context_window} tokens")
    if hasattr(model, 'created'):
        print(f"  Created: {model.created}")
```

### Filtering and Sorting

Filter and sort discovered models:

```python
def get_sorted_models(provider, api_key=None, filter_fn=None, sort_key=None):
    """Get models with custom filtering and sorting"""
    models = AIFactory.get_provider_models(provider, api_key=api_key)

    # Apply filter
    if filter_fn:
        models = [m for m in models if filter_fn(m)]

    # Apply sort
    if sort_key:
        models.sort(key=sort_key)

    return models

# Get only GPT-4 models, sorted by name
gpt4_models = get_sorted_models(
    "openai",
    api_key="your-key",
    filter_fn=lambda m: "gpt-4" in m.id,
    sort_key=lambda m: m.id
)

# Get embedding models
embedding_models = get_sorted_models(
    "openai",
    api_key="your-key",
    filter_fn=lambda m: "embedding" in m.id
)
```

### Caching Strategy

The built-in cache expires after 1 hour. For custom caching:

```python
from functools import lru_cache
from datetime import datetime, timedelta

class ModelCache:
    def __init__(self, ttl_minutes=60):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)

    def get_models(self, provider, **kwargs):
        cache_key = f"{provider}:{str(kwargs)}"

        # Check cache
        if cache_key in self.cache:
            models, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                return models

        # Fetch fresh data
        models = AIFactory.get_provider_models(provider, **kwargs)

        # Update cache
        self.cache[cache_key] = (models, datetime.now())

        return models

# Usage
cache = ModelCache(ttl_minutes=30)  # Custom 30-minute cache
models = cache.get_models("openai", api_key="your-key")
```

## Error Handling

### Handle Missing Credentials

```python
def safe_get_models(provider, api_key=None):
    """Safely get models with error handling"""
    try:
        return AIFactory.get_provider_models(provider, api_key=api_key)
    except Exception as e:
        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
            print(f"Authentication error for {provider}: {e}")
            return []
        else:
            print(f"Error fetching models from {provider}: {e}")
            return []

# Usage
models = safe_get_models("openai", api_key="invalid-key")
if not models:
    print("Using fallback configuration...")
```

### Retry Logic

```python
import time

def get_models_with_retry(provider, max_retries=3, **kwargs):
    """Get models with retry on failure"""
    for attempt in range(max_retries):
        try:
            return AIFactory.get_provider_models(provider, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

# Usage
models = get_models_with_retry("openai", api_key="your-key", max_retries=3)
```

## Best Practices

### 1. Use Static Discovery

```python
# ✅ Good - Use static method
models = AIFactory.get_provider_models("openai", api_key="your-key")

# ❌ Bad - Using deprecated instance property
instance = AIFactory.create_language("openai", "gpt-4")
models = instance.models  # Deprecated
```

### 2. Handle Errors Gracefully

```python
try:
    models = AIFactory.get_provider_models("openai", api_key="your-key")
except Exception as e:
    print(f"Error: {e}")
    models = []  # Fallback to empty list
```

### 3. Cache Results Appropriately

```python
# The built-in cache handles this automatically
# But for custom TTL, implement your own caching
```

### 4. Validate Before Use

```python
models = AIFactory.get_provider_models("openai", api_key="your-key")
if not models:
    raise ValueError("No models available")

model_ids = {m.id for m in models}
if "gpt-4" not in model_ids:
    raise ValueError("Required model not available")
```

### 5. Document Provider Requirements

```python
def get_models(provider):
    """
    Get available models from provider

    Requirements:
    - OpenAI: Requires api_key
    - Anthropic: No API key needed for discovery
    - Ollama: Requires local Ollama installation
    - OpenAI-compatible: Requires base_url
    """
    # Implementation
    pass
```

## Migration Guide

If you're using the deprecated `.models` property:

### Before (Deprecated)

```python
model = AIFactory.create_language("openai", "gpt-4", api_key="your-key")
available_models = model.models
```

### After (Current)

```python
available_models = AIFactory.get_provider_models("openai", api_key="your-key")
```

## See Also

- [Language Model Capabilities](/docs/capabilities/language.md) - LLM features overview
- [Embedding Capabilities](/docs/capabilities/embeddings.md) - Embedding features
- [Provider Guides](/docs/providers/) - Provider-specific documentation
- [AIFactory Reference](/docs/api/factory.md) - Factory API documentation

