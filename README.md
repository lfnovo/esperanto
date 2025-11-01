# Esperanto 🌐

[![PyPI version](https://badge.fury.io/py/esperanto.svg)](https://badge.fury.io/py/esperanto)
[![PyPI Downloads](https://img.shields.io/pypi/dm/esperanto)](https://pypi.org/project/esperanto/)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](https://github.com/lfnovo/esperanto)
[![Python Versions](https://img.shields.io/pypi/pyversions/esperanto)](https://pypi.org/project/esperanto/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Esperanto is a powerful Python library that provides a unified interface for interacting with various Large Language Model (LLM) providers. It simplifies the process of working with different AI models (LLMs, Embedders, Transcribers, and TTS) APIs by offering a consistent interface while maintaining provider-specific optimizations.

## Why Esperanto? 🚀

**🪶 Ultra-Lightweight Architecture**
- **Direct HTTP Communication**: All providers communicate directly via HTTP APIs using `httpx` - no bulky vendor SDKs required
- **Minimal Dependencies**: Unlike LangChain and similar frameworks, Esperanto has a tiny footprint with zero overhead layers
- **Production-Ready Performance**: Direct API calls mean faster response times and lower memory usage

**🔄 True Provider Flexibility**
- **Standardized Responses**: Switch between any provider (OpenAI ↔ Anthropic ↔ Google ↔ etc.) without changing a single line of code
- **Consistent Interface**: Same methods, same response objects, same patterns across all 15+ providers
- **Future-Proof**: Add new providers or change existing ones without refactoring your application

**⚡ Perfect for Production**
- **Prototyping to Production**: Start experimenting and deploy the same code to production
- **No Vendor Lock-in**: Test different providers, optimize costs, and maintain flexibility
- **Enterprise-Ready**: Direct HTTP calls, standardized error handling, and comprehensive async support

Whether you're building a quick prototype or a production application serving millions of requests, Esperanto gives you the performance of direct API calls with the convenience of a unified interface.

## Features ✨

- **Unified Interface**: Work with multiple LLM providers using a consistent API
- **Provider Support**:
  - OpenAI (GPT-4o, o1, o3, o4, Whisper, TTS)
  - OpenAI-Compatible (LM Studio, Ollama, vLLM, custom endpoints)
  - Anthropic (Claude models)
  - OpenRouter (Access to multiple models)
  - xAI (Grok)
  - Perplexity (Sonar models)
  - Groq (Mixtral, Llama, Whisper)
  - Google GenAI (Gemini LLM, Text To Speech, Embedding with native task optimization)
  - Vertex AI (Google Cloud, LLM, Embedding, TTS)
  - Ollama (Local deployment multiple models)
  - Transformers (Universal local models - Qwen, CrossEncoder, BAAI, Jina, Mixedbread)
  - ElevenLabs (Text-to-Speech, Speech-to-Text)
  - Azure OpenAI (Chat, Embedding, Whisper, TTS)
  - Mistral (Mistral Large, Small, Embedding, etc.)
  - DeepSeek (deepseek-chat)
  - Voyage (Embeddings, Reranking)
  - Jina (Advanced embedding models with task optimization, Reranking)
- **Embedding Support**: Multiple embedding providers for vector representations
- **Reranking Support**: Universal reranking interface for improving search relevance
- **Speech-to-Text Support**: Transcribe audio using multiple providers
- **Text-to-Speech Support**: Generate speech using multiple providers
- **Async Support**: Both synchronous and asynchronous API calls
- **Streaming**: Support for streaming responses
- **Structured Output**: JSON output formatting (where supported)
- **LangChain Integration**: Easy conversion to LangChain chat models

## 📚 Documentation

- **[Quick Start Guide](https://github.com/lfnovo/esperanto/blob/main/docs/quickstart.md)** - Get started in 5 minutes
- **[Documentation Index](https://github.com/lfnovo/esperanto/blob/main/docs/README.md)** - Complete documentation hub
- **[Provider Comparison](https://github.com/lfnovo/esperanto/blob/main/docs/providers/README.md)** - Choose the right provider
- **[Configuration Guide](https://github.com/lfnovo/esperanto/blob/main/docs/configuration.md)** - Environment setup

### By Capability
- [Language Models (LLM)](https://github.com/lfnovo/esperanto/blob/main/docs/capabilities/llm.md) - Text generation and chat
- [Embeddings](https://github.com/lfnovo/esperanto/blob/main/docs/capabilities/embedding.md) - Vector representations
- [Reranking](https://github.com/lfnovo/esperanto/blob/main/docs/capabilities/reranking.md) - Search relevance
- [Speech-to-Text](https://github.com/lfnovo/esperanto/blob/main/docs/capabilities/speech-to-text.md) - Audio transcription
- [Text-to-Speech](https://github.com/lfnovo/esperanto/blob/main/docs/capabilities/text-to-speech.md) - Voice generation

### By Provider
- [Provider Setup Guides](https://github.com/lfnovo/esperanto/blob/main/docs/providers/) - Complete setup for all 17 providers

### Advanced Topics
- [Task-Aware Embeddings](https://github.com/lfnovo/esperanto/blob/main/docs/advanced/task-aware-embeddings.md)
- [LangChain Integration](https://github.com/lfnovo/esperanto/blob/main/docs/advanced/langchain-integration.md)
- [Timeout Configuration](https://github.com/lfnovo/esperanto/blob/main/docs/advanced/timeout-configuration.md)
- [Model Discovery](https://github.com/lfnovo/esperanto/blob/main/docs/advanced/model-discovery.md)
- [Transformers Features](https://github.com/lfnovo/esperanto/blob/main/docs/advanced/transformers-features.md)

**[CHANGELOG](https://github.com/lfnovo/esperanto/blob/main/CHANGELOG.md)** - Version history and migration guides

## Installation 🚀

Install Esperanto using pip:

```bash
pip install esperanto
```

### Optional Dependencies

**Transformers Provider**

If you plan to use the transformers provider, install with the transformers extra:

```bash
pip install "esperanto[transformers]"
```

This installs:
- `transformers` - Core Hugging Face library
- `torch` - PyTorch framework
- `tokenizers` - Fast tokenization
- `sentence-transformers` - CrossEncoder support
- `scikit-learn` - Advanced embedding features
- `numpy` - Numerical computations

**LangChain Integration**

If you plan to use any of the `.to_langchain()` methods, you need to install the correct LangChain SDKs manually:

```bash
# Core LangChain dependencies (required)
pip install "langchain>=0.3.8,<0.4.0" "langchain-core>=0.3.29,<0.4.0"

# Provider-specific LangChain packages (install only what you need)
pip install "langchain-openai>=0.2.9"
pip install "langchain-anthropic>=0.3.0"
pip install "langchain-google-genai>=2.1.2"
pip install "langchain-ollama>=0.2.0"
pip install "langchain-groq>=0.2.1"
pip install "langchain_mistralai>=0.2.1"
pip install "langchain_deepseek>=0.1.3"
pip install "langchain-google-vertexai>=2.0.24"
```

## Provider Support Matrix

| Provider     | LLM Support | Embedding Support | Reranking Support | Speech-to-Text | Text-to-Speech | JSON Mode |
|--------------|-------------|------------------|-------------------|----------------|----------------|-----------|
| OpenAI       | ✅          | ✅               | ❌                | ✅             | ✅             | ✅        |
| OpenAI-Compatible | ✅          | ✅               | ❌                | ✅             | ✅             | ⚠️*       |
| Anthropic    | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |
| Groq         | ✅          | ❌               | ❌                | ✅             | ❌             | ✅        |
| Google (GenAI) | ✅          | ✅               | ❌                | ❌             | ✅             | ✅        |
| Vertex AI    | ✅          | ✅               | ❌                | ❌             | ✅             | ❌        |
| Ollama       | ✅          | ✅               | ❌                | ❌             | ❌             | ❌        |
| Perplexity   | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |
| Transformers | ❌          | ✅               | ✅                | ❌             | ❌             | ❌        |
| ElevenLabs   | ❌          | ❌               | ❌                | ✅             | ✅             | ❌        |
| Azure OpenAI | ✅          | ✅               | ❌                | ✅             | ✅             | ✅        |
| Mistral      | ✅          | ✅               | ❌                | ❌             | ❌             | ✅        |
| DeepSeek     | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |
| Voyage       | ❌          | ✅               | ✅                | ❌             | ❌             | ❌        |
| Jina         | ❌          | ✅               | ✅                | ❌             | ❌             | ❌        |
| xAI          | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |
| OpenRouter   | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |

*⚠️ OpenAI-Compatible: JSON mode support depends on the specific endpoint implementation

## Quick Start 🏃‍♂️

You can use Esperanto in two ways: directly with provider-specific classes or through the AI Factory.

### Using AI Factory

The AI Factory provides a convenient way to create model instances and discover available providers:

```python
from esperanto.factory import AIFactory

# Get available providers for each model type
providers = AIFactory.get_available_providers()
print(providers)
# Output:
# {
#     'language': ['openai', 'openai-compatible', 'anthropic', 'google', 'groq', 'ollama', 'openrouter', 'xai', 'perplexity', 'azure', 'mistral', 'deepseek'],
#     'embedding': ['openai', 'openai-compatible', 'google', 'ollama', 'vertex', 'transformers', 'voyage', 'mistral', 'azure', 'jina'],
#     'reranker': ['jina', 'voyage', 'transformers'],
#     'speech_to_text': ['openai', 'openai-compatible', 'groq', 'elevenlabs', 'azure'],
#     'text_to_speech': ['openai', 'openai-compatible', 'elevenlabs', 'google', 'vertex', 'azure']
# }

# Create model instances
model = AIFactory.create_language(
    "openai", 
    "gpt-3.5-turbo",
    config={"structured": {"type": "json"}}
)  # Language model
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")  # Embedding model
reranker = AIFactory.create_reranker("transformers", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # Universal reranker model
transcriber = AIFactory.create_speech_to_text("openai", "whisper-1")  # Speech-to-text model
speaker = AIFactory.create_text_to_speech("openai", "tts-1")  # Text-to-speech model

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
]
response = model.chat_complete(messages)

# Create an embedding instance
texts = ["Hello, world!", "Another text"]
# Synchronous usage
embeddings = embedder.embed(texts)
# Async usage
embeddings = await embedder.aembed(texts)
```

### Model Discovery 🔍

Esperanto provides a convenient way to discover available models from providers without creating instances:

```python
from esperanto.factory import AIFactory

# Discover available models from OpenAI
models = AIFactory.get_provider_models("openai", api_key="your-api-key")
for model in models:
    print(f"{model.id} - owned by {model.owned_by}")

# Filter by model type (for providers like OpenAI that support multiple types)
language_models = AIFactory.get_provider_models(
    "openai",
    api_key="your-api-key",
    model_type="language"  # Options: 'language', 'embedding', 'speech_to_text', 'text_to_speech'
)

# Some providers return hardcoded lists (e.g., Anthropic)
claude_models = AIFactory.get_provider_models("anthropic")
for model in claude_models:
    print(f"{model.id} - Context: {model.context_window} tokens")

# Example output:
# claude-3-5-sonnet-20241022 - Context: 200000 tokens
# claude-3-5-haiku-20241022 - Context: 200000 tokens
# claude-3-opus-20240229 - Context: 200000 tokens

# OpenAI-compatible endpoints (requires base_url)
local_models = AIFactory.get_provider_models(
    "openai-compatible",
    base_url="http://localhost:1234/v1"  # LM Studio, vLLM, etc.
)
for model in local_models:
    print(f"{model.id} - {model.owned_by}")
```

**Benefits of Static Discovery:**
- ✅ **No instance creation required** - Query models without setting up providers
- ✅ **Cached results** - Model lists are cached for 1 hour to reduce API calls
- ✅ **Flexible configuration** - Pass provider-specific config (API keys, base URLs, etc.)
- ✅ **Type filtering** - Filter models by type for multi-model providers

**Supported Providers:**
- **OpenAI** - Fetches models via API (supports type filtering)
- **OpenAI-Compatible** - Fetches models from any OpenAI-compatible endpoint (LM Studio, vLLM, etc.)
- **Anthropic** - Returns hardcoded list of Claude models
- **Google/Gemini** - Fetches models via API
- **Groq** - Fetches models via API
- **Mistral** - Fetches models via API
- **Ollama** - Fetches locally available models
- **Jina** - Returns hardcoded list of embedding/reranking models
- **Voyage** - Returns hardcoded list of embedding/reranking models
- **And more...**

> **Note**: This is the recommended way to discover models. The `.models` property on provider instances is deprecated and will be removed in version 3.0.

### Using Provider-Specific Classes

Here's a simple example to get you started:

```python
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.llm.anthropic import AnthropicLanguageModel

# Initialize a provider with structured output
model = OpenAILanguageModel(
    api_key="your-api-key",
    model_name="gpt-4",  # Optional, defaults to gpt-4
    structured={"type": "json"}  # Optional, for JSON output
)

# Simple chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "List three colors in JSON format"}
]

# Synchronous call
response = model.chat_complete(messages)
print(response.choices[0].message.content)  # Will be in JSON format

# Async call
async def get_response():
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)  # Will be in JSON format
```

## Standardized Responses

All providers in Esperanto return standardized response objects, making it easy to work with different models without changing your code.

### LLM Responses

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "openai", 
    "gpt-3.5-turbo",
    config={"structured": {"type": "json"}}
)
messages = [{"role": "user", "content": "Hello!"}]

# All LLM responses follow this structure
response = model.chat_complete(messages)
print(response.choices[0].message.content)  # The actual response text
print(response.choices[0].message.role)     # 'assistant'
print(response.model)                       # The model used
print(response.usage.total_tokens)          # Token usage information
print(response.content)          # Shortcut for response.choices[0].message.content

# For streaming responses
for chunk in model.chat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Embedding Responses

```python
from esperanto.factory import AIFactory

model = AIFactory.create_embedding("openai", "text-embedding-3-small")
texts = ["Hello, world!", "Another text"]

# All embedding responses follow this structure
response = model.embed(texts)
print(response.data[0].embedding)     # Vector for first text
print(response.data[0].index)         # Index of the text (0)
print(response.model)                 # The model used
print(response.usage.total_tokens)    # Token usage information
```

### Reranking Responses

```python
from esperanto.factory import AIFactory

reranker = AIFactory.create_reranker("transformers", "BAAI/bge-reranker-base")
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "The weather is nice today.",
    "Python is a programming language used in ML."
]

# All reranking responses follow this structure
response = reranker.rerank(query, documents, top_k=2)
print(response.results[0].document)          # Highest ranked document
print(response.results[0].relevance_score)   # Normalized 0-1 relevance score
print(response.results[0].index)             # Original document index
print(response.model)                        # The model used
```

### Task-Aware Embeddings 🎯

Esperanto supports advanced task-aware embeddings that optimize vector representations for specific use cases. This works across **all embedding providers** through a universal interface:

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Task-optimized embeddings work with ANY provider
model = AIFactory.create_embedding(
    provider="jina",  # Also works with: "openai", "google", "transformers", etc.
    model_name="jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,  # Optimize for search queries
        "late_chunking": True,                           # Better long-context handling
        "output_dimensions": 512                         # Control vector size
    }
)

# Generate optimized embeddings
query = "What is machine learning?"
embeddings = model.embed([query])
```

**Universal Task Types:**
- `RETRIEVAL_QUERY` - Optimize for search queries
- `RETRIEVAL_DOCUMENT` - Optimize for document storage  
- `SIMILARITY` - General text similarity
- `CLASSIFICATION` - Text classification tasks
- `CLUSTERING` - Document clustering
- `CODE_RETRIEVAL` - Code search optimization
- `QUESTION_ANSWERING` - Optimize for Q&A tasks
- `FACT_VERIFICATION` - Optimize for fact checking

**Provider Support:**
- **Jina**: Native API support for all features
- **Google**: Native task type translation to Gemini API
- **OpenAI**: Task optimization via intelligent text prefixes
- **Transformers**: Local emulation with task-specific processing
- **Others**: Graceful degradation with consistent interface

The standardized response objects ensure consistency across different providers, making it easy to:
- Switch between providers without changing your application code
- Handle responses in a uniform way
- Access common attributes like token usage and model information

## Provider Configuration 🔧

### OpenAI

```python
from esperanto.providers.llm.openai import OpenAILanguageModel

model = OpenAILanguageModel(
    api_key="your-api-key",  # Or set OPENAI_API_KEY env var
    model_name="gpt-4",      # Optional
    temperature=0.7,         # Optional
    max_tokens=850,         # Optional
    streaming=False,        # Optional
    top_p=0.9,             # Optional
    structured={"type": "json"},      # Optional, for JSON output
    base_url=None,         # Optional, for custom endpoint
    organization=None      # Optional, for org-specific API
)
```

### OpenAI-Compatible Endpoints

Use any OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, custom deployments) with the same interface:

```python
from esperanto.factory import AIFactory

# Using factory config
model = AIFactory.create_language(
    "openai-compatible",
    "your-model-name",  # Use any model name supported by your endpoint
    config={
        "base_url": "http://localhost:1234/v1",  # Your endpoint URL (required)
        "api_key": "your-api-key"                # Your API key (optional)
    }
)

# Or set environment variables
# Generic (works for all provider types):
# OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1
# OPENAI_COMPATIBLE_API_KEY=your-api-key  # Optional for endpoints that don't require auth

# Provider-specific (takes precedence over generic):
# OPENAI_COMPATIBLE_BASE_URL_LLM=http://localhost:1234/v1
# OPENAI_COMPATIBLE_API_KEY_LLM=your-api-key
model = AIFactory.create_language("openai-compatible", "your-model-name")

# Works with any OpenAI-compatible endpoint
messages = [{"role": "user", "content": "Hello!"}]
response = model.chat_complete(messages)
print(response.content)

# Streaming support
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Common Use Cases:**
- **LM Studio**: Local model serving with GUI
- **Ollama**: `ollama serve` with OpenAI compatibility
- **vLLM**: High-performance inference server
- **Custom Deployments**: Any server implementing OpenAI chat completions API

**Features:**
- ✅ **Streaming**: Real-time response streaming
- ✅ **Pass-through Model Names**: Use any model name your endpoint supports
- ✅ **Graceful Degradation**: Automatically handles varying feature support
- ✅ **Error Handling**: Clear error messages for troubleshooting
- ⚠️ **JSON Mode**: Depends on endpoint implementation

**Environment Variable Configuration:**

OpenAI-compatible providers support both generic and provider-specific environment variables:

- **Generic variables** (work for all provider types):
  - `OPENAI_COMPATIBLE_BASE_URL` - Base URL for the endpoint
  - `OPENAI_COMPATIBLE_API_KEY` - API key (if required)

- **Provider-specific variables** (take precedence over generic):
  - Language Models: `OPENAI_COMPATIBLE_BASE_URL_LLM`, `OPENAI_COMPATIBLE_API_KEY_LLM`
  - Embeddings: `OPENAI_COMPATIBLE_BASE_URL_EMBEDDING`, `OPENAI_COMPATIBLE_API_KEY_EMBEDDING`
  - Speech-to-Text: `OPENAI_COMPATIBLE_BASE_URL_STT`, `OPENAI_COMPATIBLE_API_KEY_STT`
  - Text-to-Speech: `OPENAI_COMPATIBLE_BASE_URL_TTS`, `OPENAI_COMPATIBLE_API_KEY_TTS`

**Configuration Precedence** (highest to lowest):
1. Direct parameters (`base_url=`, `api_key=`)
2. Config dictionary (`config={"base_url": ...}`)
3. Provider-specific environment variables
4. Generic environment variables
5. Default values

This allows you to use different OpenAI-compatible endpoints for different AI capabilities without code changes.

### Perplexity

Perplexity uses an OpenAI-compatible API but includes additional parameters for controlling search behavior.

```python
from esperanto.providers.llm.perplexity import PerplexityLanguageModel

model = PerplexityLanguageModel(
    api_key="your-api-key",  # Or set PERPLEXITY_API_KEY env var
    model_name="llama-3-sonar-large-32k-online", # Recommended default
    temperature=0.7,         # Optional
    max_tokens=850,         # Optional
    streaming=False,        # Optional
    top_p=0.9,             # Optional
    structured={"type": "json"}, # Optional, for JSON output

    # Perplexity-specific parameters
    search_domain_filter=["example.com", "-excluded.com"], # Optional, limit search domains
    return_images=False,             # Optional, include images in search results
    return_related_questions=True,  # Optional, return related questions
    search_recency_filter="week",    # Optional, filter search by time ('day', 'week', 'month', 'year')
    web_search_options={"search_context_size": "high"} # Optional, control search context ('low', 'medium', 'high')
)
```

## Timeout Configuration ⏱️

Esperanto provides flexible timeout configuration across all provider types with intelligent defaults and multiple configuration methods.

### Default Timeouts

Different provider types have optimized default timeouts based on typical operation duration:

- **LLM, Embedding, Reranking**: 60 seconds (text processing operations)
- **Speech-to-Text, Text-to-Speech**: 300 seconds (audio processing operations)

### Configuration Methods

Configure timeouts using three methods with clear priority hierarchy:

#### 1. Config Dictionary (Highest Priority)

```python
from esperanto.factory import AIFactory

# LLM with custom timeout
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"timeout": 120.0}  # 2 minutes
)

# Embedding with custom timeout
embedder = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-small",
    config={"timeout": 90.0}  # 1.5 minutes
)

# Speech-to-Text with longer timeout for large files
transcriber = AIFactory.create_speech_to_text(
    "openai",
    config={"timeout": 600.0}  # 10 minutes
)
```

#### 2. Direct Parameters (STT/TTS)

```python
# Text-to-Speech with direct timeout parameter
speaker = AIFactory.create_text_to_speech(
    "elevenlabs",
    timeout=180.0  # 3 minutes
)

# Speech-to-Text with direct timeout parameter
transcriber = AIFactory.create_speech_to_text(
    "openai",
    timeout=450.0  # 7.5 minutes
)
```

#### 3. Environment Variables

Set global defaults for all instances of a provider type:

```bash
# Set environment variables
export ESPERANTO_LLM_TIMEOUT=90          # 90 seconds for all LLM providers
export ESPERANTO_EMBEDDING_TIMEOUT=120   # 2 minutes for all embedding providers
export ESPERANTO_RERANKER_TIMEOUT=75     # 75 seconds for all reranker providers
export ESPERANTO_STT_TIMEOUT=600         # 10 minutes for all STT providers
export ESPERANTO_TTS_TIMEOUT=400         # 6.5 minutes for all TTS providers
```

```python
# These will use environment variable defaults
model = AIFactory.create_language("openai", "gpt-4")  # Uses ESPERANTO_LLM_TIMEOUT
embedder = AIFactory.create_embedding("voyage", "voyage-2")  # Uses ESPERANTO_EMBEDDING_TIMEOUT
```

### Priority Order

Configuration resolves in this priority order:

1. **Config parameter** (highest priority)
2. **Environment variable**
3. **Provider type default** (lowest priority)

```python
# Example: Final timeout will be 150 seconds (config overrides env var)
# Even if ESPERANTO_LLM_TIMEOUT=90 is set
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"timeout": 150.0}  # This takes precedence
)
```

### Validation

All timeout values are validated with clear error messages:

- **Type**: Must be a number (int or float)
- **Range**: Must be between 1 and 3600 seconds (1 hour maximum)

```python
# These will raise ValueError with descriptive messages
AIFactory.create_language("openai", "gpt-4", config={"timeout": "invalid"})  # TypeError
AIFactory.create_language("openai", "gpt-4", config={"timeout": -1})         # Out of range
AIFactory.create_language("openai", "gpt-4", config={"timeout": 4000})       # Too large
```

### Production Use Cases

**Batch Processing**
```python
# Long timeout for batch embedding operations
embedder = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-large",
    config={"timeout": 300.0}  # 5 minutes for large batches
)
```

**Real-time Applications**
```python
# Shorter timeout for real-time chat
model = AIFactory.create_language(
    "openai",
    "gpt-3.5-turbo",
    config={"timeout": 30.0}  # 30 seconds for quick responses
)
```

**Audio Processing**
```python
# Extended timeout for long audio files
transcriber = AIFactory.create_speech_to_text(
    "openai",
    config={"timeout": 900.0}  # 15 minutes for hour-long audio files
)
```

## Streaming Responses 🌊

Enable streaming to receive responses token by token:

```python
# Enable streaming
model = OpenAILanguageModel(api_key="your-api-key", streaming=True)

# Synchronous streaming
for chunk in model.chat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

## Structured Output 📊

Request JSON-formatted responses (supported by OpenAI and some OpenRouter models):

```python
model = OpenAILanguageModel(
    api_key="your-api-key", # or use ENV
    structured={"type": "json"}
)

messages = [
    {"role": "user", "content": "List three European capitals as JSON"}
]

response = model.chat_complete(messages)
# Response will be in JSON format
```

## LangChain Integration 🔗

Convert any provider to a LangChain chat model:

```python
model = OpenAILanguageModel(api_key="your-api-key")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Documentation 📚

Complete documentation is available in the [docs](https://github.com/lfnovo/esperanto/tree/main/docs) directory:

- **[Quick Start Guide](https://github.com/lfnovo/esperanto/blob/main/docs/quickstart.md)** - Get up and running in 5 minutes
- **[Documentation Index](https://github.com/lfnovo/esperanto/blob/main/docs/README.md)** - Navigation hub for all documentation
- **[Provider Comparison](https://github.com/lfnovo/esperanto/blob/main/docs/providers/README.md)** - Compare and choose providers
- **[Capability Guides](https://github.com/lfnovo/esperanto/tree/main/docs/capabilities)** - Learn about LLM, Embeddings, Reranking, STT, TTS
- **[Provider Setup Guides](https://github.com/lfnovo/esperanto/tree/main/docs/providers)** - Setup instructions for all 17 providers
- **[Advanced Topics](https://github.com/lfnovo/esperanto/tree/main/docs/advanced)** - Task-aware embeddings, LangChain, timeouts, and more

## Contributing 🤝

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/lfnovo/esperanto/blob/main/CONTRIBUTING.md) for details on how to get started.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](https://github.com/lfnovo/esperanto/blob/main/LICENSE) file for details.

## Development 🛠️

1. Clone the repository:
```bash
git clone https://github.com/lfnovo/esperanto.git
cd esperanto
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
pytest
