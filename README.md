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
  - Azure OpenAI (Chat, Embedding)
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

For detailed information about our providers, check out:
- [LLM Providers Documentation](https://github.com/lfnovo/esperanto/blob/main/docs/llm.md)
- [Embedding Providers Documentation](https://github.com/lfnovo/esperanto/blob/main/docs/embedding.md)
- [Reranking Providers Documentation](https://github.com/lfnovo/esperanto/blob/main/docs/rerank.md)
- [Speech-to-Text Providers Documentation](https://github.com/lfnovo/esperanto/blob/main/docs/speech_to_text.md)
- [Text-to-Speech Providers Documentation](https://github.com/lfnovo/esperanto/blob/main/docs/text_to_speech.md)

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
| OpenAI-Compatible | ✅          | ❌               | ❌                | ❌             | ❌             | ⚠️*       |
| Anthropic    | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |
| Groq         | ✅          | ❌               | ❌                | ✅             | ❌             | ✅        |
| Google (GenAI) | ✅          | ✅               | ❌                | ❌             | ✅             | ✅        |
| Vertex AI    | ✅          | ✅               | ❌                | ❌             | ✅             | ❌        |
| Ollama       | ✅          | ✅               | ❌                | ❌             | ❌             | ❌        |
| Perplexity   | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |
| Transformers | ❌          | ✅               | ✅                | ❌             | ❌             | ❌        |
| ElevenLabs   | ❌          | ❌               | ❌                | ✅             | ✅             | ❌        |
| Azure OpenAI | ✅          | ✅               | ❌                | ❌             | ❌             | ✅        |
| Mistral      | ✅          | ✅               | ❌                | ❌             | ❌             | ✅        |
| DeepSeek     | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |
| Voyage       | ❌          | ✅               | ✅                | ❌             | ❌             | ❌        |
| Jina         | ❌          | ✅               | ✅                | ❌             | ❌             | ❌        |
| xAI          | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |
| OpenRouter   | ✅          | ❌               | ❌                | ❌             | ❌             | ✅        |

*⚠️ OpenAI-Compatible: JSON mode support depends on the specific endpoint implementation

## Quick Start 🏃‍♂️

You can use Esperanto in two ways: directly with provider-specific classes or through the AI Factory.

## AIFactory - Smart Model Management 🏭

The `AIFactory` is Esperanto's intelligent model management system that provides significant performance benefits through its **singleton cache architecture**.

### 🚀 **Singleton Cache Benefits**

AIFactory automatically caches model instances based on their configuration. This means:
- **No duplicate model creation** - same provider + model + config = same instance returned
- **Faster subsequent calls** - cached instances are returned immediately
- **Memory efficient** - prevents memory bloat from multiple identical models
- **Connection reuse** - HTTP clients and configurations are preserved

### 💡 **How It Works**

```python
from esperanto.factory import AIFactory

# First call - creates new model instance
model1 = AIFactory.create_language("openai", "gpt-4", temperature=0.7)

# Second call with same config - returns cached instance (instant!)
model2 = AIFactory.create_language("openai", "gpt-4", temperature=0.7)

# They're the exact same object
assert model1 is model2  # True!

# Different config - creates new instance
model3 = AIFactory.create_language("openai", "gpt-4", temperature=0.9)
assert model1 is not model3  # True - different config
```

### 🎯 **Perfect for Production**

This caching is especially powerful in production scenarios:

```python
# In a web application
def handle_chat_request(messages):
    # This model is cached - no recreation overhead!
    model = AIFactory.create_language("anthropic", "claude-3-sonnet-20240229")
    return model.chat_complete(messages)

def handle_embedding_request(texts):
    # This embedding model is also cached
    embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")
    return embedder.embed(texts)

# Multiple calls to these functions reuse the same model instances
# = Better performance + Lower memory usage
```

### 🔍 **Cache Key Strategy**

The cache key includes:
- **Provider name** (e.g., "openai", "anthropic")
- **Model name** (e.g., "gpt-4", "claude-3-sonnet")
- **All configuration parameters** (temperature, max_tokens, etc.)

Only models with **identical configurations** share the same cache entry.

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
#     'embedding': ['openai', 'google', 'ollama', 'vertex', 'transformers', 'voyage', 'mistral', 'azure', 'jina'],
#     'reranker': ['jina', 'voyage', 'transformers'],
#     'speech_to_text': ['openai', 'groq', 'elevenlabs'],
#     'text_to_speech': ['openai', 'elevenlabs', 'google', 'vertex']
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
# OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1
# OPENAI_COMPATIBLE_API_KEY=your-api-key  # Optional for endpoints that don't require auth
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

You can find the documentation for Esperanto in the [docs](https://github.com/lfnovo/esperanto/tree/main/docs) directory.

There is also a cool beginner's tutorial in the [tutorial](https://github.com/lfnovo/esperanto/blob/main/docs/tutorial/index.md) directory.

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
