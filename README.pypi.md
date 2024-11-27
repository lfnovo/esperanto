# Esperanto 🌐

[![PyPI version](https://badge.fury.io/py/esperanto.svg)](https://badge.fury.io/py/esperanto)
[![PyPI Downloads](https://img.shields.io/pypi/dm/esperanto)](https://pypi.org/project/esperanto/)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](https://github.com/lfnovo/esperanto)
[![Python Versions](https://img.shields.io/pypi/pyversions/esperanto)](https://pypi.org/project/esperanto/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Esperanto is a powerful Python library that provides a unified interface for interacting with various Large Language Model (LLM) providers. It simplifies the process of working with different LLM APIs by offering a consistent interface while maintaining provider-specific optimizations.

## Features ✨

- **Unified Interface**: Work with multiple LLM providers using a consistent API
- **Provider Support**:
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude 3)
  - OpenRouter (Access to multiple models)
  - xAI (Grok)
  - Groq (Mixtral, Llama)
  - Gemini
  - Vertex AI (Google Cloud)
  - Ollama (Local deployment)
- **Embedding Support**: Multiple embedding providers for vector representations
- **Async Support**: Both synchronous and asynchronous API calls
- **Streaming**: Support for streaming responses
- **Structured Output**: JSON output formatting (where supported)
- **LangChain Integration**: Easy conversion to LangChain chat models

For detailed information about our providers, check out:
- [LLM Providers Documentation](https://github.com/lfnovo/esperanto/blob/main/docs/llm.md)
- [Embedding Providers Documentation](https://github.com/lfnovo/esperanto/blob/main/docs/embedding.md)

## Installation 🚀

Install Esperanto using Poetry:

```bash
poetry add esperanto
```

Or with pip:

```bash
pip install esperanto
```

### Optional Dependencies

Esperanto supports multiple providers through optional dependencies. Install only what you need:

```bash
# OpenAI support
poetry add esperanto[openai]
# or pip install esperanto[openai]

# Anthropic support
poetry add esperanto[anthropic]
# or pip install esperanto[anthropic]

# Gemini support
poetry add esperanto[gemini]
# or pip install esperanto[gemini]

# Vertex AI support
poetry add esperanto[vertex]
# or pip install esperanto[vertex]

# Ollama support
poetry add esperanto[ollama]
# or pip install esperanto[ollama]

# Groq support
poetry add esperanto[groq]
# or pip install esperanto[groq]

# Install multiple providers
poetry add "esperanto[openai,anthropic,gemini,groq]"
# or pip install "esperanto[openai,anthropic,gemini,groq]"

# Install all providers
poetry add "esperanto[all]"
# or pip install "esperanto[all]"
```

## Provider Support Matrix

| Provider  | LLM Support | Embedding Support |
|-----------|-------------|------------------|
| OpenAI    | ✅          | ✅               |
| Anthropic | ✅          | ❌               |
| Groq      | ✅          | ❌               |
| Gemini    | ✅          | ✅               |
| Vertex AI | ✅          | ✅               |
| Ollama    | ✅          | ✅               |

## Quick Start 🏃‍♂️

You can use Esperanto in two ways: directly with provider-specific classes or through the AI Factory.

### Using AI Factory

```python
from esperanto.factory import AIFactory

# Create an LLM instance
model = AIFactory.create_llm("openai", "gpt-3.5-turbo")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
]
response = model.chat_complete(messages)

# Create an embedding instance
model = AIFactory.create_embedding("openai", "text-embedding-3-small")
texts = ["Hello, world!", "Another text"]
embeddings = model.embed(texts)
```

## Standardized Responses

All providers in Esperanto return standardized response objects, making it easy to work with different models without changing your code.

### LLM Responses

```python
from esperanto.factory import AIFactory

model = AIFactory.create_llm("openai", "gpt-3.5-turbo")
messages = [{"role": "user", "content": "Hello!"}]

# All LLM responses follow this structure
response = model.chat_complete(messages)
print(response.choices[0].message.content)  # The actual response text
print(response.choices[0].message.role)     # 'assistant'
print(response.model)                       # The model used
print(response.usage.total_tokens)          # Token usage information

# For streaming responses
for chunk in model.chat_complete(messages):
    print(chunk.choices[0].delta.content)   # Partial response text
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

The standardized response objects ensure consistency across different providers, making it easy to:
- Switch between providers without changing your application code
- Handle responses in a uniform way
- Access common attributes like token usage and model information

## Links 🔗

- **Documentation**: [GitHub Documentation](https://github.com/lfnovo/esperanto#readme)
- **Source Code**: [GitHub Repository](https://github.com/lfnovo/esperanto)
- **Issue Tracker**: [GitHub Issues](https://github.com/lfnovo/esperanto/issues)

## License 📄

This project is licensed under the MIT License - see the [LICENSE](https://github.com/lfnovo/esperanto/blob/main/LICENSE) file for details.
