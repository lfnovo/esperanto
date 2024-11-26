# Esperanto 🌐

Esperanto is a powerful Python library that provides a unified interface for interacting with various Large Language Model (LLM) providers. It simplifies the process of working with different LLM APIs by offering a consistent interface while maintaining provider-specific optimizations.

## Features ✨

- **Unified Interface**: Work with multiple LLM providers using a consistent API
- **Embedding Support**: Multiple embedding providers for vector representations
- **Async Support**: Both synchronous and asynchronous API calls
- **Streaming**: Support for streaming responses
- **Structured Output**: JSON output formatting (where supported)
- **LangChain Integration**: Easy conversion to LangChain chat models

For detailed information about our providers, check out:
- [LLM Providers Documentation](docs/llm.md)
- [Embedding Providers Documentation](docs/embedding.md)

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

# Google support
poetry add esperanto[google]
# or pip install esperanto[google]

# Vertex AI support
poetry add esperanto[vertex]
# or pip install esperanto[vertex]

# Ollama support
poetry add esperanto[ollama]
# or pip install esperanto[ollama]

# Install multiple providers
poetry add "esperanto[openai,anthropic,google]"
# or pip install "esperanto[openai,anthropic,google]"

# Install all providers
poetry add "esperanto[all]"
# or pip install "esperanto[all]"
```

## Provider Support Matrix

| Provider  | LLM Support | Embedding Support |
|-----------|-------------|------------------|
| OpenAI    | ✅          | ✅               |
| Anthropic | ✅          | ❌               |
| Google    | ✅          | ✅               |
| Vertex AI | ✅          | ✅               |
| Ollama    | ✅          | ✅               |


## Quick Start 🏃‍♂️

You can use Esperanto in two ways: directly with provider-specific classes or through the AI Factory.

### Using AI Factory

The AI Factory provides a convenient way to create LLM and embedding instances:

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
# Synchronous usage
embeddings = model.embed(texts)
# Async usage
embeddings = await model.aembed(texts)
```


### Using Provider-Specific Classes

Here's a simple example to get you started:

```python
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.llm.anthropic import AnthropicLanguageModel

# Initialize a provider
model = OpenAILanguageModel(
    api_key="your-api-key",
    model_name="gpt-4"  # Optional, defaults to gpt-4
)

# Simple chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

# Synchronous call
response = model.chat_complete(messages)
print(response.choices[0].message.content)

# Async call
async def get_response():
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)
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
    structured="json",      # Optional, for JSON output
    base_url=None,         # Optional, for custom endpoint
    organization=None      # Optional, for org-specific API
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
    structured="json"
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

## Contributing 🤝

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Development 🛠️

1. Clone the repository:
```bash
git clone https://github.com/lfnovo/esperanto.git
cd esperanto
```

2. Install dependencies:
```bash
poetry install
```

3. Run tests:
```bash
poetry run pytest
```
