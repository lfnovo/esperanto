# Esperanto 🌐

Esperanto is a powerful Python library that provides a unified interface for interacting with various Large Language Model (LLM) providers. It simplifies the process of working with different LLM APIs by offering a consistent interface while maintaining provider-specific optimizations.

## Features ✨

- **Unified Interface**: Work with multiple LLM providers using a consistent API
- **Provider Support**:
  - [OpenAI](https://openai.com) (GPT-4, GPT-3.5)
  - [Anthropic](https://anthropic.com) (Claude 3)
  - [OpenRouter](https://openrouter.ai) (Access to multiple models)
  - [xAI](https://x.ai) (Grok)
  - [Ollama](https://ollama.ai) (Local deployment)
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

## Quick Start 🏃‍♂️

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
