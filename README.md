# Esperanto 🌐

[![PyPI version](https://badge.fury.io/py/esperanto.svg)](https://badge.fury.io/py/esperanto)
[![PyPI Downloads](https://img.shields.io/pypi/dm/esperanto)](https://pypi.org/project/esperanto/)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](https://github.com/lfnovo/esperanto)
[![Python Versions](https://img.shields.io/pypi/pyversions/esperanto)](https://pypi.org/project/esperanto/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Esperanto is a powerful Python library that provides a unified interface for interacting with various Large Language Model (LLM) providers. It simplifies the process of working with different AI models (LLMs, Embedders, Transcribers, and TTS) APIs by offering a consistent interface while maintaining provider-specific optimizations.

## Features ✨


- **Unified Interface**: Work with multiple LLM providers using a consistent API
- **Provider Support**:
  - OpenAI (GPT-4, GPT-3.5, o1, Whisper, TTS)
  - Anthropic (Claude 3)
  - OpenRouter (Access to multiple models)
  - xAI (Grok)
  - Groq (Mixtral, Llama, Whisper)
  - Google GenAI (Gemini LLM, Text To Speech, Embedding)
  - Vertex AI (Google Cloud)
  - Ollama (Local deployment)
  - ElevenLabs (Text-to-Speech)
- **Embedding Support**: Multiple embedding providers for vector representations
- **Speech-to-Text Support**: Transcribe audio using multiple providers
- **Text-to-Speech Support**: Generate speech using multiple providers
- **Async Support**: Both synchronous and asynchronous API calls
- **Streaming**: Support for streaming responses
- **Structured Output**: JSON output formatting (where supported)
- **LangChain Integration**: Easy conversion to LangChain chat models


For detailed information about our providers, check out:
- [LLM Providers Documentation](docs/llm.md)
- [Embedding Providers Documentation](docs/embedding.md)
- [Speech-to-Text Providers Documentation](docs/speech_to_text.md)
- [Text-to-Speech Providers Documentation](docs/text_to_speech.md)

## Installation 🚀

Install Esperanto using pip:

```bash
pip install esperanto
```

For specific providers, install with their extras:

```bash
# For OpenAI support
pip install "esperanto[openai]"

# For Anthropic support
pip install "esperanto[anthropic]"

# For Google (GenAI) support
pip install "esperanto[google]"

# For Vertex AI support
pip install "esperanto[vertex]"

# For Groq support
pip install "esperanto[groq]"

# For Ollama support
pip install "esperanto[ollama]"

# For ElevenLabs support
pip install "esperanto[elevenlabs]"

# For Google TTS support
pip install "esperanto[googletts]"

# For all providers
pip install "esperanto[all]"
```

## Provider Support Matrix

| Provider    | LLM Support | Embedding Support | Speech-to-Text | Text-to-Speech | JSON Mode |
|------------|-------------|------------------|----------------|----------------|-----------|
| OpenAI     | ✅          | ✅               | ✅             | ✅             | ✅        |
| Anthropic  | ✅          | ❌               | ❌             | ❌             | ✅        |
| Groq       | ✅          | ❌               | ✅             | ❌             | ✅        |
| Google (GenAI)     | ✅          | ✅               | ❌             | ✅             | ✅        |
| Vertex AI  | ✅          | ✅               | ❌             | ❌             | ❌        |
| Ollama     | ✅          | ✅               | ❌             | ❌             | ❌        |
| ElevenLabs | ❌          | ❌               | ❌             | ✅             | ❌        |

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
#     'language': ['openai', 'anthropic', 'google', 'groq', 'ollama', 'openrouter', 'xai'],
#     'embedding': ['openai', 'google', 'ollama', 'vertex'],
#     'speech_to_text': ['openai', 'groq'],
#     'text_to_speech': ['openai', 'elevenlabs', 'google']
# }

# Create model instances
model = AIFactory.create_language(
    "openai", 
    "gpt-3.5-turbo",
    structured={"type": "json"}
)  # Language model
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")  # Embedding model
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
    structured={"type": "json"}
)
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
    structured={"type": "json"},      # Optional, for JSON output
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
pip install -r requirements.txt
```

3. Run tests:
```bash
pytest
```
