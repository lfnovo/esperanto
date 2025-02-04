# Language Models

Esperanto provides a unified interface for working with various language model providers. This document outlines how to use the language model functionality.

## Supported Providers

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Groq
- Ollama
- OpenRouter
- XAI

## Basic Usage

```python
from esperanto import LanguageModel

# Initialize a model
model = LanguageModel.create("openai", api_key="your-api-key")

# Simple completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
response = model.chat_complete(messages)
print(response.content)

# Async completion
response = await model.achat_complete(messages)
print(response.content)

# Streaming completion
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

## Configuration

Each provider can be configured with various parameters:

```python
model = LanguageModel.create(
    "openai",
    api_key="your-api-key",
    model_name="gpt-4",  # Specific model to use
    max_tokens=1000,     # Maximum tokens in completion
    temperature=0.7,     # Randomness of output
    streaming=True,      # Enable streaming responses
    top_p=0.9,          # Nucleus sampling parameter
    base_url="custom-endpoint",  # Custom API endpoint
    organization="org-id"        # Organization ID
)
```

## Accessing Provider Clients

Each language model instance provides access to the underlying provider client through two properties:

- `client`: The synchronous client instance
- `async_client`: The asynchronous client instance

This allows you to access provider-specific functionality when needed:

```python
# Access the OpenAI client directly
openai_model = LanguageModel.create("openai", api_key="your-api-key")
raw_client = openai_model.client  # Get the OpenAI client instance
async_client = openai_model.async_client  # Get the async OpenAI client instance

# Use the raw client for provider-specific operations
models = raw_client.models.list()
```

## LangChain Integration

All models can be converted to LangChain chat models:

```python
langchain_model = model.to_langchain()
```

## Structured Output

Models can be configured to return structured output:

```python
model = LanguageModel.create(
    "openai",
    api_key="your-api-key",
    structured={"type": "json"}  # Request JSON output
)
```
