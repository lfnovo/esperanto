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
- Perplexity
- Azure OpenAI

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

### Azure OpenAI

Esperanto supports Azure OpenAI Service via the official `openai` Python SDK. This allows you to use your Azure-hosted OpenAI models with the familiar Esperanto interface.

**Key Considerations:**

-   **Deployment Name**: When using Azure OpenAI with Esperanto, the `model_name` parameter you provide corresponds to your **Azure OpenAI deployment name**, not the underlying model ID (e.g., "gpt-35-turbo").
-   **Environment Variables**: The Azure provider requires the following environment variables to be set:
    -   `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key.
    -   `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI resource endpoint (e.g., `https://your-resource-name.openai.azure.com/`).
    -   `OPENAI_API_VERSION`: The API version to use (e.g., `2023-12-01-preview`).

**Initialization:**

*Using AI Factory:*

```python
from esperanto.factory import AIFactory

# Ensure environment variables are set
azure_model = AIFactory.create_language(
    provider="azure",
    model_name="your-azure-deployment-name", # This is your deployment name
    # Optional parameters
    temperature=0.7,
    max_tokens=1000,
    structured={"type": "json"} # Azure OpenAI supports JSON mode
)

messages = [{"role": "user", "content": "Translate 'hello' to Esperanto."}]
response = azure_model.chat_complete(messages)
print(response.choices[0].message.content)
```

*Direct Initialization:*

```python
from esperanto.providers.llm.azure import AzureLanguageModel

# Ensure environment variables are set or pass parameters directly
azure_model = AzureLanguageModel(
    model_name="your-azure-deployment-name", # This is your deployment name
    # api_key="your_azure_key", # Can be set via env or directly
    # config={
    #     "azure_endpoint": "https://your-resource.openai.azure.com/",
    #     "api_version": "2023-12-01-preview"
    # }, 
    temperature=0.7,
    structured={"type": "json"}
)

messages = [{"role": "user", "content": "What is the capital of Portugal?"}]
response = azure_model.chat_complete(messages)
print(response.choices[0].message.content)
```

Azure OpenAI also supports streaming and asynchronous operations just like other LLM providers in Esperanto.

### Perplexity

Perplexity uses an OpenAI-compatible API but includes additional parameters for controlling search behavior. You can pass these parameters via the `config` dictionary when using the `AIFactory`.

```python
from esperanto.factory import AIFactory

# Ensure PERPLEXITY_API_KEY environment variable is set

model = AIFactory.create_language(
    provider="perplexity",
    model_name="llama-3-sonar-large-32k-online", # Recommended default
    config={
        "temperature": 0.7,         # Optional
        "max_tokens": 850,         # Optional
        "streaming": False,        # Optional
        "top_p": 0.9,             # Optional
        "structured": {"type": "json"}, # Optional, for JSON output

        # Perplexity-specific parameters
        "search_domain_filter": ["example.com", "-excluded.com"], # Optional, limit search domains
        "return_images": False,             # Optional, include images in search results
        "return_related_questions": True,  # Optional, return related questions
        "search_recency_filter": "week",    # Optional, filter search by time ('day', 'week', 'month', 'year')
        "web_search_options": {"search_context_size": "high"} # Optional, control search context ('low', 'medium', 'high')
    }
)

# Now you can use the model instance
messages = [{"role": "user", "content": "What are the latest AI news?"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
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
