# Language Models

Language models (LLMs) are AI systems that can understand and generate human-like text. Esperanto provides a unified interface for working with various language model providers, allowing you to perform tasks like chat completion, text generation, and structured output generation across different AI models.

## Supported Providers

- **OpenAI** (GPT-4, GPT-3.5, o1)
- **OpenAI-Compatible** (LM Studio, Ollama, vLLM, custom endpoints)
- **Anthropic** (Claude 3)
- **Google** (Gemini 2.0 Flash, Gemini 1.5 Pro)
- **Groq** (Mixtral, Llama)
- **Ollama** (Local deployment)
- **OpenRouter** (Access to multiple models)
- **xAI** (Grok)
- **Perplexity** (Sonar models with web search)
- **Azure OpenAI** (Azure-hosted OpenAI models)
- **Mistral** (Mistral Large, Small, etc.)
- **DeepSeek** (deepseek-chat)

## Available Methods

All language model providers implement the following methods:

- **`chat_complete(messages, stream=None)`**: Generate a chat completion for the given messages
- **`achat_complete(messages, stream=None)`**: Async version of chat completion
- **`to_langchain()`**: Convert to a LangChain chat model for integration

### Parameters:
- `messages`: List of message dictionaries with 'role' and 'content' keys
- `stream`: Boolean to enable streaming responses (optional)

## Common Interface

All language models return standardized response objects:

### ChatCompletion Response
```python
response = model.chat_complete(messages)
# Access attributes:
response.choices[0].message.content  # The response text
response.choices[0].message.role     # 'assistant'
response.model                       # Model used
response.provider                    # Provider name
response.usage.total_tokens          # Token usage
```

### Streaming Response
```python
for chunk in model.chat_complete(messages, stream=True):
    chunk.choices[0].delta.content   # Partial response text
    chunk.model                      # Model used
    chunk.provider                   # Provider name
```

## Examples

### Basic Chat Completion
```python
from esperanto.factory import AIFactory

# Create a language model
model = AIFactory.create_language("openai", "gpt-4")

# Simple chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Async Chat Completion
```python
async def chat_async():
    model = AIFactory.create_language("anthropic", "claude-3-sonnet-20240229")
    
    messages = [{"role": "user", "content": "Explain quantum computing"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)
```

### Streaming Responses
```python
model = AIFactory.create_language("openai", "gpt-4")

messages = [{"role": "user", "content": "Write a short story"}]

# Sync streaming
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Structured Output (JSON)
```python
model = AIFactory.create_language(
    "openai", 
    "gpt-4",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user", 
    "content": "List three European capitals as JSON"
}]

response = model.chat_complete(messages)
# Response will be in JSON format
```

### LangChain Integration
```python
model = AIFactory.create_language("anthropic", "claude-3-sonnet-20240229")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Provider-Specific Information

### OpenAI-Compatible Endpoints

OpenAI-compatible endpoints allow you to use any server that implements the OpenAI chat completions API with Esperanto's unified interface.

**Supported Endpoints:**
- **LM Studio**: Local model serving with GUI
- **Ollama**: Local model deployment (`ollama serve`)
- **vLLM**: High-performance inference server
- **Custom Deployments**: Any server implementing OpenAI chat completions API

**Configuration Options:**

1. **Using Factory Config:**
```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "openai-compatible",
    "your-model-name",  # Use any model name supported by your endpoint
    config={
        "base_url": "http://localhost:1234/v1",  # Your endpoint URL (required)
        "api_key": "your-api-key",               # Your API key (optional)
        "temperature": 0.7,                      # Optional parameters
        "max_tokens": 1000,
        "streaming": True
    }
)
```

2. **Using Environment Variables:**
```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:1234/v1"
export OPENAI_COMPATIBLE_API_KEY="your-api-key"  # Optional for endpoints that don't require auth
```

```python
model = AIFactory.create_language("openai-compatible", "your-model-name")
```

**Example Usage:**
```python
from esperanto.factory import AIFactory

# Connect to LM Studio running locally
model = AIFactory.create_language(
    "openai-compatible",
    "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    config={
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio"
    }
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

# Regular completion
response = model.chat_complete(messages)
print(response.choices[0].message.content)

# Streaming completion
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Features:**
- ✅ **Streaming**: Real-time response streaming
- ✅ **Pass-through Model Names**: Use any model name your endpoint supports
- ✅ **Graceful Degradation**: Automatically handles varying feature support
- ✅ **Error Handling**: Clear error messages for troubleshooting
- ✅ **Model Discovery**: Automatic discovery of available models via `/models` endpoint
- ⚠️ **JSON Mode**: Support depends on endpoint implementation

**Common Endpoint URLs:**
- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`
- vLLM: `http://localhost:8000/v1`
- Custom: `https://your-endpoint.com/v1`

### Azure OpenAI

Azure OpenAI Service allows you to use OpenAI models hosted on Microsoft Azure infrastructure.

**Key Requirements:**
- **Deployment Name**: The `model_name` parameter corresponds to your Azure OpenAI deployment name
- **Environment Variables**: 
  - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
  - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI resource endpoint
  - `OPENAI_API_VERSION`: The API version (e.g., "2023-12-01-preview")

**Example:**
```python
from esperanto.factory import AIFactory

# Ensure environment variables are set
model = AIFactory.create_language(
    "azure",
    "your-deployment-name",  # Your Azure deployment name
    config={"temperature": 0.7, "structured": {"type": "json"}}  # Azure supports JSON mode
)

messages = [{"role": "user", "content": "Translate 'hello' to Spanish."}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Perplexity

Perplexity provides AI models with web search capabilities for up-to-date information.

**Special Parameters:**
- `search_domain_filter`: Limit search to specific domains
- `return_images`: Include images in search results
- `return_related_questions`: Return related questions
- `search_recency_filter`: Filter by time ("day", "week", "month", "year")
- `web_search_options`: Control search context size

**Example:**
```python
from esperanto.factory import AIFactory

# Set PERPLEXITY_API_KEY environment variable
model = AIFactory.create_language(
    provider="perplexity",
    model_name="llama-3-sonar-large-32k-online",
    search_domain_filter=["news.com", "-spam.com"],
    return_related_questions=True,
    search_recency_filter="week"
)

messages = [{"role": "user", "content": "What are the latest AI developments?"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```