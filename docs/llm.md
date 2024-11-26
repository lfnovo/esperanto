# LLM Providers

Esperanto supports various Large Language Model (LLM) providers, offering a unified interface while maintaining provider-specific optimizations.

## Supported Providers

### OpenAI
- Models: GPT-4, GPT-3.5
- Features: 
  - Streaming responses
  - JSON structured output
  - Custom API endpoints
  - Organization-specific API support

### Anthropic
- Models: Claude 3 family
- Features:
  - Streaming responses
  - High-quality reasoning and analysis

### OpenRouter
- Access to multiple models from different providers
- Unified billing and API access
- Features:
  - JSON structured output (model-dependent)
  - Streaming responses

### xAI
- Models: Grok
- Features:
  - Real-time knowledge integration
  - Streaming responses

### Ollama
- Local model deployment and inference
- Support for various open-source models
- Features:
  - No API key required
  - Custom model configuration
  - Low latency for local deployments

## Usage Examples

### Basic Usage
```python
from esperanto.providers.llm.openai import OpenAILanguageModel

model = OpenAILanguageModel(
    api_key="your-api-key",
    model_name="gpt-4"
)

response = model.chat_complete([
    {"role": "user", "content": "Hello!"}
])
```

### Streaming Example
```python
model = OpenAILanguageModel(api_key="your-api-key", streaming=True)

for chunk in model.chat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

## Provider-Specific Configuration

Each provider may have specific configuration options. Here are some examples:

### OpenAI
```python
model = OpenAILanguageModel(
    api_key="your-api-key",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=850,
    streaming=False,
    structured="json"
)
```

### Ollama
```python
from esperanto.providers.llm.ollama import OllamaLanguageModel

model = OllamaLanguageModel(
    model_name="llama2",  # or any other supported model
    base_url="http://localhost:11434"  # default Ollama server
)
```

### Anthropic
```python
from esperanto.providers.llm.anthropic import AnthropicLanguageModel

model = AnthropicLanguageModel(
    api_key="your-api-key",
    model_name="claude-3-opus-20240229"
)
```
