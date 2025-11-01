# OpenRouter

## Overview

OpenRouter provides unified access to multiple AI models from different providers through a single API. It acts as a gateway to models from OpenAI, Anthropic, Google, Meta, Mistral, and many others, offering flexibility and easy model switching.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Access to 100+ models from multiple providers |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://openrouter.ai/docs

## Prerequisites

### Account Requirements
- OpenRouter account (sign up at https://openrouter.ai)
- API key with credits or payment method

### Getting API Keys
1. Visit https://openrouter.ai/keys
2. Click "Create Key"
3. Copy and store the key securely

## Environment Variables

```bash
# OpenRouter API key (required)
OPENROUTER_API_KEY="sk-or-v1-..."

# OpenRouter base URL (optional, defaults to https://openrouter.ai/api/v1)
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`, `base_url="..."`)
2. Environment variables (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`)
3. Default base URL (`https://openrouter.ai/api/v1`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Create OpenRouter model
# You can use any model available on OpenRouter
model = AIFactory.create_language("openrouter", "anthropic/claude-3.5-sonnet")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Direct Instantiation

```python
from esperanto.providers.llm.openrouter import OpenRouterLanguageModel

# Create model instance
model = OpenRouterLanguageModel(
    api_key="your-api-key",
    model_name="anthropic/claude-3.5-sonnet"
)

# Use the model
messages = [{"role": "user", "content": "Hello!"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Capabilities

### Language Models (LLM)

**Available Model Categories:**

OpenRouter provides access to 100+ models. Here are some popular choices:

**OpenAI Models:**
- `openai/gpt-4o` - Latest GPT-4 Optimized
- `openai/gpt-4-turbo` - Fast GPT-4
- `openai/gpt-3.5-turbo` - Cost-effective

**Anthropic Models:**
- `anthropic/claude-3.5-sonnet` - Latest Claude
- `anthropic/claude-3-opus` - Most capable Claude
- `anthropic/claude-3-haiku` - Fast Claude

**Google Models:**
- `google/gemini-2.0-flash-exp` - Latest Gemini
- `google/gemini-pro-1.5` - Balanced Gemini

**Meta Models:**
- `meta-llama/llama-3.1-405b-instruct` - Largest Llama
- `meta-llama/llama-3.1-70b-instruct` - Balanced Llama
- `meta-llama/llama-3.1-8b-instruct` - Fast Llama

**Mistral Models:**
- `mistralai/mistral-large` - Most capable Mistral
- `mistralai/mistral-small` - Fast Mistral
- `mistralai/codestral` - Code specialist

**Other Popular Models:**
- `perplexity/llama-3.1-sonar-large-128k-online` - With web search
- `deepseek/deepseek-chat` - Cost-effective
- `qwen/qwen-2.5-72b-instruct` - Multilingual

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "openrouter",
    "anthropic/claude-3.5-sonnet",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 2.0)
        "max_tokens": 1000,           # Maximum response length
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode (model-dependent)
        "timeout": 60.0               # Request timeout
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create OpenRouter model
model = AIFactory.create_language("openrouter", "anthropic/claude-3.5-sonnet")

# Simple chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Switch Models Easily:**

```python
# Try different models with same code
models_to_try = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-exp",
    "meta-llama/llama-3.1-70b-instruct"
]

messages = [{"role": "user", "content": "Explain machine learning in simple terms"}]

for model_name in models_to_try:
    model = AIFactory.create_language("openrouter", model_name)
    response = model.chat_complete(messages)
    print(f"\n{model_name}:")
    print(response.choices[0].message.content[:200] + "...")
```

**Example - Streaming:**

```python
model = AIFactory.create_language("openrouter", "anthropic/claude-3.5-sonnet")

messages = [{"role": "user", "content": "Write a short story about AI"}]

# Synchronous streaming
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Example - JSON Mode:**

```python
# Note: JSON mode support depends on the specific model
model = AIFactory.create_language(
    "openrouter",
    "openai/gpt-4o",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three programming languages as JSON"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Free Models:**

```python
# OpenRouter offers some free models
free_model = AIFactory.create_language("openrouter", "meta-llama/llama-3.1-8b-instruct:free")

messages = [{"role": "user", "content": "Hello!"}]
response = free_model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Code Generation:**

```python
# Use a code-specialized model
code_model = AIFactory.create_language("openrouter", "mistralai/codestral")

messages = [{
    "role": "user",
    "content": "Write a Python function to implement quicksort"
}]

response = code_model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Async Chat:**

```python
async def chat_async():
    model = AIFactory.create_language("openrouter", "anthropic/claude-3.5-sonnet")

    messages = [{"role": "user", "content": "Explain quantum computing"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Run async
# await chat_async()
```

**Example - Multi-turn Conversation:**

```python
# Build conversation with context
model = AIFactory.create_language("openrouter", "openai/gpt-4o")

messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."},
    {"role": "user", "content": "What are its main advantages?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Temperature Control:**

```python
# More creative (higher temperature)
creative_model = AIFactory.create_language(
    "openrouter",
    "anthropic/claude-3.5-sonnet",
    config={"temperature": 1.2, "max_tokens": 1024}
)

# More focused (lower temperature)
focused_model = AIFactory.create_language(
    "openrouter",
    "anthropic/claude-3.5-sonnet",
    config={"temperature": 0.3, "max_tokens": 1024}
)
```

## Advanced Features

### Model Discovery

Browse available models at https://openrouter.ai/models or use the API:

```python
import httpx

response = httpx.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {your_api_key}"}
)

models = response.json()
for model in models['data'][:10]:  # Show first 10
    print(f"{model['id']}: {model.get('name', 'N/A')}")
```

### Free Models

OpenRouter offers free access to some models:

```python
# Free models (append :free to model ID)
free_models = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free"
]

model = AIFactory.create_language("openrouter", free_models[0])
```

### Cost Optimization

Choose models based on your budget:

```python
# Expensive but highest quality
premium_model = AIFactory.create_language("openrouter", "anthropic/claude-3-opus")

# Balanced cost/performance
balanced_model = AIFactory.create_language("openrouter", "openai/gpt-4o-mini")

# Budget-friendly
budget_model = AIFactory.create_language("openrouter", "meta-llama/llama-3.1-8b-instruct")
```

### Timeout Configuration

Customize request timeouts:

```python
# Extended timeout for complex tasks
model = AIFactory.create_language(
    "openrouter",
    "anthropic/claude-3-opus",
    config={
        "timeout": 120.0,    # 2 minutes
        "max_tokens": 4096
    }
)
```

### LangChain Integration

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("openrouter", "anthropic/claude-3.5-sonnet")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Model Selection Guide

### For Quality
**Best:** Claude 3.5 Sonnet, GPT-4o, Claude 3 Opus
```python
model = AIFactory.create_language("openrouter", "anthropic/claude-3.5-sonnet")
```

### For Speed
**Best:** GPT-3.5 Turbo, Claude 3 Haiku, Gemini Flash
```python
model = AIFactory.create_language("openrouter", "google/gemini-2.0-flash-exp")
```

### For Coding
**Best:** Codestral, GPT-4o, Claude 3.5 Sonnet
```python
model = AIFactory.create_language("openrouter", "mistralai/codestral")
```

### For Cost
**Best:** Free models, Llama 3.1 8B, GPT-4o-mini
```python
model = AIFactory.create_language("openrouter", "meta-llama/llama-3.1-8b-instruct")
```

### For Long Context
**Best:** Claude 3 (200K), GPT-4 Turbo (128K), Gemini 1.5 Pro (2M)
```python
model = AIFactory.create_language("openrouter", "google/gemini-pro-1.5")
```

### For Multilingual
**Best:** Qwen 2.5, Mistral models, Gemini
```python
model = AIFactory.create_language("openrouter", "qwen/qwen-2.5-72b-instruct")
```

## Use Cases

### When to Choose OpenRouter

**Perfect for:**
- Model comparison and benchmarking
- Flexibility to switch providers easily
- Access to models not directly available
- Fallback strategies (try multiple models)
- Cost optimization across providers
- Single API for multiple providers
- Avoiding vendor lock-in

**Consider alternatives if:**
- Using only one provider consistently
- Need provider-specific features
- Want direct provider billing
- Require lowest possible latency

### Common Applications

**1. Model Comparison:**
```python
def compare_models(question, models):
    results = {}
    for model_name in models:
        model = AIFactory.create_language("openrouter", model_name)
        messages = [{"role": "user", "content": question}]
        response = model.chat_complete(messages)
        results[model_name] = response.choices[0].message.content
    return results

models = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-exp"
]

results = compare_models("Explain quantum computing", models)
```

**2. Fallback Strategy:**
```python
async def chat_with_fallback(messages):
    # Try models in order of preference
    models = [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "meta-llama/llama-3.1-70b-instruct"
    ]

    for model_name in models:
        try:
            model = AIFactory.create_language("openrouter", model_name)
            response = await model.achat_complete(messages)
            return response
        except Exception as e:
            print(f"Failed with {model_name}: {e}")
            continue

    raise Exception("All models failed")
```

**3. Cost-Optimized Pipeline:**
```python
# Use cheap model for simple tasks, premium for complex
def smart_completion(question, complexity="low"):
    if complexity == "low":
        model = AIFactory.create_language("openrouter", "meta-llama/llama-3.1-8b-instruct")
    elif complexity == "medium":
        model = AIFactory.create_language("openrouter", "openai/gpt-4o-mini")
    else:
        model = AIFactory.create_language("openrouter", "anthropic/claude-3-opus")

    messages = [{"role": "user", "content": question}]
    return model.chat_complete(messages)
```

**4. Specialized Tasks:**
```python
# Use best model for each task type
def get_specialized_model(task_type):
    models = {
        "code": "mistralai/codestral",
        "creative": "anthropic/claude-3.5-sonnet",
        "analysis": "openai/gpt-4o",
        "chat": "meta-llama/llama-3.1-70b-instruct"
    }
    return AIFactory.create_language("openrouter", models[task_type])

code_model = get_specialized_model("code")
creative_model = get_specialized_model("creative")
```

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key at https://openrouter.ai/keys

**Insufficient Credits:**
```
Error: Insufficient credits
```
**Solution:** Add credits at https://openrouter.ai/credits

**Model Not Available:**
```
Error: Model not found
```
**Solution:**
- Check model ID at https://openrouter.ai/models
- Ensure correct format: `provider/model-name`

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic or upgrade plan

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase timeout:
```python
config={"timeout": 120.0}
```

### Best Practices

1. **Use Full Model IDs:** Always include provider prefix (e.g., `anthropic/claude-3.5-sonnet`)

2. **Monitor Costs:** Different models have different pricing - check https://openrouter.ai/models

3. **Free Models:** Append `:free` for free tier (limited availability)

4. **Model Selection:** Choose based on your specific needs (quality, speed, cost)

5. **Fallback Strategy:** Implement fallbacks for production applications

6. **Check Capabilities:** Not all models support all features (JSON mode, function calling, etc.)

7. **Credits:** Keep credits topped up for uninterrupted service

## Performance Characteristics

### Response Times
Varies by model:
- **Fast models**: GPT-3.5, Claude Haiku (1-2 seconds)
- **Balanced**: GPT-4o, Gemini Flash (2-4 seconds)
- **Premium**: Claude Opus, GPT-4 (3-8 seconds)

### Context Windows
Varies by model:
- **Standard**: 4K-32K tokens (most models)
- **Extended**: 128K tokens (GPT-4 Turbo, Claude)
- **Long**: 200K tokens (Claude 3)
- **Very Long**: 1-2M tokens (Gemini 1.5)

### Pricing
Check current pricing at https://openrouter.ai/models
- Ranges from free to premium
- Pay only for what you use
- No subscription required

## See Also

- [Language Models Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/llm.md)
- [OpenAI Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/openai.md)
- [Anthropic Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/anthropic.md)
- [Google Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/google.md)
- [Mistral Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/mistral.md)
