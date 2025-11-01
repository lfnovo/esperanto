# Anthropic

## Overview

Anthropic provides access to the Claude family of large language models, known for their strong performance on reasoning, analysis, and longer-form tasks.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Claude 3.5, Claude 3 Opus/Sonnet/Haiku |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://docs.anthropic.com

## Prerequisites

### Account Requirements
- Anthropic account (sign up at https://console.anthropic.com)
- API key with credits or billing enabled

### Getting API Keys
1. Visit https://console.anthropic.com/settings/keys
2. Click "Create Key"
3. Copy and store the key securely

## Environment Variables

```bash
# Anthropic API key (required)
ANTHROPIC_API_KEY="sk-ant-..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`ANTHROPIC_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Create Claude model
model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Direct Instantiation

```python
from esperanto.providers.llm.anthropic import AnthropicLanguageModel

# Create model instance
model = AnthropicLanguageModel(
    api_key="your-api-key",
    model_name="claude-3-5-sonnet-20241022"
)

# Use the model
messages = [{"role": "user", "content": "Hello!"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Capabilities

### Language Models (LLM)

**Available Models:**

| Model | Context Window | Best For |
|-------|----------------|----------|
| **claude-3-5-sonnet-20241022** | 200K tokens | Latest, balanced performance and speed |
| **claude-3-5-haiku-20241022** | 200K tokens | Fast responses, cost-effective |
| **claude-3-opus-20240229** | 200K tokens | Complex tasks, highest capability |
| **claude-3-sonnet-20240229** | 200K tokens | Balanced performance |
| **claude-3-haiku-20240307** | 200K tokens | Speed and efficiency |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "anthropic",
    "claude-3-5-sonnet-20241022",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 1.0)
        "max_tokens": 1024,           # Maximum response length (required)
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode
        "timeout": 60.0               # Request timeout
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create Claude model
model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")

# Simple chat
messages = [
    {"role": "user", "content": "What's the capital of France?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - With System Message:**

```python
# Claude handles system messages naturally
messages = [
    {"role": "system", "content": "You are a helpful assistant specializing in geography."},
    {"role": "user", "content": "Tell me about the capital of Japan."}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Async Chat:**

```python
async def chat_async():
    model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")

    messages = [{"role": "user", "content": "Explain quantum computing"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Run async
# await chat_async()
```

**Example - Streaming:**

```python
# Synchronous streaming
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Example - JSON Mode:**

```python
# Enable JSON output
model = AIFactory.create_language(
    "anthropic",
    "claude-3-5-sonnet-20241022",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three programming languages with their typical use cases as JSON"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
# Response will be valid JSON
```

**Example - Long Context:**

```python
# Claude excels at long-context tasks with 200K token window
long_document = """
[Your long document content here - can be up to 200K tokens]
"""

messages = [
    {"role": "user", "content": f"Summarize this document:\n\n{long_document}"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Multi-turn Conversation:**

```python
# Build conversation history
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
    "anthropic",
    "claude-3-5-sonnet-20241022",
    config={"temperature": 1.0, "max_tokens": 1024}
)

# More focused (lower temperature)
focused_model = AIFactory.create_language(
    "anthropic",
    "claude-3-5-sonnet-20241022",
    config={"temperature": 0.2, "max_tokens": 1024}
)

messages = [{"role": "user", "content": "Write a creative story about AI."}]

creative_response = creative_model.chat_complete(messages)
focused_response = focused_model.chat_complete(messages)
```

## Advanced Features

### JSON Mode
Claude supports structured JSON output:

```python
model = AIFactory.create_language(
    "anthropic",
    "claude-3-5-sonnet-20241022",
    config={"structured": {"type": "json"}}
)

# Claude will return valid JSON
messages = [{
    "role": "user",
    "content": "Create a JSON object with user information"
}]

response = model.chat_complete(messages)
```

### Temperature and Top-P Priority
Claude prioritizes temperature over top_p when both are specified:

```python
# Temperature takes precedence
model = AIFactory.create_language(
    "anthropic",
    "claude-3-5-sonnet-20241022",
    config={
        "temperature": 0.7,  # This will be used
        "top_p": 0.9         # This will be ignored
    }
)
```

### Timeout Configuration
Customize request timeouts:

```python
# Extended timeout for complex tasks
model = AIFactory.create_language(
    "anthropic",
    "claude-3-5-sonnet-20241022",
    config={
        "timeout": 120.0,    # 2 minutes
        "max_tokens": 4096
    }
)
```

### LangChain Integration
Convert to LangChain models:

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Model Selection Guide

### Claude 3.5 Sonnet (Recommended)
**Best for:** Most use cases, balanced performance
- Excellent reasoning and analysis
- Fast response times
- Cost-effective for production
- 200K token context window

```python
model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
```

### Claude 3.5 Haiku
**Best for:** High-volume, fast responses
- Fastest Claude model
- Most cost-effective
- Good for simple tasks
- Still maintains strong capabilities

```python
model = AIFactory.create_language("anthropic", "claude-3-5-haiku-20241022")
```

### Claude 3 Opus
**Best for:** Complex reasoning, highest accuracy
- Most capable Claude model
- Best for complex analysis
- Highest cost
- Use when quality is paramount

```python
model = AIFactory.create_language("anthropic", "claude-3-opus-20240229")
```

## Performance Characteristics

### Context Window
All Claude 3 models support 200K token context:
- Approximately 150,000 words
- Entire codebases or long documents
- Extensive conversation history

### Response Quality
- **Opus**: Highest quality, best reasoning
- **Sonnet**: Excellent balance of quality and speed
- **Haiku**: Fast, still maintains good quality

### Speed
- **Haiku**: Fastest (sub-second for short responses)
- **Sonnet**: Fast (1-3 seconds typical)
- **Opus**: Slower but most thorough (3-10 seconds)

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key is correct and active in the Anthropic console.

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic with exponential backoff or contact Anthropic for higher limits.

**Context Length Exceeded:**
```
Error: Prompt is too long
```
**Solution:** Reduce the total tokens in your messages (Claude 3 supports up to 200K tokens).

**Missing max_tokens:**
```
Error: max_tokens is required
```
**Solution:** Always specify max_tokens in your configuration:
```python
config={"max_tokens": 1024}
```

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase the timeout configuration:
```python
config={"timeout": 120.0, "max_tokens": 1024}
```

### Best Practices

1. **Always Set max_tokens:** Unlike some providers, Anthropic requires max_tokens to be specified.

2. **Use Appropriate Model:** Choose the right model for your use case:
   - Haiku for speed and cost
   - Sonnet for balanced performance
   - Opus for complex reasoning

3. **System Messages:** Claude handles system messages naturally - use them to set context and behavior.

4. **Long Context:** Take advantage of the 200K context window for complex tasks.

5. **Temperature Settings:** Use lower temperatures (0.2-0.5) for factual tasks, higher (0.7-1.0) for creative tasks.

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [OpenAI Provider](./openai.md)
- [Google Provider](./google.md)
- [Groq Provider](./groq.md)
