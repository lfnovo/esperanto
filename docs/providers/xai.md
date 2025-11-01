# xAI (Grok)

## Overview

xAI provides access to Grok, an AI assistant with real-time knowledge and a unique personality. Grok models are designed to be helpful, truthful, and engaging, with access to current information.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Grok models |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://docs.x.ai

## Prerequisites

### Account Requirements
- xAI account (sign up at https://x.ai)
- API key with access enabled

### Getting API Keys
1. Visit https://console.x.ai
2. Navigate to API Keys section
3. Create a new API key
4. Copy and store the key securely

## Environment Variables

```bash
# xAI API key (required)
XAI_API_KEY="xai-..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`XAI_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Create Grok model
model = AIFactory.create_language("xai", "grok-beta")

# Chat completion
messages = [{"role": "user", "content": "What's happening in AI today?"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Direct Instantiation

```python
from esperanto.providers.llm.xai import XAILanguageModel

# Create model instance
model = XAILanguageModel(
    api_key="your-api-key",
    model_name="grok-beta"
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
| **grok-beta** | 128K tokens | Latest Grok with current knowledge |
| **grok-2-latest** | 128K tokens | Stable production model |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "xai",
    "grok-beta",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 2.0)
        "max_tokens": 1000,           # Maximum response length
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

# Create Grok model
model = AIFactory.create_language("xai", "grok-beta")

# Simple chat
messages = [
    {"role": "system", "content": "You are Grok, a helpful AI assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Current Events:**

```python
# Grok has access to real-time information
model = AIFactory.create_language("xai", "grok-beta")

messages = [{
    "role": "user",
    "content": "What are the latest developments in AI this week?"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
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
    "xai",
    "grok-beta",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three tech companies as JSON with name, CEO, and founding year"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
# Response will be valid JSON
```

**Example - Async Chat:**

```python
async def chat_async():
    model = AIFactory.create_language("xai", "grok-beta")

    messages = [{"role": "user", "content": "Explain quantum computing"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Run async
# await chat_async()
```

**Example - Multi-turn Conversation:**

```python
# Build conversation with Grok's personality
messages = [
    {"role": "user", "content": "What do you think about AI safety?"},
    {"role": "assistant", "content": "AI safety is crucial..."},
    {"role": "user", "content": "What are the main challenges?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Temperature Control:**

```python
# More creative (higher temperature)
creative_model = AIFactory.create_language(
    "xai",
    "grok-beta",
    config={"temperature": 1.2, "max_tokens": 1024}
)

# More focused (lower temperature)
focused_model = AIFactory.create_language(
    "xai",
    "grok-beta",
    config={"temperature": 0.3, "max_tokens": 1024}
)

messages = [{"role": "user", "content": "Write a creative story about Mars colonization."}]

creative_response = creative_model.chat_complete(messages)
focused_response = focused_model.chat_complete(messages)
```

**Example - Long Context:**

```python
# Grok supports 128K token context
model = AIFactory.create_language("xai", "grok-beta")

long_document = """
[Your long document content here - up to 128K tokens]
"""

messages = [
    {"role": "user", "content": f"Analyze this document and provide key insights:\n\n{long_document}"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Code Generation:**

```python
model = AIFactory.create_language("xai", "grok-beta")

messages = [{
    "role": "user",
    "content": "Write a Python function to implement a simple neural network from scratch"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Advanced Features

### JSON Mode

Grok supports structured JSON output:

```python
model = AIFactory.create_language(
    "xai",
    "grok-beta",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "Create a JSON representing a product catalog with 3 items"
}]

response = model.chat_complete(messages)
# Response will be valid JSON
```

### Real-Time Knowledge

Grok has access to current information:

```python
model = AIFactory.create_language("xai", "grok-beta")

# Ask about recent events
messages = [{
    "role": "user",
    "content": "What happened in tech news today?"
}]

response = model.chat_complete(messages)
# Grok provides up-to-date information
```

### Timeout Configuration

Customize request timeouts:

```python
# Extended timeout for complex tasks
model = AIFactory.create_language(
    "xai",
    "grok-beta",
    config={
        "timeout": 120.0,    # 2 minutes
        "max_tokens": 4096
    }
)
```

### LangChain Integration

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("xai", "grok-beta")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Model Selection Guide

### Grok Beta (Latest)
**Best for:** Latest features, cutting-edge capabilities
- Most recent Grok version
- Latest improvements
- May have new features
- Good for experimentation

```python
model = AIFactory.create_language("xai", "grok-beta")
```

### Grok 2 Latest (Stable)
**Best for:** Production deployments, stability
- Stable production model
- Reliable performance
- Good for production use
- Consistent behavior

```python
model = AIFactory.create_language("xai", "grok-2-latest")
```

## Performance Characteristics

### Context Window
All Grok models support 128K token context:
- Approximately 96,000 words
- Long document processing
- Extensive conversation history

### Response Speed
- Typical: 2-4 seconds
- Streaming: Starts in < 1 second

### Unique Features
- **Real-time Knowledge**: Access to current information
- **Personality**: Engaging, conversational style
- **Truthfulness**: Focus on accurate, honest responses

## Use Cases

### When to Choose xAI/Grok

**Perfect for:**
- Applications needing current information
- Conversational AI with personality
- General-purpose chat applications
- Research and analysis with up-to-date data
- Creative content generation

**Consider alternatives if:**
- Need strongest reasoning (use Claude Opus)
- Need multimodal capabilities (use GPT-4 Vision, Gemini)
- Need specialized capabilities (embeddings, STT, TTS)
- Need guaranteed enterprise SLA

### Common Applications

**1. Current Events Analysis:**
```python
model = AIFactory.create_language("xai", "grok-beta")

messages = [{
    "role": "user",
    "content": "Summarize today's major tech announcements"
}]

response = model.chat_complete(messages)
```

**2. Research Assistance:**
```python
model = AIFactory.create_language("xai", "grok-beta")

messages = [{
    "role": "user",
    "content": "What are the latest breakthroughs in quantum computing?"
}]

response = model.chat_complete(messages)
```

**3. Conversational AI:**
```python
model = AIFactory.create_language("xai", "grok-beta")

messages = [
    {"role": "user", "content": "Tell me an interesting fact about space"},
    {"role": "assistant", "content": "Did you know that..."},
    {"role": "user", "content": "That's fascinating! Tell me more"}
]

response = model.chat_complete(messages)
```

**4. Code Explanation:**
```python
model = AIFactory.create_language("xai", "grok-beta")

code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

messages = [{
    "role": "user",
    "content": f"Explain this code and suggest optimizations:\n\n{code}"
}]

response = model.chat_complete(messages)
```

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key at https://console.x.ai

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic with exponential backoff or contact xAI for higher limits

**Context Length Exceeded:**
```
Error: Prompt is too long
```
**Solution:**
- Reduce message history
- Summarize earlier messages
- Maximum is 128K tokens

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase timeout:
```python
config={"timeout": 120.0, "max_tokens": 1024}
```

**Model Not Available:**
```
Error: Model not found
```
**Solution:** Use exact model names:
- `grok-beta`
- `grok-2-latest`

### Best Practices

1. **Leverage Real-Time Knowledge:** Take advantage of Grok's access to current information

2. **Use Appropriate Model:**
   - Beta for latest features
   - Grok-2-latest for production stability

3. **Temperature Settings:**
   - 0.3-0.5 for factual tasks
   - 0.7-0.9 for conversational tasks
   - 1.0-1.5 for creative tasks

4. **System Messages:** Set clear context and behavior expectations

5. **Streaming:** Enable streaming for better UX with longer responses

6. **Context Management:** Take advantage of 128K context for long documents

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [OpenAI Provider](./openai.md)
- [Anthropic Provider](./anthropic.md)
- [Google Provider](./google.md)
- [Perplexity Provider](./perplexity.md)
