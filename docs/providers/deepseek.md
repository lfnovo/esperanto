# DeepSeek

## Overview

DeepSeek provides powerful language models with a focus on reasoning, coding, and general-purpose tasks. Their models offer competitive performance at attractive pricing, making them a strong choice for production applications.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | deepseek-chat, deepseek-reasoner |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://platform.deepseek.com/docs

## Prerequisites

### Account Requirements
- DeepSeek account (sign up at https://platform.deepseek.com)
- API key with credits or billing enabled

### Getting API Keys
1. Visit https://platform.deepseek.com/api_keys
2. Click "Create API Key"
3. Copy and store the key securely

## Environment Variables

```bash
# DeepSeek API key (required)
DEEPSEEK_API_KEY="sk-..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`DEEPSEEK_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Create DeepSeek model
model = AIFactory.create_language("deepseek", "deepseek-chat")

# Chat completion
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Direct Instantiation

```python
from esperanto.providers.llm.deepseek import DeepSeekLanguageModel

# Create model instance
model = DeepSeekLanguageModel(
    api_key="your-api-key",
    model_name="deepseek-chat"
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
| **deepseek-chat** | 64K tokens | General purpose, balanced performance |
| **deepseek-reasoner** | 64K tokens | Complex reasoning, step-by-step thinking |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "deepseek",
    "deepseek-chat",
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

# Create DeepSeek model
model = AIFactory.create_language("deepseek", "deepseek-chat")

# Simple chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

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
    "deepseek",
    "deepseek-chat",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three programming languages as JSON with name, year, and creator"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
# Response will be valid JSON
```

**Example - Reasoning Model:**

```python
# Use deepseek-reasoner for complex reasoning
reasoner = AIFactory.create_language("deepseek", "deepseek-reasoner")

messages = [{
    "role": "user",
    "content": "Solve this logic puzzle: If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
}]

response = reasoner.chat_complete(messages)
print(response.choices[0].message.content)
# Model will show step-by-step reasoning
```

**Example - Code Generation:**

```python
# DeepSeek excels at coding tasks
model = AIFactory.create_language("deepseek", "deepseek-chat")

messages = [{
    "role": "user",
    "content": "Write a Python function to implement binary search with error handling"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Async Chat:**

```python
async def chat_async():
    model = AIFactory.create_language("deepseek", "deepseek-chat")

    messages = [{"role": "user", "content": "Explain machine learning"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Run async
# await chat_async()
```

**Example - Multi-turn Conversation:**

```python
# Build conversation with context
messages = [
    {"role": "user", "content": "What is Rust?"},
    {"role": "assistant", "content": "Rust is a systems programming language..."},
    {"role": "user", "content": "What are its main advantages over C++?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Temperature Control:**

```python
# More creative (higher temperature)
creative_model = AIFactory.create_language(
    "deepseek",
    "deepseek-chat",
    config={"temperature": 1.5, "max_tokens": 1024}
)

# More focused (lower temperature)
focused_model = AIFactory.create_language(
    "deepseek",
    "deepseek-chat",
    config={"temperature": 0.3, "max_tokens": 1024}
)

messages = [{"role": "user", "content": "Write a creative story about AI."}]

creative_response = creative_model.chat_complete(messages)
focused_response = focused_model.chat_complete(messages)
```

**Example - Long Context:**

```python
# DeepSeek supports 64K token context
model = AIFactory.create_language("deepseek", "deepseek-chat")

long_document = """
[Your long document content here - up to 64K tokens]
"""

messages = [
    {"role": "user", "content": f"Analyze and summarize this document:\n\n{long_document}"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Advanced Features

### JSON Mode

DeepSeek supports structured JSON output:

```python
model = AIFactory.create_language(
    "deepseek",
    "deepseek-chat",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "Create a JSON object representing a book with title, author, year, and genres"
}]

response = model.chat_complete(messages)
# Response will be valid JSON
```

### Reasoning Model

Use deepseek-reasoner for complex logical tasks:

```python
reasoner = AIFactory.create_language("deepseek", "deepseek-reasoner")

# Complex reasoning task
messages = [{
    "role": "user",
    "content": """
    Three friends - Alice, Bob, and Carol - have different favorite colors: red, blue, and green.
    - Alice doesn't like red
    - Bob's favorite is not blue
    - Carol's favorite is green
    What is each person's favorite color?
    """
}]

response = reasoner.chat_complete(messages)
# Model provides step-by-step reasoning
```

### Timeout Configuration

Customize request timeouts:

```python
# Extended timeout for complex tasks
model = AIFactory.create_language(
    "deepseek",
    "deepseek-chat",
    config={
        "timeout": 120.0,    # 2 minutes
        "max_tokens": 4096
    }
)
```

### LangChain Integration

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("deepseek", "deepseek-chat")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Model Selection Guide

### DeepSeek Chat (Recommended for General Use)
**Best for:** Most use cases, balanced performance
- Excellent general-purpose capabilities
- Strong coding abilities
- Good reasoning skills
- Fast response times
- Cost-effective for production

```python
model = AIFactory.create_language("deepseek", "deepseek-chat")
```

### DeepSeek Reasoner (Best for Complex Reasoning)
**Best for:** Logic puzzles, complex analysis, step-by-step reasoning
- Shows reasoning process
- Excellent for logical tasks
- Good for mathematical problems
- Educational use cases
- Transparent thinking process

```python
model = AIFactory.create_language("deepseek", "deepseek-reasoner")
```

## Performance Characteristics

### Context Window
Both models support 64K token context:
- Approximately 48,000 words
- Long document processing
- Extensive conversation history

### Response Speed
- **Chat**: Fast (1-3 seconds typical)
- **Reasoner**: Moderate (2-5 seconds, includes reasoning steps)

### Strengths
- **Coding**: Excellent code generation and understanding
- **Reasoning**: Strong logical reasoning capabilities
- **Cost**: Competitive pricing
- **Context**: Good long-context handling (64K tokens)

## Use Cases

### When to Choose DeepSeek

**Perfect for:**
- Code generation and analysis
- Logical reasoning tasks
- Cost-sensitive production deployments
- General-purpose applications
- Educational tools (with reasoner model)
- Long-context tasks

**Consider alternatives if:**
- Need strongest reasoning (use Claude Opus or GPT-4)
- Need multimodal capabilities
- Require specialized domain knowledge
- Need embeddings or other modalities

### Common Applications

**1. Code Generation:**
```python
model = AIFactory.create_language("deepseek", "deepseek-chat")

messages = [{
    "role": "user",
    "content": "Create a Python class for a binary search tree with insert, search, and delete methods"
}]

response = model.chat_complete(messages)
```

**2. Logical Analysis:**
```python
reasoner = AIFactory.create_language("deepseek", "deepseek-reasoner")

messages = [{
    "role": "user",
    "content": "Analyze the logic of this argument: [complex argument]"
}]

response = reasoner.chat_complete(messages)
```

**3. Code Review:**
```python
model = AIFactory.create_language("deepseek", "deepseek-chat")

code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

messages = [{
    "role": "user",
    "content": f"Review this code and suggest improvements:\n\n{code}"
}]

response = model.chat_complete(messages)
```

**4. Educational Explanations:**
```python
reasoner = AIFactory.create_language("deepseek", "deepseek-reasoner")

messages = [{
    "role": "user",
    "content": "Explain how quicksort algorithm works with step-by-step reasoning"
}]

response = reasoner.chat_complete(messages)
```

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key at https://platform.deepseek.com/api_keys

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic with exponential backoff or upgrade plan

**Context Length Exceeded:**
```
Error: Prompt is too long
```
**Solution:**
- Reduce message history
- Summarize earlier messages
- Maximum is 64K tokens

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase timeout:
```python
config={"timeout": 120.0, "max_tokens": 1024}
```

**Invalid Model Name:**
```
Error: Model not found
```
**Solution:** Use exact model names:
- `deepseek-chat`
- `deepseek-reasoner`

### Best Practices

1. **Choose Right Model:**
   - Use `deepseek-chat` for most tasks
   - Use `deepseek-reasoner` when you need to see the thinking process

2. **Temperature Settings:**
   - 0.3-0.5 for factual/code tasks
   - 0.7-1.0 for creative tasks
   - Up to 2.0 for highly creative outputs

3. **System Messages:** Use clear system messages to set context and behavior

4. **Streaming:** Enable streaming for better UX with longer responses

5. **JSON Mode:** Use structured output when you need parseable results

6. **Context Management:** Take advantage of 64K context for long documents

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [OpenAI Provider](./openai.md)
- [Anthropic Provider](./anthropic.md)
- [Mistral Provider](./mistral.md)
- [Google Provider](./google.md)
