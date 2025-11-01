# Mistral

## Overview

Mistral AI provides high-performance language models and embeddings with a focus on efficiency, multilingual capabilities, and European data residency. Their models are known for excellent performance-to-cost ratios.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Mistral Large, Small, Codestral, etc. |
| Embeddings | ✅ | mistral-embed |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://docs.mistral.ai

## Prerequisites

### Account Requirements
- Mistral AI account (sign up at https://console.mistral.ai)
- API key with credits or billing enabled

### Getting API Keys
1. Visit https://console.mistral.ai/api-keys
2. Click "Create new key"
3. Copy and store the key securely

## Environment Variables

```bash
# Mistral API key (required)
MISTRAL_API_KEY="..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`MISTRAL_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Language model
model = AIFactory.create_language("mistral", "mistral-large-latest")

# Embedding model
embedder = AIFactory.create_embedding("mistral", "mistral-embed")
```

### Direct Instantiation

```python
from esperanto.providers.llm.mistral import MistralLanguageModel
from esperanto.providers.embedding.mistral import MistralEmbeddingModel

# Language model
llm = MistralLanguageModel(
    api_key="your-api-key",
    model_name="mistral-large-latest"
)

# Embedding model
embedder = MistralEmbeddingModel(
    api_key="your-api-key",
    model_name="mistral-embed"
)
```

## Capabilities

### Language Models (LLM)

**Available Models:**

| Model | Context Window | Best For |
|-------|----------------|----------|
| **mistral-large-latest** | 128K tokens | Most capable, complex reasoning |
| **mistral-small-latest** | 32K tokens | Fast, cost-effective |
| **codestral-latest** | 32K tokens | Code generation and understanding |
| **mistral-nemo** | 128K tokens | Balanced performance |
| **pixtral-12b-latest** | 128K tokens | Multimodal (text + images) |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "mistral",
    "mistral-large-latest",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 1.0)
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

# Create Mistral model
model = AIFactory.create_language("mistral", "mistral-large-latest")

# Chat completion
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
    "mistral",
    "mistral-large-latest",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three European capitals as JSON with country and capital"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
# Response will be valid JSON
```

**Example - Code Generation:**

```python
# Use Codestral for code tasks
code_model = AIFactory.create_language("mistral", "codestral-latest")

messages = [{
    "role": "user",
    "content": "Write a Python function to calculate Fibonacci numbers"
}]

response = code_model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Multilingual:**

```python
# Mistral models excel at multilingual tasks
model = AIFactory.create_language("mistral", "mistral-large-latest")

messages = [
    {"role": "user", "content": "Explain quantum computing in French"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Async Chat:**

```python
async def chat_async():
    model = AIFactory.create_language("mistral", "mistral-large-latest")

    messages = [{"role": "user", "content": "Explain machine learning"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Run async
# await chat_async()
```

**Example - Temperature Control:**

```python
# More creative (higher temperature)
creative_model = AIFactory.create_language(
    "mistral",
    "mistral-large-latest",
    config={"temperature": 1.0, "max_tokens": 1024}
)

# More focused (lower temperature)
focused_model = AIFactory.create_language(
    "mistral",
    "mistral-large-latest",
    config={"temperature": 0.2, "max_tokens": 1024}
)
```

**Example - Multi-turn Conversation:**

```python
# Build conversation history
messages = [
    {"role": "user", "content": "What is Rust programming language?"},
    {"role": "assistant", "content": "Rust is a systems programming language..."},
    {"role": "user", "content": "How does it compare to C++?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Embeddings

**Available Models:**
- **mistral-embed** - High-quality embeddings (1024 dimensions)

**Configuration:**

```python
from esperanto.factory import AIFactory

embedder = AIFactory.create_embedding(
    "mistral",
    "mistral-embed",
    config={
        "timeout": 60.0  # Request timeout in seconds
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
embedder = AIFactory.create_embedding("mistral", "mistral-embed")

# Generate embeddings
texts = ["Hello, world!", "Mistral embeddings are great"]
response = embedder.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
    print(f"First 5 values: {embedding_obj.embedding[:5]}")
```

**Example - Batch Processing:**

```python
# Process many documents
documents = [f"Document {i} content here" for i in range(100)]

response = embedder.embed(documents)
embeddings = [item.embedding for item in response.data]

print(f"Generated {len(embeddings)} embeddings")
```

**Example - Async Embeddings:**

```python
async def embed_documents():
    embedder = AIFactory.create_embedding("mistral", "mistral-embed")

    texts = ["Document 1", "Document 2", "Document 3"]
    response = await embedder.aembed(texts)

    return [item.embedding for item in response.data]

# Run async
# embeddings = await embed_documents()
```

**Example - Semantic Search:**

```python
# Create embeddings for search
embedder = AIFactory.create_embedding("mistral", "mistral-embed")

# Embed documents
documents = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "Rome is the capital of Italy"
]
doc_embeddings = embedder.embed(documents)

# Embed query
query = "What is the capital of France?"
query_embedding = embedder.embed([query])

# Use embeddings for similarity search (with your vector DB)
```

## Advanced Features

### JSON Mode

Mistral supports structured JSON output:

```python
model = AIFactory.create_language(
    "mistral",
    "mistral-large-latest",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "Create a JSON object with user information"
}]

response = model.chat_complete(messages)
# Response will be valid JSON
```

### Timeout Configuration

Customize request timeouts:

```python
# Extended timeout for complex tasks
model = AIFactory.create_language(
    "mistral",
    "mistral-large-latest",
    config={"timeout": 120.0, "max_tokens": 4096}
)

# Quick timeout for simple queries
model = AIFactory.create_language(
    "mistral",
    "mistral-small-latest",
    config={"timeout": 30.0}
)
```

### LangChain Integration

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("mistral", "mistral-large-latest")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Model Selection Guide

### Mistral Large (Recommended for Quality)
**Best for:** Complex reasoning, analysis, high-quality outputs
- Most capable Mistral model
- Excellent reasoning and analysis
- Strong multilingual performance
- 128K token context window
- Best for production use cases

```python
model = AIFactory.create_language("mistral", "mistral-large-latest")
```

### Mistral Small (Recommended for Speed)
**Best for:** Fast responses, cost optimization
- Fast inference
- Cost-effective
- Good for simple tasks
- 32K token context window
- Ideal for high-volume applications

```python
model = AIFactory.create_language("mistral", "mistral-small-latest")
```

### Codestral (Best for Code)
**Best for:** Code generation, code understanding
- Optimized for code tasks
- Supports multiple programming languages
- Fast code generation
- 32K token context window
- Best for developer tools

```python
model = AIFactory.create_language("mistral", "codestral-latest")
```

### Mistral Nemo
**Best for:** Balanced performance
- Good balance of speed and quality
- 128K token context window
- Multilingual capabilities
- Moderate cost

```python
model = AIFactory.create_language("mistral", "mistral-nemo")
```

## Performance Characteristics

### Context Windows
- **Large**: 128K tokens (~96,000 words)
- **Small**: 32K tokens (~24,000 words)
- **Codestral**: 32K tokens
- **Nemo**: 128K tokens

### Multilingual Support
Mistral models excel at:
- French (native language)
- English
- German
- Spanish
- Italian
- Portuguese
- And many more European languages

### Response Speed
- **Small**: Fastest (1-2 seconds)
- **Nemo**: Fast (2-3 seconds)
- **Large**: Moderate (2-4 seconds)
- **Codestral**: Fast for code (1-3 seconds)

## Use Cases

### When to Choose Mistral

**Perfect for:**
- European data residency requirements
- Multilingual applications (especially European languages)
- French language content
- Cost-effective production deployments
- Code generation and analysis
- Balanced performance and cost

**Consider alternatives if:**
- Need strongest possible reasoning (use Claude or GPT-4)
- Primary language is not European
- Need embeddings with task optimization (use Jina or OpenAI)

### Common Applications

**1. Multilingual Customer Support:**
```python
model = AIFactory.create_language("mistral", "mistral-large-latest")

# Handles French, German, Spanish, etc.
messages = [{"role": "user", "content": "Comment puis-je vous aider?"}]
```

**2. Code Generation:**
```python
code_model = AIFactory.create_language("mistral", "codestral-latest")

# Generate code in multiple languages
messages = [{"role": "user", "content": "Write a REST API in Python"}]
```

**3. Content Analysis:**
```python
model = AIFactory.create_language("mistral", "mistral-large-latest")

# Analyze multilingual content
messages = [{"role": "user", "content": "Analyze this European market report..."}]
```

**4. Document Embeddings:**
```python
embedder = AIFactory.create_embedding("mistral", "mistral-embed")

# Create semantic search for multilingual docs
docs = ["English doc", "Document français", "Documento español"]
embeddings = embedder.embed(docs)
```

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key at https://console.mistral.ai/api-keys

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic with exponential backoff or upgrade plan

**Context Length Exceeded:**
```
Error: Prompt too long
```
**Solution:**
- Use Mistral Large or Nemo for longer contexts (128K)
- Reduce message history
- Summarize earlier messages

**Model Not Available:**
```
Error: Model not found
```
**Solution:** Check model name - use "mistral-large-latest" not "mistral-large"

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase timeout:
```python
config={"timeout": 120.0, "max_tokens": 1024}
```

### Best Practices

1. **Use Latest Versions:** Always use "-latest" suffix for newest models

2. **Choose Appropriate Model:**
   - Large for quality
   - Small for speed
   - Codestral for code
   - Nemo for balance

3. **Leverage Multilingual:** Mistral excels at European languages

4. **Temperature Settings:**
   - 0.2-0.5 for factual tasks
   - 0.7-0.9 for creative tasks

5. **Streaming:** Use streaming for better UX with longer responses

## See Also

- [Language Models Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/llm.md)
- [Embeddings Guide](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/capabilities/embedding.md)
- [OpenAI Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/openai.md)
- [Anthropic Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/anthropic.md)
- [Google Provider](/Users/luisnovo/dev/projetos/esperanto/esperanto/docs/providers/google.md)
