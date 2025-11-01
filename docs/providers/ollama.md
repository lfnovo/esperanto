# Ollama

## Overview

Ollama enables local deployment of various open-source language models and embeddings with a simple API interface. It provides privacy-focused, self-hosted AI capabilities without cloud dependencies.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Llama, Mistral, Qwen, Phi, and many more |
| Embeddings | ✅ | nomic-embed-text, mxbai-embed-large, etc. |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://ollama.ai

## Prerequisites

### Installation Requirements
1. Download and install Ollama from https://ollama.ai
2. Start the Ollama service (runs on `http://localhost:11434` by default)

### Getting Started with Ollama
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a language model
ollama pull llama3.1

# Pull an embedding model
ollama pull nomic-embed-text

# List installed models
ollama list
```

## Environment Variables

```bash
# Ollama base URL (optional, defaults to http://localhost:11434)
OLLAMA_BASE_URL="http://localhost:11434"
```

**Variable Priority:**
1. Direct parameter in code (`base_url="..."`)
2. Environment variable (`OLLAMA_BASE_URL`)
3. Default value (`http://localhost:11434`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Language model
model = AIFactory.create_language("ollama", "llama3.1")

# Embedding model
embedder = AIFactory.create_embedding("ollama", "nomic-embed-text")
```

### Direct Instantiation

```python
from esperanto.providers.llm.ollama import OllamaLanguageModel
from esperanto.providers.embedding.ollama import OllamaEmbeddingModel

# Language model
llm = OllamaLanguageModel(
    base_url="http://localhost:11434",
    model_name="llama3.1"
)

# Embedding model
embedder = OllamaEmbeddingModel(
    base_url="http://localhost:11434",
    model_name="nomic-embed-text"
)
```

## Capabilities

### Language Models (LLM)

**Popular Models:**

| Model Family | Example Models | Size Range | Best For |
|--------------|---------------|------------|----------|
| **Llama** | llama3.1, llama3.2, llama2 | 1B - 405B | General purpose, instruction following |
| **Mistral** | mistral, mistral-nemo, mixtral | 7B - 8x7B | Fast, efficient reasoning |
| **Qwen** | qwen2.5, qwen2.5-coder | 0.5B - 72B | Coding, multilingual tasks |
| **Phi** | phi3, phi3.5 | 3.8B - 14B | Efficient, mobile deployment |
| **Gemma** | gemma2, gemma | 2B - 27B | Google's open models |
| **Codellama** | codellama | 7B - 70B | Code generation |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "ollama",
    "llama3.1",
    config={
        "base_url": "http://localhost:11434",  # Custom endpoint
        "temperature": 0.7,                     # Randomness (0.0 - 1.0)
        "num_predict": 1000,                    # Max tokens to generate
        "top_p": 0.9,                           # Nucleus sampling
        "streaming": True,                      # Enable streaming
        "keep_alive": "5m"                      # Keep model loaded
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create Ollama model
model = AIFactory.create_language("ollama", "llama3.1")

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

**Example - Code Generation:**

```python
# Use code-specialized model
code_model = AIFactory.create_language("ollama", "qwen2.5-coder")

messages = [{
    "role": "user",
    "content": "Write a Python function to calculate factorial"
}]

response = code_model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Keep Alive Control:**

```python
# Keep model loaded for 10 minutes (reduces startup latency)
model = AIFactory.create_language(
    "ollama",
    "llama3.1",
    config={"keep_alive": "10m"}
)

# Unload immediately after use (free memory)
model = AIFactory.create_language(
    "ollama",
    "llama3.1",
    config={"keep_alive": "0"}
)
```

**Example - Remote Ollama Server:**

```python
# Connect to Ollama running on another machine
model = AIFactory.create_language(
    "ollama",
    "llama3.1",
    config={"base_url": "http://192.168.1.100:11434"}
)

response = model.chat_complete(messages)
```

### Embeddings

**Available Models:**

```bash
# Popular embedding models
ollama pull nomic-embed-text     # 274MB, good quality
ollama pull mxbai-embed-large    # 669MB, high quality
ollama pull snowflake-arctic-embed  # 669MB, enterprise-grade
ollama pull all-minilm           # Small, fast
```

| Model | Size | Dimensions | Best For |
|-------|------|------------|----------|
| **nomic-embed-text** | 274MB | 768 | General purpose, efficient |
| **mxbai-embed-large** | 669MB | 1024 | High quality embeddings |
| **snowflake-arctic-embed** | 669MB | 1024 | Enterprise applications |
| **all-minilm** | 133MB | 384 | Fast, lightweight |

**Configuration:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

embedder = AIFactory.create_embedding(
    "ollama",
    "nomic-embed-text",
    config={
        "base_url": "http://localhost:11434",  # Custom endpoint
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,  # Task optimization via prefixes
        "truncate": True,                       # Truncate long texts
        "keep_alive": "5m"                      # Keep model loaded
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
embedder = AIFactory.create_embedding("ollama", "nomic-embed-text")

# Generate embeddings
texts = ["Hello, world!", "Another document", "More text"]
response = embedder.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
```

**Example - Task-Optimized Embeddings:**

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Optimize for queries
query_model = AIFactory.create_embedding(
    "ollama",
    "nomic-embed-text",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

# Optimize for documents
doc_model = AIFactory.create_embedding(
    "ollama",
    "nomic-embed-text",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

# Generate embeddings
query = query_model.embed(["search query"])
docs = doc_model.embed(["doc 1", "doc 2"])
```

**Example - Async Embeddings:**

```python
async def embed_documents():
    embedder = AIFactory.create_embedding("ollama", "nomic-embed-text")

    texts = ["Document 1", "Document 2", "Document 3"]
    response = await embedder.aembed(texts)

    return [item.embedding for item in response.data]

# Run async
# embeddings = await embed_documents()
```

## Advanced Features

### Model Management

**List Available Models:**
```bash
# Via Ollama CLI
ollama list

# Via Ollama API
curl http://localhost:11434/api/tags
```

**Pull New Models:**
```bash
# Download a specific model
ollama pull llama3.1

# Download specific version
ollama pull llama3.1:13b

# Download specific quantization
ollama pull llama3.1:70b-q4_K_M
```

**Remove Models:**
```bash
ollama rm llama3.1
```

### Model Variants and Quantization

Ollama supports different quantization levels (Q4, Q5, Q6, Q8) that trade off model size vs quality:

```bash
# Smallest, fastest (lower quality)
ollama pull llama3.1:q4_K_M

# Balanced
ollama pull llama3.1:q5_K_M

# Higher quality (larger size)
ollama pull llama3.1:q8_0
```

### Custom Base URL

Connect to Ollama running on a different machine or port:

```python
# Remote server
model = AIFactory.create_language(
    "ollama",
    "llama3.1",
    config={"base_url": "http://192.168.1.100:11434"}
)

# Custom port
model = AIFactory.create_language(
    "ollama",
    "llama3.1",
    config={"base_url": "http://localhost:8080"}
)
```

### LangChain Integration

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("ollama", "llama3.1")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

### Performance Optimization

**Keep Model Loaded:**
```python
# Keep model in memory for faster subsequent calls
model = AIFactory.create_language(
    "ollama",
    "llama3.1",
    config={"keep_alive": "30m"}  # Keep for 30 minutes
)
```

**Unload Model:**
```python
# Free memory immediately after use
model = AIFactory.create_language(
    "ollama",
    "llama3.1",
    config={"keep_alive": "0"}
)
```

## Troubleshooting

### Common Errors

**Connection Error:**
```
Error: Failed to connect to Ollama
```
**Solution:**
1. Ensure Ollama is running: `ollama serve`
2. Check the base URL is correct
3. Verify the port is accessible

**Model Not Found:**
```
Error: Model 'llama3.1' not found
```
**Solution:** Pull the model first:
```bash
ollama pull llama3.1
```

**Out of Memory:**
```
Error: Out of memory
```
**Solution:**
1. Use a smaller model variant (e.g., `llama3.1:7b` instead of `llama3.1:70b`)
2. Use quantized models (q4_K_M is most memory-efficient)
3. Close other applications to free RAM
4. Unload models after use with `keep_alive: "0"`

**Slow Performance:**
**Solution:**
1. Use quantized models (Q4, Q5) for faster inference
2. Use smaller model variants
3. Enable GPU acceleration if available
4. Keep models loaded with `keep_alive` for repeated calls

### System Requirements

**Minimum Requirements:**
- 8GB RAM for 7B models
- 16GB RAM for 13B models
- 32GB+ RAM for 70B+ models

**GPU Acceleration:**
- NVIDIA GPU with CUDA support (recommended)
- Apple Silicon (M1/M2/M3) - automatic Metal acceleration
- AMD ROCm support (Linux)

### Checking Ollama Status

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# View logs
ollama logs

# Check GPU usage
nvidia-smi  # For NVIDIA GPUs
```

## Use Cases

### When to Choose Ollama

**Perfect for:**
- Privacy-sensitive applications (100% local)
- Offline/air-gapped environments
- Development and testing
- Learning and experimentation
- Cost optimization (no API fees)
- Full control over model versions
- Custom model fine-tuning

**Consider alternatives if:**
- Need guaranteed enterprise SLA
- Want latest cutting-edge models immediately
- Limited local compute resources
- Require 24/7 availability without managing infrastructure

### Model Selection Guide

**For General Chat:**
- llama3.1 (7B or 13B) - Balanced performance
- mistral (7B) - Fast, efficient
- phi3.5 - Mobile/edge deployment

**For Coding:**
- qwen2.5-coder - Best code generation
- codellama - Strong code understanding
- deepseek-coder - Competitive alternative

**For Embeddings:**
- nomic-embed-text - General purpose (recommended)
- mxbai-embed-large - Higher quality
- all-minilm - Fast, lightweight

**For Multilingual:**
- qwen2.5 - Excellent multilingual support
- llama3.1 - Good multilingual capabilities

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [Embeddings Guide](../capabilities/embedding.md)
- [OpenAI-Compatible Provider](./openai-compatible.md)
- [Transformers Provider](./transformers.md)
