# Quick Start Guide

Get started with Esperanto in 5 minutes! This guide walks you through installation, setup, and your first AI interactions.

## Installation

Install Esperanto via pip:

```bash
pip install esperanto
```

### Optional Dependencies

**For local Transformers models:**
```bash
pip install "esperanto[transformers]"
```

**For LangChain integration:**
```bash
pip install "langchain>=0.3.8" "langchain-core>=0.3.29"
# Plus provider-specific packages as needed
```

## Your First LLM Call

### 1. Get an API Key

For this quickstart, we'll use OpenAI. Get your API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

Other providers work similarly - see [Provider Comparison](./providers/README.md) to choose.

### 2. Set Environment Variable

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```bash
# .env
OPENAI_API_KEY=your-api-key-here
```

### 3. Generate Text

```python
from esperanto.factory import AIFactory

# Create a language model
model = AIFactory.create_language("openai", "gpt-4")

# Have a conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Esperanto?"}
]

response = model.chat_complete(messages)
print(response.content)
```

**Output:**
```
Esperanto is an international auxiliary language created in the late 19th century by L. L. Zamenhof...
```

🎉 **Congratulations!** You just made your first AI call with Esperanto.

## More Examples

### Text Embeddings

Convert text to vectors for semantic search:

```python
from esperanto.factory import AIFactory

# Create an embedding model
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Generate embeddings
texts = [
    "Esperanto is a universal AI interface",
    "Python is a programming language"
]

response = embedder.embed(texts)
vectors = [item.embedding for item in response.data]

print(f"Generated {len(vectors)} vectors")
print(f"Vector dimension: {len(vectors[0])}")
```

### Speech-to-Text

Transcribe audio files:

```python
from esperanto.factory import AIFactory

# Create a transcriber
transcriber = AIFactory.create_speech_to_text("openai", "whisper-1")

# Transcribe audio
transcript = transcriber.transcribe("meeting_recording.mp3")
print(transcript)
```

### Text-to-Speech

Generate natural-sounding audio:

```python
from esperanto.factory import AIFactory

# Create a TTS model
speaker = AIFactory.create_text_to_speech("openai", "tts-1")

# Generate speech
audio_bytes = speaker.generate_speech(
    text="Hello! This is Esperanto text to speech.",
    voice="nova"
)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

### Reranking

Improve search relevance:

```python
from esperanto.factory import AIFactory

# Create a reranker
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")

# Rerank documents
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence",
    "The weather is nice today",
    "Python is used in ML development"
]

response = reranker.rerank(query, documents, top_k=2)

for result in response.results:
    print(f"Score: {result.relevance_score:.4f} - {result.document}")
```

## Switching Providers

The beauty of Esperanto is that switching providers is as simple as changing two parameters:

```python
# OpenAI
model = AIFactory.create_language("openai", "gpt-4")

# Switch to Anthropic
model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")

# Switch to Google
model = AIFactory.create_language("google", "gemini-pro")

# Switch to local Ollama
model = AIFactory.create_language("ollama", "llama3.2")

# Everything else stays the same!
messages = [{"role": "user", "content": "Hello!"}]
response = model.chat_complete(messages)
```

No code changes needed - just provider name and model!

## Common Configurations

### Streaming Responses

Get responses token by token:

```python
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"streaming": True}
)

messages = [{"role": "user", "content": "Write a haiku about coding"}]

for chunk in model.chat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### JSON Output

Request structured JSON responses:

```python
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three programming languages in JSON format"
}]

response = model.chat_complete(messages)
print(response.content)  # Valid JSON string
```

### Temperature Control

Adjust creativity (0.0 = deterministic, 2.0 = very creative):

```python
model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"temperature": 0.3}  # More focused
)

# Or per-request
response = model.chat_complete(messages, temperature=0.9)  # More creative
```

### Async Operations

For better performance with multiple requests:

```python
import asyncio
from esperanto.factory import AIFactory

async def main():
    model = AIFactory.create_language("openai", "gpt-4")

    messages = [{"role": "user", "content": "Hello!"}]

    # Async call
    response = await model.achat_complete(messages)
    print(response.content)

asyncio.run(main())
```

## Multi-Capability Example

Use multiple AI capabilities together:

```python
from esperanto.factory import AIFactory

# Create models for different capabilities
llm = AIFactory.create_language("openai", "gpt-4")
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")
speaker = AIFactory.create_text_to_speech("openai", "tts-1")

# 1. Generate text with LLM
messages = [{"role": "user", "content": "Explain quantum computing in one sentence"}]
explanation = llm.chat_complete(messages).content

# 2. Create embeddings for search
texts = [explanation, "Quantum computers use qubits"]
embeddings = embedder.embed(texts)

# 3. Convert to speech
audio = speaker.generate_speech(explanation, voice="nova")
with open("explanation.mp3", "wb") as f:
    f.write(audio)

print(f"Generated explanation: {explanation}")
print(f"Created {len(embeddings.data)} embeddings")
print("Saved audio to explanation.mp3")
```

## Local Models (No API Costs!)

Use local models for privacy and zero API costs:

```python
from esperanto.factory import AIFactory

# Local LLM with Ollama (requires ollama installed)
llm = AIFactory.create_language("ollama", "llama3.2")

# Local embeddings with Transformers
embedder = AIFactory.create_embedding(
    "transformers",
    "BAAI/bge-base-en-v1.5"
)

# Local reranking
reranker = AIFactory.create_reranker(
    "transformers",
    "BAAI/bge-reranker-base"
)

# Use exactly like cloud models!
response = llm.chat_complete([{"role": "user", "content": "Hello!"}])
```

## RAG (Retrieval-Augmented Generation) Pipeline

Complete RAG in 20 lines:

```python
from esperanto.factory import AIFactory

# Setup models
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")
reranker = AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")
llm = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")

# Your knowledge base
documents = [
    "Esperanto is a universal AI interface for Python",
    "It supports 17 different AI providers",
    "You can switch providers without changing code"
]

# User query
query = "What is Esperanto?"

# Step 1: Embed and retrieve (simplified - normally you'd use vector DB)
doc_embeddings = embedder.embed(documents)
query_embedding = embedder.embed([query])
# ... compute similarity and get top candidates ...

# Step 2: Rerank for accuracy
reranked = reranker.rerank(query, documents, top_k=2)
context = "\n".join([r.document for r in reranked.results])

# Step 3: Generate answer with LLM
messages = [{
    "role": "user",
    "content": f"Context:\n{context}\n\nQuestion: {query}"
}]
answer = llm.chat_complete(messages)

print(answer.content)
```

## Error Handling

Always handle potential errors:

```python
from esperanto.factory import AIFactory

try:
    model = AIFactory.create_language("openai", "gpt-4")
    messages = [{"role": "user", "content": "Hello!"}]
    response = model.chat_complete(messages)
    print(response.content)

except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Environment Setup Best Practices

Create a `.env` file for your API keys:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...

# Optional timeout overrides
ESPERANTO_LLM_TIMEOUT=90
ESPERANTO_EMBEDDING_TIMEOUT=120
```

Then load in your Python code:

```python
from dotenv import load_dotenv
load_dotenv()

# API keys are now available to Esperanto
```

## Next Steps

Now that you've got the basics, explore more:

### Learn Capabilities
- **[Language Models Guide](./capabilities/llm.md)** - Complete LLM documentation
- **[Embeddings Guide](./capabilities/embedding.md)** - Semantic search and vectors
- **[Reranking Guide](./capabilities/reranking.md)** - Improve search relevance
- **[Speech-to-Text Guide](./capabilities/speech-to-text.md)** - Audio transcription
- **[Text-to-Speech Guide](./capabilities/text-to-speech.md)** - Voice generation

### Choose Providers
- **[Provider Comparison](./providers/README.md)** - Compare all 17 providers
- **[Provider Setup Guides](./providers/)** - Detailed setup for each provider

### Advanced Features
- **[Task-Aware Embeddings](./advanced/task-aware-embeddings.md)** - Optimize for specific tasks
- **[LangChain Integration](./advanced/langchain-integration.md)** - Use with LangChain
- **[Timeout Configuration](./advanced/timeout-configuration.md)** - Control request timeouts
- **[Model Discovery](./advanced/model-discovery.md)** - Discover available models
- **[Transformers Features](./advanced/transformers-features.md)** - Advanced local model features

### Configuration
- **[Configuration Guide](./configuration.md)** - Complete configuration reference

## Get Help

- **[Documentation Index](./README.md)** - All documentation
- **[GitHub Issues](https://github.com/lfnovo/esperanto/issues)** - Report bugs or ask questions
- **[Changelog](../CHANGELOG.md)** - Version history

---

**Questions?** Check the [Documentation Index](./README.md) or [Provider Comparison](./providers/README.md).
