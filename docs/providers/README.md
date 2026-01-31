# Provider Guide

Welcome to the Esperanto provider guide. This page helps you choose the right AI provider for your needs.

## Provider Support Matrix

| Provider | LLM | Embedding | Reranking | Speech-to-Text | Text-to-Speech | JSON Mode |
|----------|-----|-----------|-----------|----------------|----------------|-----------|
| [OpenAI](./openai.md) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| [OpenAI-Compatible](./openai-compatible.md) | ✅ | ✅ | ❌ | ✅ | ✅ | ⚠️* |
| [Anthropic](./anthropic.md) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| [Google (GenAI)](./google.md) | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| [Vertex AI](./vertex.md) | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [Azure OpenAI](./azure.md) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| [Groq](./groq.md) | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| [Ollama](./ollama.md) | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| [Mistral](./mistral.md) | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| [DeepSeek](./deepseek.md) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| [DashScope](./dashscope.md) | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| [Perplexity](./perplexity.md) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| [xAI](./xai.md) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| [OpenRouter](./openrouter.md) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| [Transformers](./transformers.md) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [Jina](./jina.md) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [Voyage](./voyage.md) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [ElevenLabs](./elevenlabs.md) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |

*⚠️ OpenAI-Compatible: JSON mode support depends on the specific endpoint implementation

## Quick Selection Guide

### By Use Case

#### Need Multi-Modal Capabilities?

**All-in-One Providers:**
- **[OpenAI](./openai.md)**: LLM + Embedding + STT + TTS (industry standard)
- **[Azure OpenAI](./azure.md)**: Same as OpenAI + enterprise compliance
- **[Google GenAI](./google.md)**: LLM + Embedding + TTS (competitive pricing)
- **[OpenAI-Compatible](./openai-compatible.md)**: Use different endpoints for different capabilities

**Partial Multi-Modal:**
- **[Groq](./groq.md)**: LLM + STT (fastest inference)
- **[Vertex AI](./vertex.md)**: LLM + Embedding + TTS (Google Cloud)
- **[ElevenLabs](./elevenlabs.md)**: STT + TTS (premium voice quality)

#### Need Privacy/Local Deployment?

**Fully Local:**
- **[Ollama](./ollama.md)**: LLM + Embedding (simple setup, no API costs)
- **[Transformers](./transformers.md)**: Embedding + Reranking (100+ models, offline)
- **[OpenAI-Compatible](./openai-compatible.md)**: Connect to local LM Studio, vLLM, or Ollama

**Self-Hosted Options:**
- **[Azure OpenAI](./azure.md)**: Private cloud deployment
- **[Vertex AI](./vertex.md)**: Google Cloud with VPC/security controls

#### Need Enterprise Features?

**Enterprise-Ready:**
- **[Azure OpenAI](./azure.md)**: SLA, compliance, private endpoints, regional control
- **[Vertex AI](./vertex.md)**: Google Cloud security, IAM, audit logs
- **[OpenAI](./openai.md)**: Enterprise tier available
- **[Mistral](./mistral.md)**: European data residency

#### Need Cost Optimization?

**Most Cost-Effective:**
- **[Ollama](./ollama.md)**: Free (local deployment)
- **[Transformers](./transformers.md)**: Free (local deployment)
- **[OpenAI-Compatible](./openai-compatible.md)**: Free (local deployment)
- **[DeepSeek](./deepseek.md)**: Low API costs, strong performance
- **[OpenRouter](./openrouter.md)**: Compare prices across providers, some free models

**Good Value:**
- **[Google GenAI](./google.md)**: Competitive pricing, generous free tier
- **[Groq](./groq.md)**: Fast + affordable

### By Capability

#### Language Models (LLM)

**Best Overall Quality:**
- **[OpenAI](./openai.md)**: GPT-4o, o1, o3, o4 (industry standard)
- **[Anthropic](./anthropic.md)**: Claude 3.5 (excellent reasoning, long context)

**Best for Reasoning:**
- **[Anthropic](./anthropic.md)**: Claude 3.5 Sonnet, Opus (complex reasoning)
- **[DeepSeek](./deepseek.md)**: deepseek-reasoner (step-by-step reasoning)
- **[OpenAI](./openai.md)**: o1, o3, o4 series (advanced reasoning)

**Fastest Inference:**
- **[Groq](./groq.md)**: LPU architecture, ultra-fast responses
- **[OpenAI](./openai.md)**: GPT-4o-mini, GPT-3.5 Turbo

**Best for Code:**
- **[OpenAI](./openai.md)**: GPT-4o, o1 series
- **[Anthropic](./anthropic.md)**: Claude 3.5 Sonnet
- **[DeepSeek](./deepseek.md)**: Strong coding capabilities

**Real-Time Information:**
- **[Perplexity](./perplexity.md)**: Web search integration, citations
- **[xAI](./xai.md)**: Real-time knowledge

**Multiple Models Access:**
- **[OpenRouter](./openrouter.md)**: 100+ models from various providers

**Local Deployment:**
- **[Ollama](./ollama.md)**: Llama, Mistral, Qwen, etc.
- **[OpenAI-Compatible](./openai-compatible.md)**: LM Studio, vLLM

#### Embeddings

**Best Overall:**
- **[OpenAI](./openai.md)**: text-embedding-3-large, text-embedding-3-small (industry standard)
- **[Voyage](./voyage.md)**: voyage-3 (specialized retrieval, 32K context)

**Advanced Features:**
- **[Jina](./jina.md)**: Native task types, late chunking, dimension control, multilingual
- **[Google](./google.md)**: Native task type support, 8 task types

**Local/Privacy:**
- **[Transformers](./transformers.md)**: 100+ HuggingFace models, completely offline
- **[Ollama](./ollama.md)**: Local models with simple setup

**Domain-Specific:**
- **[Voyage](./voyage.md)**: voyage-code-2 (code), voyage-law-2 (legal), voyage-finance-2 (finance)

**Multilingual:**
- **[Jina](./jina.md)**: jina-embeddings-v3 (multilingual excellence)
- **[Google](./google.md)**: text-multilingual-embedding-002
- **[Mistral](./mistral.md)**: mistral-embed (multilingual)

**Enterprise:**
- **[Azure](./azure.md)**: text-embedding-3-large, text-embedding-3-small (private cloud)
- **[Vertex AI](./vertex.md)**: text-embedding-004 (Google Cloud)

#### Reranking

**All Reranking Providers:**
- **[Jina](./jina.md)**: Multilingual (100+ languages), production-ready
- **[Voyage](./voyage.md)**: rerank-2, rerank-1 (high accuracy)
- **[Transformers](./transformers.md)**: Universal support (any CrossEncoder model), local/offline

**Best for Multilingual:**
- **[Jina](./jina.md)**: jina-reranker-v2-base-multilingual

**Best for Privacy:**
- **[Transformers](./transformers.md)**: Completely local processing

**Best for Accuracy:**
- **[Voyage](./voyage.md)**: Specialized for retrieval tasks

#### Speech-to-Text

**Best Overall:**
- **[OpenAI](./openai.md)**: Whisper-1 (accurate, multilingual)
- **[Groq](./groq.md)**: whisper-large-v3 (fastest transcription)

**Enterprise:**
- **[Azure](./azure.md)**: whisper (private cloud, compliance)

**Premium Quality:**
- **[ElevenLabs](./elevenlabs.md)**: Advanced speech recognition

**Local Deployment:**
- **[OpenAI-Compatible](./openai-compatible.md)**: faster-whisper, local Whisper

#### Text-to-Speech

**Best Voice Quality:**
- **[ElevenLabs](./elevenlabs.md)**: Premium voices, voice cloning, emotional control

**Best Overall:**
- **[OpenAI](./openai.md)**: tts-1, tts-1-hd (natural voices, good quality)

**Most Voices:**
- **[Google](./google.md)**: 30+ unique voices with personalities

**Enterprise:**
- **[Azure](./azure.md)**: Custom neural voices, private cloud
- **[Vertex AI](./vertex.md)**: Multi-speaker support, Google Cloud

**Local Deployment:**
- **[OpenAI-Compatible](./openai-compatible.md)**: Connect to local TTS endpoints

## Feature Comparison

### LLM Features

| Provider | Streaming | JSON Mode | Long Context | Max Context |
|----------|-----------|-----------|--------------|-------------|
| [OpenAI](./openai.md) | ✅ | ✅ | ✅ | 128K-200K |
| [Anthropic](./anthropic.md) | ✅ | ✅ | ✅ | 200K |
| [Google](./google.md) | ✅ | ✅ | ✅ | 2M (Gemini 1.5) |
| [Azure](./azure.md) | ✅ | ✅ | ✅ | 128K-200K |
| [Groq](./groq.md) | ✅ | ✅ | ❌ | 8K-32K |
| [Mistral](./mistral.md) | ✅ | ✅ | ✅ | 128K |
| [DeepSeek](./deepseek.md) | ✅ | ✅ | ✅ | 64K |
| [Perplexity](./perplexity.md) | ✅ | ✅ | ❌ | 32K |
| [xAI](./xai.md) | ✅ | ✅ | ✅ | 128K |
| [OpenRouter](./openrouter.md) | ✅ | ✅ | ✅ | Varies |
| [Ollama](./ollama.md) | ✅ | ❌ | ✅ | Model-dependent |
| [OpenAI-Compatible](./openai-compatible.md) | ✅ | ⚠️ | ⚠️ | Endpoint-dependent |

### Embedding Features

| Provider | Task Types | Late Chunking | Output Dimensions | Max Input |
|----------|-----------|---------------|-------------------|-----------|
| [OpenAI](./openai.md) | Emulated | ❌ | Some models | 8K tokens |
| [Google](./google.md) | Native (8 types) | ❌ | ❌ | 3K tokens |
| [Jina](./jina.md) | Native | Native | ✅ (64-1024) | 8K tokens |
| [Voyage](./voyage.md) | ❌ | ❌ | ❌ | 4K-32K tokens |
| [Azure](./azure.md) | Emulated | ❌ | Some models | 8K tokens |
| [Mistral](./mistral.md) | ❌ | ❌ | ❌ | 8K tokens |
| [Transformers](./transformers.md) | Emulated | Emulated | ❌ | Model-dependent |
| [Ollama](./ollama.md) | ❌ | ❌ | ❌ | Model-dependent |
| [Vertex AI](./vertex.md) | ❌ | ❌ | ❌ | 3K tokens |
| [OpenAI-Compatible](./openai-compatible.md) | ❌ | ❌ | ❌ | Endpoint-dependent |

## Getting Started

1. **Choose a provider** based on your requirements (see selection guide above)
2. **Read the provider page** for detailed setup instructions
3. **Get API credentials** (if required)
4. **Set environment variables** from `.env.example`
5. **Start coding** with the examples in each provider guide

## Environment Setup

See the root `.env.example` file for all available environment variables:

```bash
# Copy and customize
cp .env.example .env

# Edit with your API keys
nano .env
```

For detailed configuration, see [Configuration Guide](../configuration.md).

## Common Patterns

### Single Provider Setup

```python
from esperanto.factory import AIFactory

# All capabilities from one provider
llm = AIFactory.create_language("openai", "gpt-4")
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")
transcriber = AIFactory.create_speech_to_text("openai", "whisper-1")
speaker = AIFactory.create_text_to_speech("openai", "tts-1")
```

### Multi-Provider Setup

```python
from esperanto.factory import AIFactory

# Best-in-class for each capability
llm = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
embedder = AIFactory.create_embedding("jina", "jina-embeddings-v3")
reranker = AIFactory.create_reranker("voyage", "rerank-2")
speaker = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")
```

### Local/Cloud Hybrid

```python
from esperanto.factory import AIFactory

# Local models for privacy, cloud for specialized tasks
local_embedder = AIFactory.create_embedding("transformers", "BAAI/bge-large-en-v1.5")
cloud_llm = AIFactory.create_language("openai", "gpt-4")

# Or all local
local_llm = AIFactory.create_language("ollama", "llama3.2")
local_embedder = AIFactory.create_embedding("ollama", "nomic-embed-text")
```

### Cost-Optimized Setup

```python
from esperanto.factory import AIFactory

# Free local models
llm = AIFactory.create_language("ollama", "llama3.2")
embedder = AIFactory.create_embedding("transformers", "BAAI/bge-base-en-v1.5")
reranker = AIFactory.create_reranker("transformers", "BAAI/bge-reranker-base")

# Or cost-effective cloud
cheap_llm = AIFactory.create_language("deepseek", "deepseek-chat")
```

## Provider Categories

### Cloud API Providers

Require API keys, pay-per-use:
- [OpenAI](./openai.md)
- [Anthropic](./anthropic.md)
- [Google GenAI](./google.md)
- [Groq](./groq.md)
- [Mistral](./mistral.md)
- [DeepSeek](./deepseek.md)
- [DashScope](./dashscope.md)
- [Perplexity](./perplexity.md)
- [xAI](./xai.md)
- [OpenRouter](./openrouter.md)
- [Jina](./jina.md)
- [Voyage](./voyage.md)
- [ElevenLabs](./elevenlabs.md)

### Cloud Enterprise Providers

Enterprise features, private deployment:
- [Azure OpenAI](./azure.md)
- [Vertex AI](./vertex.md)

### Local/Self-Hosted Providers

No API costs, privacy-focused:
- [Ollama](./ollama.md)
- [Transformers](./transformers.md)
- [OpenAI-Compatible](./openai-compatible.md)

## Next Steps

- **Quick Start**: [docs/quickstart.md](../quickstart.md)
- **Configuration**: [docs/configuration.md](../configuration.md)
- **Capabilities**: [docs/capabilities/](../capabilities/)
- **Advanced Topics**: [docs/advanced/](../advanced/)

## See Also

- [Capability Guides](../capabilities/)
  - [Language Models](../capabilities/llm.md)
  - [Embeddings](../capabilities/embedding.md)
  - [Reranking](../capabilities/reranking.md)
  - [Speech-to-Text](../capabilities/speech-to-text.md)
  - [Text-to-Speech](../capabilities/text-to-speech.md)
