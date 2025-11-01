# Esperanto Documentation

Welcome to the Esperanto documentation! Esperanto provides a unified interface for interacting with multiple AI model providers through a consistent API.

## 🚀 Quick Links

- **[Quick Start Guide](./quickstart.md)** - Get started in 5 minutes
- **[Provider Comparison](./providers/README.md)** - Choose the right provider
- **[Configuration Guide](./configuration.md)** - Environment setup
- **[Main README](../README.md)** - Project overview and installation

## 📚 Documentation Structure

### Capabilities (What Can I Do?)

Learn about each AI capability and how to use it:

- **[Language Models (LLM)](./capabilities/llm.md)** - Text generation, chat, reasoning
- **[Embeddings](./capabilities/embedding.md)** - Vector representations for semantic search
- **[Reranking](./capabilities/reranking.md)** - Improve search relevance
- **[Speech-to-Text](./capabilities/speech-to-text.md)** - Audio transcription
- **[Text-to-Speech](./capabilities/text-to-speech.md)** - Voice generation

### Providers (How Do I Set This Up?)

Complete setup guides for each provider:

**Cloud API Providers:**
- [OpenAI](./providers/openai.md) - GPT-4, Embeddings, Whisper, TTS
- [Anthropic](./providers/anthropic.md) - Claude models
- [Google (GenAI)](./providers/google.md) - Gemini, Embeddings, TTS
- [Groq](./providers/groq.md) - Fastest inference
- [Mistral](./providers/mistral.md) - European AI
- [DeepSeek](./providers/deepseek.md) - Cost-effective reasoning
- [Perplexity](./providers/perplexity.md) - Web-search LLM
- [xAI](./providers/xai.md) - Grok models
- [OpenRouter](./providers/openrouter.md) - 100+ models, one API
- [Jina](./providers/jina.md) - Advanced embeddings & reranking
- [Voyage](./providers/voyage.md) - Specialized retrieval
- [ElevenLabs](./providers/elevenlabs.md) - Premium voice quality

**Enterprise Providers:**
- [Azure OpenAI](./providers/azure.md) - Enterprise compliance
- [Vertex AI](./providers/vertex.md) - Google Cloud

**Local/Self-Hosted:**
- [Ollama](./providers/ollama.md) - Local model deployment
- [Transformers](./providers/transformers.md) - HuggingFace models
- [OpenAI-Compatible](./providers/openai-compatible.md) - LM Studio, vLLM, etc.

**[📊 Provider Comparison Matrix](./providers/README.md)** - Compare all providers

### Advanced Topics

Deep dives into specialized features:

- **[Task-Aware Embeddings](./advanced/task-aware-embeddings.md)** - Optimize embeddings for specific tasks
- **[Transformers Advanced Features](./advanced/transformers-features.md)** - Local model optimizations
- **[LangChain Integration](./advanced/langchain-integration.md)** - Use with LangChain
- **[Timeout Configuration](./advanced/timeout-configuration.md)** - Request timeout management
- **[Model Discovery](./advanced/model-discovery.md)** - Discover available models

## 🎯 Find What You Need

### By Goal

**"I want to generate text"**
→ [Language Models Guide](./capabilities/llm.md) → [Choose Provider](./providers/README.md)

**"I want semantic search"**
→ [Embeddings Guide](./capabilities/embedding.md) → [Reranking Guide](./capabilities/reranking.md)

**"I want voice capabilities"**
→ [Speech-to-Text Guide](./capabilities/speech-to-text.md) or [Text-to-Speech Guide](./capabilities/text-to-speech.md)

**"I need privacy/local deployment"**
→ [Ollama](./providers/ollama.md), [Transformers](./providers/transformers.md), or [OpenAI-Compatible](./providers/openai-compatible.md)

**"I need enterprise features"**
→ [Azure OpenAI](./providers/azure.md) or [Vertex AI](./providers/vertex.md)

**"I want the best quality"**
→ [OpenAI](./providers/openai.md) (GPT-4), [Anthropic](./providers/anthropic.md) (Claude), [ElevenLabs](./providers/elevenlabs.md) (voice)

**"I want the fastest responses"**
→ [Groq](./providers/groq.md)

**"I want cost optimization"**
→ [DeepSeek](./providers/deepseek.md), [Ollama](./providers/ollama.md), or [OpenRouter](./providers/openrouter.md)

### By Provider

**"How do I set up [Provider]?"**
→ Check [Providers directory](./providers/) for provider-specific guides

**"Which provider should I use?"**
→ [Provider Comparison Matrix](./providers/README.md)

**"What models are available?"**
→ See individual provider pages or use [Model Discovery](./advanced/model-discovery.md)

### By Feature

**"How do I use task-aware embeddings?"**
→ [Task-Aware Embeddings](./advanced/task-aware-embeddings.md)

**"How do I integrate with LangChain?"**
→ [LangChain Integration](./advanced/langchain-integration.md)

**"How do I configure timeouts?"**
→ [Timeout Configuration](./advanced/timeout-configuration.md)

**"How do I use local models?"**
→ [Transformers Advanced Features](./advanced/transformers-features.md)

## 💡 Common Patterns

### Single Provider Setup

Use one provider for all capabilities:

```python
from esperanto.factory import AIFactory

# OpenAI for everything
llm = AIFactory.create_language("openai", "gpt-4")
embedder = AIFactory.create_embedding("openai", "text-embedding-3-small")
transcriber = AIFactory.create_speech_to_text("openai", "whisper-1")
speaker = AIFactory.create_text_to_speech("openai", "tts-1")
```

### Multi-Provider Setup

Choose best-in-class for each capability:

```python
# Best reasoning
llm = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")

# Best embeddings with advanced features
embedder = AIFactory.create_embedding("jina", "jina-embeddings-v3")

# Best voice quality
speaker = AIFactory.create_text_to_speech("elevenlabs", "eleven_multilingual_v2")
```

### Local/Cloud Hybrid

Privacy-sensitive data locally, specialized tasks in cloud:

```python
# Local for privacy
local_llm = AIFactory.create_language("ollama", "llama3.2")
local_embedder = AIFactory.create_embedding("transformers", "BAAI/bge-large-en-v1.5")

# Cloud for specialized needs
cloud_llm = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
```

## 📖 Documentation Conventions

### Code Examples

All examples use the **Factory pattern** (recommended):

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("provider", "model-name")
```

Direct instantiation is also supported (see individual capability guides).

### Environment Variables

Configure providers via environment variables (see [Configuration Guide](./configuration.md)):

```bash
# Copy example file
cp .env.example .env

# Edit with your API keys
nano .env
```

See `.env.example` in project root for all available variables.

### Async Support

All methods have async equivalents with `a` prefix:

```python
# Sync
response = model.chat_complete(messages)

# Async
response = await model.achat_complete(messages)
```

## 🔄 Migration from Old Docs

The documentation has been restructured for better navigation:

**Old Structure:**
- docs/llm.md (all LLM providers mixed)
- docs/embedding/* (embedding-specific)
- docs/speech_to_text.md (all STT providers)
- docs/text_to_speech.md (all TTS providers)

**New Structure:**
- docs/capabilities/* (what each capability does)
- docs/providers/* (how to set up each provider)
- docs/advanced/* (specialized topics)

**Key Changes:**
- Provider-specific info now in dedicated provider pages
- Capability guides focus on API and usage patterns
- Environment variables documented per-provider
- Comparison matrices for easy provider selection

## 🆘 Getting Help

### Documentation Issues

- **Provider setup unclear?** → Check the provider page
- **Capability usage unclear?** → Check the capability guide
- **Feature not working?** → Check provider page troubleshooting section
- **Need examples?** → Every guide includes multiple examples

### Common Questions

**Q: Which provider should I use?**
→ See [Provider Comparison Matrix](./providers/README.md)

**Q: How do I get API keys?**
→ See Prerequisites section in each provider page

**Q: Can I use multiple providers?**
→ Yes! See Multi-Provider Setup above

**Q: Do I need to pay for everything?**
→ No! See [Ollama](./providers/ollama.md), [Transformers](./providers/transformers.md) for free options

**Q: How do I configure timeouts?**
→ See [Timeout Configuration](./advanced/timeout-configuration.md)

**Q: What about LangChain integration?**
→ See [LangChain Integration](./advanced/langchain-integration.md)

### External Resources

- **GitHub**: [github.com/lfnovo/esperanto](https://github.com/lfnovo/esperanto)
- **PyPI**: [pypi.org/project/esperanto](https://pypi.org/project/esperanto)
- **Issues**: [github.com/lfnovo/esperanto/issues](https://github.com/lfnovo/esperanto/issues)
- **Changelog**: [CHANGELOG.md](../CHANGELOG.md)

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to documentation or code.

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details.

---

**Ready to get started?** → [Quick Start Guide](./quickstart.md)
