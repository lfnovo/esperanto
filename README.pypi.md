# Esperanto 🌍

A powerful, unified interface for AI services that simplifies working with multiple providers. Esperanto provides a consistent API for Language Models (LLMs), Speech-to-Text, Text-to-Speech, and Embedding services, making it easy to switch between providers or use them in combination.

## 🌟 Key Features

- **Unified Interface**: Write once, use anywhere - switch providers without changing your code
- **Extensive Provider Support**:
  - **LLMs**: OpenAI, Anthropic, Google (Vertex AI & Gemini), Groq, Ollama, and more
  - **Speech-to-Text**: OpenAI Whisper, Google Cloud Speech
  - **Text-to-Speech**: ElevenLabs, OpenAI TTS, Google Cloud TTS
  - **Embeddings**: Various providers through a consistent interface
- **Provider-Specific Features**: Access unique capabilities of each provider while maintaining a consistent API
- **LangChain Integration**: Seamlessly convert models to LangChain format for advanced workflows
- **Async Support**: Built for modern, high-performance applications
- **Type Safety**: Full type hints and Pydantic models for better development experience

## 📚 Documentation

For detailed documentation on each component, please refer to the [GitHub repository](https://github.com/lfnovo/esperanto#readme).

## 🚀 Quick Start

```bash
# Install using poetry (minimal installation)
poetry add esperanto

# Install all providers using poetry
poetry add "esperanto[all]"

# Or using pip (minimal installation)
pip install esperanto

# Install all providers using pip
pip install "esperanto[all]"
```

## 💡 Usage Examples

### Language Models (LLMs)

```python
from esperanto.factory import AIFactory

# OpenAI
llm = AIFactory.create_llm(
    provider="openai",
    model_name="gpt-4",
    config={"temperature": 0.7}
)
response = await llm.complete("What's the weather like?")

# Anthropic
llm = AIFactory.create_llm(
    provider="anthropic",
    model_name="claude-3-opus-20240229",
    config={"max_tokens": 1000}
)
response = await llm.complete("Explain quantum computing")

# Google Gemini
llm = AIFactory.create_llm(
    provider="gemini",
    model_name="gemini-pro",
    config={"temperature": 0.9}
)
response = await llm.complete("Translate to French")
```

### Speech Services

```python
# Speech to Text
stt = AIFactory.create_stt(
    provider="openai",
    model_name="whisper-1"
)
text = await stt.transcribe("audio.mp3")

# Text to Speech
tts = AIFactory.create_tts(
    provider="elevenlabs",
    config={"voice": "Adam"}
)
audio = await tts.synthesize("Hello, world!")
```

### Easy Provider Switching

One of the key benefits of using the factory is how easy it is to switch between providers:

```python
# Using OpenAI
llm = AIFactory.create_llm("openai", "gpt-4")
response = await llm.complete("Explain AI")

# Switch to Anthropic by just changing the provider
llm = AIFactory.create_llm("anthropic", "claude-3-opus-20240229")
response = await llm.complete("Explain AI")  # Same code, different provider
```

## 🛠️ Configuration

Esperanto supports configuration through environment variables or direct configuration in code:

```python
# Environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export ELEVENLABS_API_KEY="your-key"

# Or in code
model = AIFactory.create_llm(
    provider="openai",
    model_name="gpt-4",
    config={
        "api_key": "your-key",
        "temperature": 0.7,
        "max_tokens": 500
    }
)
```

## 🤝 Contributing

We welcome contributions! Please check our [Contributing Guidelines](https://github.com/lfnovo/esperanto/blob/main/CONTRIBUTING.md) for details on how to get started.

## 📄 License

MIT License - see the [LICENSE](https://github.com/lfnovo/esperanto/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to all the AI providers and the open-source community that make this project possible.
