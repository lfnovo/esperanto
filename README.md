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

For detailed documentation on each component, please refer to the following guides:

- [Language Models (LLM)](docs/llm.md): Comprehensive guide to using Language Models
- [Speech-to-Text (STT)](docs/speech_to_text.md): Guide to Speech-to-Text services
- [Text-to-Speech (TTS)](docs/text_to_speech.md): Guide to Text-to-Speech services
- [Embeddings](docs/embeddings.md): Guide to using Embedding Models
- [LangChain Integration](docs/langchain.md): Guide to using Esperanto with LangChain

## 🚀 Quick Start

```bash
# Install using poetry
poetry add esperanto

# Or using pip
pip install esperanto
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

## 🛠️ Installation

Esperanto can be installed using either Poetry (recommended) or pip. The base package is lightweight, including only core dependencies (pydantic and loguru). Additional features and providers can be installed as needed.

### Using Poetry

#### Basic Installation
```bash
poetry install
```

#### Installing LangChain Support
If you want to use LangChain features:
```bash
poetry install --with langchain
```

#### Installing Provider Dependencies
Install only the providers you need (each provider automatically includes LangChain):

```bash
# OpenAI (GPT models, Whisper, TTS)
poetry install --with openai

# Anthropic (Claude models)
poetry install --with anthropic

# Google (Gemini, Vertex AI, Speech)
poetry install --with google

# Groq
poetry install --with groq

# LiteLLM
poetry install --with litellm

# Ollama
poetry install --with ollama

# ElevenLabs (TTS)
poetry install --with elevenlabs
```

You can install multiple providers at once:
```bash
poetry install --with openai,anthropic,google
```

### Using pip

#### Basic Installation
```bash
pip install esperanto
```

#### Installing LangChain Support
```bash
pip install "esperanto[langchain]"
```

#### Installing Provider Dependencies
Install providers using pip's "extras" syntax (each provider automatically includes LangChain):

```bash
# OpenAI (GPT models, Whisper, TTS)
pip install "esperanto[openai]"

# Anthropic (Claude models)
pip install "esperanto[anthropic]"

# Google (Gemini, Vertex AI, Speech)
pip install "esperanto[google]"

# Groq
pip install "esperanto[groq]"

# LiteLLM
pip install "esperanto[litellm]"

# Ollama
pip install "esperanto[ollama]"

# ElevenLabs (TTS)
pip install "esperanto[elevenlabs]"
```

You can install multiple providers at once:
```bash
pip install "esperanto[openai,anthropic,google]"
```

To install all providers:
```bash
pip install "esperanto[all]"
```

If you try to use a provider without installing its dependencies, Esperanto will raise a helpful error message indicating which dependencies need to be installed.

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

## 🧪 Development

```bash
# Clone the repository
git clone https://github.com/yourusername/esperanto.git
cd esperanto

# Install dependencies
poetry install

# Run tests
poetry run pytest
```

## 🤝 Contributing

We welcome contributions! Please check our [Contributing Guidelines](link-to-contributing) for details on how to get started.

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to all the AI providers and the open-source community that make this project possible.
