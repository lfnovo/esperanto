# Esperanto

A unified interface for various AI model providers, making it easy to work with different LLMs, Speech-to-Text, Text-to-Speech, and Embedding services.

## Features

- LLM support for OpenAI, Anthropic, Google Vertex AI, Gemini, Groq, and more
- Speech-to-Text services from Google Cloud and OpenAI
- Text-to-Speech capabilities
- Embedding generation from various providers
- Unified interface for all providers

## Installation

```bash
poetry add esperanto
```

## Usage

```python
from esperanto.providers.llm import openai, anthropic, gemini
from esperanto.providers.speech_to_text import google as stt
from esperanto.providers.embedding import vertex

# Using LLM
response = await openai.complete("Tell me a joke")

# Speech to Text
text = await stt.transcribe("audio.mp3")

# Generate embeddings
embedding = await vertex.embed("Some text to embed")
```

## Development

1. Clone the repository
2. Install dependencies:
```bash
poetry install
```

3. Run tests:
```bash
poetry run pytest
```

## License

MIT License
