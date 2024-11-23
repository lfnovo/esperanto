# Language Models (LLM)

Esperanto provides a unified interface for various Language Model providers. This guide explains how to use language models with different providers.

## Interface

All language models implement the following interface:

```python
async def complete(self, prompt: str) -> str:
    """Generate text completion for the given prompt."""

async def chat(self, messages: List[Message]) -> str:
    """Generate chat completion for the given messages."""

def to_langchain(self) -> BaseChatModel:
    """Convert to a LangChain chat model (requires langchain extra)."""
```

## Basic Usage

```python
from esperanto.factory import AIFactory

# Create a language model instance
llm = AIFactory.create_llm(
    provider="openai",  # Choose your provider
    model_name="gpt-4",  # Model name specific to the provider
    config={
        "temperature": 0.7,  # Optional: Control randomness (0.0 to 1.0)
        "max_tokens": 500,   # Optional: Maximum tokens to generate
        "top_p": 0.9,       # Optional: Nucleus sampling parameter
        "streaming": True,   # Optional: Enable streaming responses
    }
)

# Text completion
response = await llm.complete("Explain quantum computing")

# Chat completion
from esperanto.base.types import Message, Role

messages = [
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="What is quantum computing?"),
]
response = await llm.chat(messages)
```

## Supported Providers

### OpenAI
```python
llm = AIFactory.create_llm(
    provider="openai",
    model_name="gpt-4",  # or gpt-3.5-turbo, etc.
    config={
        "api_key": "your-api-key",  # Optional: defaults to OPENAI_API_KEY env var
        "organization": "org-id",    # Optional: your OpenAI organization ID
    }
)
```

### Anthropic
```python
llm = AIFactory.create_llm(
    provider="anthropic",
    model_name="claude-3-opus-20240229",  # or other Claude models
    config={
        "api_key": "your-api-key",  # Optional: defaults to ANTHROPIC_API_KEY env var
    }
)
```

### Google (Gemini)
```python
llm = AIFactory.create_llm(
    provider="gemini",
    model_name="gemini-pro",
    config={
        "api_key": "your-api-key",  # Optional: defaults to GEMINI_API_KEY env var
    }
)
```

### Groq
```python
llm = AIFactory.create_llm(
    provider="groq",
    model_name="mixtral-8x7b-32768",  # or other models
    config={
        "api_key": "your-api-key",  # Optional: defaults to GROQ_API_KEY env var
    }
)
```

### Ollama
```python
llm = AIFactory.create_llm(
    provider="ollama",
    model_name="llama2",  # or any other model you have pulled
    config={
        "base_url": "http://localhost:11434",  # Optional: Ollama server URL
    }
)
```

### LiteLLM
```python
llm = AIFactory.create_llm(
    provider="litellm",
    model_name="gpt-4",  # Model identifier for the backend provider
    config={
        "api_key": "your-api-key",
        "api_base": "your-api-base",  # Optional: API base URL
    }
)
```

### Vertex AI
```python
llm = AIFactory.create_llm(
    provider="vertex",
    model_name="text-bison",  # or other Vertex AI models
    config={
        "project": "your-project",    # Optional: defaults to VERTEX_PROJECT env var
        "location": "us-central1",    # Optional: defaults to VERTEX_LOCATION env var
    }
)
```

### OpenRouter
```python
llm = AIFactory.create_llm(
    provider="openrouter",
    model_name="openai/gpt-4",  # or other supported models
    config={
        "api_key": "your-api-key",  # Optional: defaults to OPENROUTER_API_KEY env var
    }
)
```

## LangChain Integration

All language models can be converted to LangChain chat models using the `to_langchain()` method:

```python
# Requires the langchain extra: pip install "esperanto[langchain]"
llm = AIFactory.create_llm("openai", "gpt-4")
langchain_model = llm.to_langchain()

# Use with LangChain
from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
chain = prompt | langchain_model
result = await chain.ainvoke({"input": "What is quantum computing?"})
```

## Error Handling

```python
try:
    llm = AIFactory.create_llm("openai", "gpt-4")
    response = await llm.complete("Hello!")
except ImportError as e:
    print("Provider dependencies not installed:", e)
except ValueError as e:
    print("Invalid configuration:", e)
except Exception as e:
    print("Error during completion:", e)
```
