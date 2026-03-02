# Esperanto Developer Guide

**Date:** 2025-12-20
**Version:** 2.7.1

---

## Getting Started

### Prerequisites

- Python 3.9 - 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Brio-AI/esperanto.git
cd esperanto

# Install dependencies with uv
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run tests
uv run pytest -v

# Quick import check
uv run python -c "from esperanto.factory import AIFactory; print('OK')"
uv run python -c "from brio_ext.factory import BrioAIFactory; print('OK')"
```

### Environment Variables (Optional)

For testing with real providers, set API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Dependency Management

This project uses **pyproject.toml** (modern Python standard) instead of requirements.txt.

| Command | What Gets Installed |
|---------|---------------------|
| `uv sync` | Core + dev dependencies (recommended) |
| `pip install -e .` | Core only (pydantic, httpx) |
| `pip install -e ".[dev]"` | Core + dev tools (pytest, ruff, mypy) |
| `pip install -e ".[transformers]"` | Core + ML/transformer libs |

All dependencies are defined in `pyproject.toml`:
- **Core** (lines 16-19): `pydantic`, `httpx`
- **Dev** (lines 66+): `pytest`, `ruff`, `mypy`, `pytest-asyncio`, etc.
- **Optional** (lines 21-31): `transformers`, `torch`, etc.

---

## Overview

Esperanto is a unified interface library for interacting with multiple AI model providers. It provides standardized access to 15+ LLM providers, embedding models, rerankers, speech-to-text, and text-to-speech capabilities.

**Key Packages:**
- `esperanto` - Core library with provider implementations
- `brio_ext` - BrioDocs-specific extensions (custom rendering, metrics, local model support)

> **Note:** This library handles AI provider interfaces only. Database connections (SurrealDB) and application logic live in the **BrioDocs main application**, which consumes this library as a dependency.

---

## Integration with BrioDocs

BrioDocs includes brio-esperanto as a **git submodule** for CI/CD release automation. This means:

- **BrioDocs pins to specific commits** - The BrioDocs team controls when to pull in new changes
- **Releases are reproducible** - Same commit = same build every time
- **No action needed from brio-esperanto developers** - Just keep developing normally

### For BrioDocs Developers

To update brio-esperanto in BrioDocs:

```bash
cd BrioDocs/external/brio-esperanto
git pull origin main
cd ../..
git add external/brio-esperanto
git commit -m "Update brio-esperanto to latest"
git push
```

### Versioning

Use git tags for releases (e.g., `v2.7.1`) so BrioDocs can pin to specific versions rather than commit hashes:

```bash
git tag v2.7.1
git push origin v2.7.1
```

---

## Project Structure

```
src/
├── esperanto/
│   ├── factory.py                 # AIFactory - main entry point
│   ├── common_types/              # Shared data types (ChatCompletion, Message, etc.)
│   ├── providers/
│   │   ├── llm/                   # Language model providers
│   │   │   ├── base.py            # LanguageModel abstract base class
│   │   │   ├── openai.py
│   │   │   ├── anthropic.py
│   │   │   └── ...
│   │   ├── embedding/             # Embedding providers
│   │   ├── reranker/              # Reranker providers
│   │   ├── stt/                   # Speech-to-text providers
│   │   └── tts/                   # Text-to-speech providers
│   └── utils/
│       ├── logging.py
│       └── timeout.py             # TimeoutMixin for all providers
│
└── brio_ext/
    ├── factory.py                 # BrioAIFactory (extends AIFactory)
    ├── langchain_wrapper.py       # LangChain/LangGraph compatibility wrapper
    ├── registry.py                # Adapter selection logic
    ├── renderer.py                # Prompt rendering dispatcher
    ├── adapters/                  # Chat template adapters (Qwen, Llama, etc.)
    ├── providers/                 # Local model providers (LlamaCpp, HF Local)
    └── metrics/
        └── logger.py              # Performance metrics logging
```

---

## Registry Architecture

### AIFactory (`esperanto/factory.py`)

The `AIFactory` uses a **static dictionary-based registry** with dynamic imports:

```python
_provider_modules = {
    "language": {
        "openai": "esperanto.providers.llm.openai:OpenAILanguageModel",
        "anthropic": "esperanto.providers.llm.anthropic:AnthropicLanguageModel",
        # ... 15+ providers
    },
    "embedding": { ... },
    "speech_to_text": { ... },
    "text_to_speech": { ... },
    "reranker": { ... }
}
```

**How it works:**
1. User calls `AIFactory.create_language("openai", "gpt-4")`
2. Factory looks up `"openai"` in `_provider_modules["language"]`
3. `importlib` dynamically loads `esperanto.providers.llm.openai:OpenAILanguageModel`
4. Provider class is instantiated with provided config
5. Returns ready-to-use provider instance

**Factory Methods:**
- `create_language(provider, model_name, config)` → LanguageModel
- `create_embedding(provider, model_name, config)` → EmbeddingModel
- `create_reranker(provider, model_name, config)` → RerankerModel
- `create_speech_to_text(provider, model_name, config)` → SpeechToTextModel
- `create_text_to_speech(provider, model_name, config)` → TextToSpeechModel

### Adapter Registry (`brio_ext/registry.py`)

For local models, the adapter registry handles chat template rendering:

```python
ADAPTERS = (
    QwenAdapter(),
    LlamaAdapter(),
    MistralAdapter(),
    GemmaAdapter(),
    PhiAdapter(),
)
```

**Lookup order:**
1. If `chat_format` parameter provided → use exact format match
2. Otherwise → iterate adapters, call `adapter.can_handle(model_id)`
3. Return first matching adapter or None

---

## Adding a New Provider

### Step 1: Create Provider Class

Create file: `src/esperanto/providers/llm/newprovider.py`

```python
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Generator

import httpx

from esperanto.providers.llm.base import LanguageModel
from esperanto.common_types import ChatCompletion, ChatCompletionChunk, Model, Message, Choice, Usage


@dataclass
class NewProviderLanguageModel(LanguageModel):
    """NewProvider language model implementation."""

    def __post_init__(self):
        super().__post_init__()
        # Set credentials (priority: direct param > config > env var)
        self.api_key = self.api_key or os.getenv("NEWPROVIDER_API_KEY")
        self.base_url = self.base_url or "https://api.newprovider.com/v1"

        if not self.api_key:
            raise ValueError("NEWPROVIDER_API_KEY is required")

        self._create_http_clients()

    @property
    def provider(self) -> str:
        return "newprovider"

    def _get_default_model(self) -> str:
        return "newprovider-default-model"

    def _get_provider_type(self) -> str:
        return "language"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise RuntimeError(f"NewProvider API error: {response.text}")

    @property
    def models(self) -> List[Model]:
        response = self.client.get(
            f"{self.base_url}/models",
            headers=self._get_headers()
        )
        self._handle_error(response)
        data = response.json()
        return [
            Model(id=m["id"], owned_by="newprovider", type="language")
            for m in data.get("models", [])
        ]

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._get_headers()
        )
        self._handle_error(response)
        return self._normalize_response(response.json())

    async def achat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None
    ):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = await self.async_client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._get_headers()
        )
        self._handle_error(response)
        return self._normalize_response(response.json())

    def _normalize_response(self, data: Dict[str, Any]) -> ChatCompletion:
        """Convert provider response to standard ChatCompletion."""
        return ChatCompletion(
            id=data["id"],
            model=data.get("model", self.model_name),
            choices=[
                Choice(
                    index=c["index"],
                    message=Message(
                        role=c["message"]["role"],
                        content=c["message"]["content"]
                    ),
                    finish_reason=c.get("finish_reason")
                )
                for c in data["choices"]
            ],
            usage=Usage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"]
            ) if data.get("usage") else None
        )
```

### Step 2: Register in AIFactory

Edit `src/esperanto/factory.py`:

```python
_provider_modules = {
    "language": {
        # ... existing providers ...
        "newprovider": "esperanto.providers.llm.newprovider:NewProviderLanguageModel",
    },
    # ...
}
```

### Step 3: Create Tests

Create file: `tests/providers/llm/test_newprovider_provider.py`

```python
import pytest
from unittest.mock import Mock, patch

from esperanto.providers.llm.newprovider import NewProviderLanguageModel
from esperanto.common_types import ChatCompletion


@pytest.fixture
def mock_response():
    return {
        "id": "chat-123",
        "model": "newprovider-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


@pytest.fixture
def mock_client(mock_response):
    client = Mock()
    response = Mock()
    response.status_code = 200
    response.json.return_value = mock_response
    client.post.return_value = response
    return client


def test_chat_complete(mock_client):
    with patch.dict("os.environ", {"NEWPROVIDER_API_KEY": "test-key"}):
        provider = NewProviderLanguageModel(model_name="test-model")
        provider.client = mock_client

        response = provider.chat_complete([
            {"role": "user", "content": "Hi"}
        ])

        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Hello!"
        assert response.usage.total_tokens == 15
```

### Step 4: Run Tests

```bash
uv run pytest -v tests/providers/llm/test_newprovider_provider.py
```

---

## Adding a New Model/Adapter

For local models that need custom chat templates:

### Step 1: Create Adapter

Create file: `src/brio_ext/adapters/newmodel_adapter.py`

```python
from brio_ext.adapters import ChatAdapter


class NewModelAdapter(ChatAdapter):
    """Adapter for NewModel chat template format."""

    def can_handle(self, model_id: str) -> bool:
        return "newmodel" in model_id.lower()

    def render(self, messages: list) -> dict:
        """Convert messages to NewModel prompt format."""
        prompt_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"[SYSTEM]{content}[/SYSTEM]")
            elif role == "user":
                prompt_parts.append(f"[USER]{content}[/USER]")
            elif role == "assistant":
                prompt_parts.append(f"[ASSISTANT]{content}[/ASSISTANT]")

        # Add generation prompt
        prompt_parts.append("[ASSISTANT]")

        return {
            "prompt": "\n".join(prompt_parts),
            "stop": ["[/ASSISTANT]", "[USER]"]
        }

    def clean_response(self, text: str) -> str:
        """Remove format markers from response."""
        return text.replace("[/ASSISTANT]", "").strip()
```

### Step 2: Register Adapter

Edit `src/brio_ext/registry.py`:

```python
from brio_ext.adapters.newmodel_adapter import NewModelAdapter

ADAPTERS = (
    QwenAdapter(),
    LlamaAdapter(),
    MistralAdapter(),
    GemmaAdapter(),
    PhiAdapter(),
    NewModelAdapter(),  # Add here
)
```

---

## Removing a Provider/Model

### Removing a Provider

1. Remove entry from `_provider_modules` in `src/esperanto/factory.py`
2. Delete provider file from `src/esperanto/providers/{type}/`
3. Delete corresponding tests from `tests/providers/{type}/`
4. Update documentation

### Removing an Adapter

1. Remove from `ADAPTERS` tuple in `src/brio_ext/registry.py`
2. Delete adapter file from `src/brio_ext/adapters/`
3. Update any tests that reference the adapter

---

## Configuration System

### Priority Order (highest to lowest)

1. **Direct constructor parameters**: `AIFactory.create_language("openai", api_key="...")`
2. **Config dictionary**: `config={"api_key": "...", "timeout": 120}`
3. **Environment variables**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
4. **Default values**: Defined in base classes

### Common Config Options

| Option | Type | Description |
|--------|------|-------------|
| `api_key` | str | Provider API key |
| `base_url` | str | API endpoint URL |
| `model_name` | str | Model identifier |
| `max_tokens` | int | Maximum response tokens (default: 850) |
| `temperature` | float | Sampling temperature (default: 1.0) |
| `timeout` | float | Request timeout in seconds |
| `streaming` | bool | Enable streaming responses |

### Environment Variables

Each provider has specific env vars. Common patterns:
- `{PROVIDER}_API_KEY` - API key
- `{PROVIDER}_BASE_URL` - Custom endpoint
- `ESPERANTO_LLM_TIMEOUT` - Global LLM timeout
- `ESPERANTO_EMBEDDING_TIMEOUT` - Global embedding timeout

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
uv run pytest -v

# Run specific provider tests
uv run pytest -v tests/providers/llm/test_openai_provider.py

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Test Structure

```
tests/
├── providers/
│   ├── llm/
│   │   ├── test_openai_provider.py
│   │   ├── test_anthropic_provider.py
│   │   └── ...
│   ├── embedding/
│   └── ...
└── conftest.py
```

### Mocking HTTP Clients

Always mock `httpx.Client` and `httpx.AsyncClient`:

```python
@pytest.fixture
def mock_client():
    client = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {"...": "..."}
    client.post.return_value = response
    return client

def test_something(mock_client):
    provider = SomeProvider(...)
    provider.client = mock_client  # Inject mock
    # ... test ...
```

---

## Common Types Reference

### ChatCompletion

```python
class ChatCompletion(BaseModel):
    id: str
    choices: List[Choice]
    usage: Optional[Usage]
    timings: Optional[Timings]
    model: Optional[str]
    created: Optional[int]
```

### Message

```python
class Message(BaseModel):
    content: Optional[str]
    role: str  # "system", "user", "assistant"
    function_call: Optional[Dict[str, Any]]
    tool_calls: Optional[List[Dict[str, Any]]]
```

### Usage

```python
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

---

## Design Principles

1. **Consistency First** - Same interface across all providers (core value proposition)
2. **Pure HTTP** - No vendor SDKs, only httpx for control and minimal dependencies
3. **Lazy Loading** - Providers loaded dynamically only when needed
4. **Configuration Cascading** - Clear priority: params > config > env > defaults
5. **Immutable Responses** - All response objects are frozen Pydantic models
6. **Extensible Adapters** - Chat templates without modifying core providers

---

## LangChain Integration

Models created via `BrioAIFactory` automatically expose a `.to_langchain()` method:

```python
from brio_ext.factory import BrioAIFactory

model = BrioAIFactory.create_language("llamacpp", "qwen2.5-7b-instruct", config={...})
lc_model = model.to_langchain()

# Sync
result = lc_model.invoke("What is 2+2?")

# Async
result = await lc_model.ainvoke(messages)

# result.content is clean text (no <out> tags, no <think> content)
```

The `BrioLangChainWrapper` (in `src/brio_ext/langchain_wrapper.py`):
- Calls brio_ext's `chat_complete()` preserving the full rendering pipeline
- Strips `<out>...</out>` fencing from responses
- Handles `<think>` tags from reasoning models that wrap all output in think tags
- Converts LangChain message types (HumanMessage, SystemMessage) to brio_ext format
- Returns `_AIMessage` objects compatible with LangChain/LangGraph

---

## Quick Reference

| Task | Location |
|------|----------|
| Add language provider | `src/esperanto/providers/llm/` + register in `factory.py` |
| Add embedding provider | `src/esperanto/providers/embedding/` + register in `factory.py` |
| Add chat adapter | `src/brio_ext/adapters/` + register in `registry.py` |
| Modify factory logic | `src/esperanto/factory.py` or `src/brio_ext/factory.py` |
| Use LangChain wrapper | `model.to_langchain()` or `src/brio_ext/langchain_wrapper.py` |
| Add common types | `src/esperanto/common_types/` |
| Configure timeouts | `src/esperanto/utils/timeout.py` |
| Add metrics | `src/brio_ext/metrics/logger.py` |
