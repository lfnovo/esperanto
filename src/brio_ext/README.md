# brio_ext – BrioDocs Extensions for Esperanto

This package adds BrioDocs-specific helpers on top of Esperanto’s provider stack. It keeps the upstream project untouched while supplying:

- prompt adapters for local/open-weight model families (Qwen, Llama, Mistral, Gemma, Phi);
- a renderer that normalises BrioDocs message payloads into provider-friendly prompts;
- local provider shims (`llamacpp`, `hf_local`) that speak the exact HTTP APIs those runtimes expose;
- `BrioAIFactory`, a drop-in replacement for Esperanto’s `AIFactory` that wires the renderer before each call.

## Usage

```python
from brio_ext.factory import BrioAIFactory

messages = [
    {
        "role": "system",
        "content": "You are a senior legal editor. Output MUST be wrapped in <out>...</out>.",
    },
    {
        "role": "user",
        "content": "TASK: Fix grammar.\n\nTEXT:\n<<<\nThe agrement is here.\n>>>",
    },
]

model = BrioAIFactory.create_language(
    provider="llamacpp",
    model_name="qwen2.5-7b-instruct",
    config={
        "base_url": "http://localhost:8080",
        "temperature": 0.25,
        "top_p": 0.8,
    },
)

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Custom Model Names with `chat_format`

If you're using custom model names that don't match standard patterns (e.g., "phi-4-mini-reasoning"), you can explicitly specify the chat format:

```python
model = BrioAIFactory.create_language(
    provider="llamacpp",
    model_name="phi-4-mini-reasoning",  # Custom model name
    config={
        "base_url": "http://localhost:8765",
        "chat_format": "chatml",  # Explicitly specify ChatML format
        "temperature": 0.5,
    },
)
```

Supported `chat_format` values:
- `"chatml"` or `"chat-ml"` – ChatML format (Qwen, Phi-4, etc.)
- `"llama"`, `"llama3"`, or `"llama-3"` – Llama native format
- `"mistral"` or `"mistral-instruct"` – Mistral format
- `"gemma"` – Gemma format

If `chat_format` is not provided, brio_ext will automatically detect the format based on the model name.

### LangChain / LangGraph Integration

Every model created via `BrioAIFactory` has a built-in `.to_langchain()` method:

```python
model = BrioAIFactory.create_language(
    provider="llamacpp",
    model_name="qwen2.5-7b-instruct",
    config={"base_url": "http://localhost:8080"},
)

# Get a LangChain-compatible wrapper
lc_model = model.to_langchain()

# Use with LangChain/LangGraph
result = lc_model.invoke("What is 2+2?")
print(result.content)  # Clean text, no <out> tags

# Async support
result = await lc_model.ainvoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Summarize this document."),
])
```

The wrapper preserves the full brio_ext rendering pipeline (chat templates, stop tokens, response cleaning) and automatically:
- Strips `<out>...</out>` fencing from responses
- Handles `<think>` tags from reasoning models (extracts useful content instead of returning empty strings)
- Converts LangChain message types to brio_ext format

## How It Works

The factory wraps providers transparently:

- Remote providers (OpenAI, Anthropic, Grok, Ollama, etc.) keep their native chat payloads, but we enforce the shared `<out>...</out>` stop sequence by injecting it into provider configs.
- Local engines (`llamacpp`, `hf_local`) receive fully rendered prompts that include family-specific tokens (ChatML, [INST], Gemma turns, etc.).

## Extending

- Add new adapters under `brio_ext/adapters/` and register them in `brio_ext/registry.py`.
- If a provider needs a prompt-based caller, implement it in `brio_ext/providers/` and add it to `_LANGUAGE_OVERRIDES` in `brio_ext/factory.py`.
- Keep tests alongside the module under `brio_ext/tests/`.

## Provider Smoke Tests

Optional integration tests hit real providers to verify end-to-end behaviour. They are skipped unless you opt in via environment variables:

```bash
BRIO_TEST_OPENAI_MODEL=gpt-4o-mini \
BRIO_TEST_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022 \
BRIO_TEST_GROQ_MODEL=groq/llama3-8b-8192-tool-use-preview \
pytest src/brio_ext/tests/integration/test_provider_smoke.py -q -m integration

# llama.cpp server (defaults to http://127.0.0.1:8765)
BRIO_TEST_LLAMACPP_MODEL=qwen2.5-7b-instruct \
pytest src/brio_ext/tests/integration/test_provider_smoke.py -q -m integration
```

For each provider, set `BRIO_TEST_<PROVIDER>_MODEL` (e.g. `BRIO_TEST_GROK_MODEL`,
`BRIO_TEST_MISTRAL_MODEL`). Optional overrides include:

- `BRIO_TEST_<PROVIDER>_PROVIDER` – custom provider string if you’re exercising a compatible endpoint.
- `BRIO_TEST_<PROVIDER>_CONFIG` – JSON blob merged into the provider config for fields such as deployments or API versions.
- `BRIO_TEST_<PROVIDER>_BASE_URL`, `BRIO_TEST_<PROVIDER>_MAX_TOKENS`, etc.

Refer to `src/brio_ext/tests/integration/test_provider_smoke.py` for the complete list.
