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

The factory wraps providers transparently:

- Remote providers (OpenAI, Anthropic, Grok, Ollama, etc.) keep their native chat payloads, but we enforce the shared `<out>...</out>` stop sequence by injecting it into provider configs.
- Local engines (`llamacpp`, `hf_local`) receive fully rendered prompts that include family-specific tokens (ChatML, [INST], Gemma turns, etc.).

## Extending

- Add new adapters under `brio_ext/adapters/` and register them in `brio_ext/registry.py`.
- If a provider needs a prompt-based caller, implement it in `brio_ext/providers/` and add it to `_LANGUAGE_OVERRIDES` in `brio_ext/factory.py`.
- Keep tests alongside the module under `brio_ext/tests/`.
