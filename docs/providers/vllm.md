# vLLM

## Overview

[vLLM](https://docs.vllm.ai/) is a high-performance inference server for LLMs. It supports the OpenAI-compatible chat completions API, so you can use it via Esperanto's `openai-compatible` provider with zero extra code — just point `base_url` at your vLLM server.

vLLM's standout features include PagedAttention for efficient KV cache management, continuous batching, and first-class support for structured generation (guided JSON, regex, grammar).

## Quick Start

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "openai-compatible",
    "meta-llama/Llama-3-70b-Instruct",
    config={
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-required",  # vLLM does not require an API key by default
    },
)

messages = [{"role": "user", "content": "Explain PagedAttention in one paragraph."}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Common vLLM-Specific Parameters

vLLM accepts extra sampling and decoding parameters beyond the OpenAI spec. Pass them via `extra_body` so they are forwarded in the request body.

| Parameter | Type | Description |
|-----------|------|-------------|
| `top_k` | `int` | Sample from the top-k most likely tokens (e.g. `40`). |
| `min_p` | `float` | Minimum probability relative to the top token (e.g. `0.05`). Filters low-probability tokens. |
| `repetition_penalty` | `float` | Penalise repeated tokens. Values > 1 discourage repetition (e.g. `1.1`). |
| `guided_json` | `dict` | JSON schema for structured generation — vLLM guarantees the output matches the schema. |

Full list: [vLLM sampling params docs](https://docs.vllm.ai/en/latest/dev/sampling_params.html).

## Using `extra_body`

### Instance-level extras (applied to every request)

```python
model = AIFactory.create_language(
    "openai-compatible",
    "mistralai/Mistral-7B-Instruct-v0.3",
    config={
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-required",
        "extra_body": {
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
        },
    },
)

response = model.chat_complete(messages)
```

### Per-call extras (structured generation with guided_json)

```python
# Define the JSON schema your output must conform to
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

response = model.chat_complete(
    [{"role": "user", "content": "Give me a JSON object for a fictional person."}],
    extra_body={"guided_json": person_schema},
)

import json
person = json.loads(response.choices[0].message.content)
print(person["name"], person["age"])
```

For more structured output options (regex, grammar, choice), see the [vLLM structured outputs docs](https://docs.vllm.ai/en/latest/usage/structured_outputs.html).

### Merging instance and per-call extras

When you supply both instance-level and per-call `extra_body`, they are shallow-merged with the per-call values winning on key collision:

```python
model = AIFactory.create_language(
    "openai-compatible",
    "meta-llama/Llama-3-70b-Instruct",
    config={
        "base_url": "http://localhost:8000/v1",
        "api_key": "not-required",
        "extra_body": {"top_k": 40, "repetition_penalty": 1.1},
    },
)

# This call overrides top_k and adds guided_json; repetition_penalty stays at 1.1
response = model.chat_complete(
    messages,
    extra_body={"top_k": 99, "guided_json": person_schema},
)
```

## Async and Streaming

`extra_body` works identically with `achat_complete` (async) and `stream=True`:

```python
# Async
response = await model.achat_complete(messages, extra_body={"top_k": 40})

# Streaming
for chunk in model.chat_complete(messages, stream=True, extra_body={"top_k": 40}):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Reserved Keys in `extra_body`

Some keys are stripped from `extra_body` before being sent to vLLM and must be passed via the dedicated arguments instead:

- `stream` — use the `stream=` argument on `chat_complete` / `achat_complete`. The response-parsing branch is chosen from that argument, so flipping the wire value via `extra_body` would desync request mode from parsing.
- `tools`, `tool_choice`, `parallel_tool_calls` — use the `tools=` / `tool_choice=` / `parallel_tool_calls=` arguments. Tool-call validation (when `validate_tool_calls=True`) checks responses against the resolved tool set, so overriding via `extra_body` would validate against the wrong schema.

If you include any of these inside `extra_body`, they are dropped with a debug log; everything else passes through unchanged.

## See Also

- [OpenAI-Compatible Provider](./openai-compatible.md) — full `extra_body` docs and other advanced features
- [vLLM Sampling Parameters](https://docs.vllm.ai/en/latest/dev/sampling_params.html)
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/usage/structured_outputs.html)
