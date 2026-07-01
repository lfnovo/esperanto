# PayPerQ (PPQ)

## Overview

[PayPerQ](https://ppq.ai) (PPQ) is a pay-as-you-go AI gateway that exposes hundreds of models from many labs (OpenAI, Anthropic, Google, xAI, Qwen, DeepSeek, Mistral, and more) through a single OpenAI-compatible API. Instead of subscriptions, usage is billed per request, so a single API key gives access to language, embedding, speech-to-text, and text-to-speech models.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Full catalog of chat models via a single key |
| Embeddings | ✅ | `openai/text-embedding-3-small`, `openai/text-embedding-3-large` |
| Reranking | ❌ | Not available |
| Speech-to-Text | ✅ | Deepgram Nova (`nova-3`, `nova-2`) |
| Text-to-Speech | ✅ | Deepgram Aura + ElevenLabs voices |

**Official Documentation:** https://ppq.ai/api-docs

## Prerequisites

### Account Requirements
- A PayPerQ account with available credits
- An API key from the PayPerQ dashboard

### Getting API Keys
1. Visit https://ppq.ai and sign in
2. Open the API keys section of your account
3. Create a key (it looks like `sk-...`) and copy it

## Environment Variables

```bash
# PayPerQ API key (required)
PPQ_API_KEY="sk-..."

# Custom base URL (optional)
PPQ_BASE_URL="https://api.ppq.ai/v1"
```

**Default base URL:** `https://api.ppq.ai/v1`

> Note: PayPerQ's chat endpoint is also reachable at the API root
> (`https://api.ppq.ai/chat/completions`). The embedding and audio endpoints
> live under `/v1`, so Esperanto uses `https://api.ppq.ai/v1` as the base URL
> for every capability, which the chat endpoint also serves.

## Language Models

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("ppq", "claude-sonnet-5")  # or "auto", "gpt-5.4-mini", ...

messages = [{"role": "user", "content": "Explain quantum computing"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

PayPerQ proxies hundreds of chat models. Pass any model `id` returned by
`GET https://api.ppq.ai/v1/models` (or listed in the dashboard). Examples:

| Model | Provider | Best For |
|-------|----------|----------|
| `auto` | PayPerQ | Automatic routing to a suitable model (default) |
| `gpt-5.4-mini` | OpenAI | Fast, cost-effective tasks |
| `claude-sonnet-5` | Anthropic | Balanced performance, long context |
| `claude-haiku-4.5` | Anthropic | Low-latency, cheap tasks |

### Streaming

```python
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="")
```

### JSON Mode

```python
model = AIFactory.create_language(
    "ppq", "claude-sonnet-5",
    config={"structured": {"type": "json_object"}}
)
```

> PayPerQ forwards `response_format` and tool definitions to the selected model.
> Support is broad — OpenAI, Anthropic, and most other models honor both — though
> a few upstream open-weight models may ignore them.

### Tool Calling

```python
from esperanto.common_types import Tool, ToolFunction

tools = [
    Tool(function=ToolFunction(
        name="get_weather",
        description="Get weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    ))
]
response = model.chat_complete(messages, tools=tools)
```

## Embeddings

```python
embedder = AIFactory.create_embedding("ppq", "openai/text-embedding-3-small")
vectors = embedder.embed(["Hello world", "How are you?"])
print(len(vectors[0]))  # 1536
```

Available models: `openai/text-embedding-3-small` (default),
`openai/text-embedding-3-large`.

## Speech-to-Text

```python
stt = AIFactory.create_speech_to_text("ppq", "nova-3")
transcription = stt.transcribe("audio.mp3")
print(transcription.text)
```

Available models: `nova-3` (default), `nova-2` (Deepgram Nova).

## Text-to-Speech

```python
tts = AIFactory.create_text_to_speech("ppq", "deepgram_aura_2")
tts.generate_speech(
    "Hello from PayPerQ.",
    voice="aura-2-thalia-en",
    output_file="hello.mp3",
)
```

Available models: `deepgram_aura_2` (default), `eleven_multilingual_v2`,
`eleven_flash_v2_5`. Deepgram Aura voices include `aura-2-thalia-en`,
`aura-2-arcas-en`, `aura-2-andromeda-en`, `aura-2-helena-en`,
`aura-2-apollo-en`, and `aura-2-aries-en`.

## Configuration

```python
# Explicit API key
model = AIFactory.create_language("ppq", "claude-sonnet-5", config={"api_key": "your-key"})

# Custom base URL (e.g. a proxy)
model = AIFactory.create_language("ppq", "claude-sonnet-5", config={"base_url": "https://api.ppq.ai/v1"})
```

## Notes

- All PayPerQ capabilities use OpenAI-compatible endpoints under `https://api.ppq.ai/v1`, so standard Esperanto features (streaming, tool calling, JSON mode) work with models that support them.
- The language provider is implemented via Esperanto's OpenAI-compatible profile system; embeddings, STT, and TTS are thin provider classes over the OpenAI-compatible implementations. All four resolve under the `ppq` provider name.
- Because PayPerQ aggregates many upstream providers, per-model feature support depends on the model you select, not on PayPerQ itself.
- `GET https://api.ppq.ai/v1/models` lists the chat catalog; embedding/STT/TTS model ids are documented above and in the [API docs](https://ppq.ai/api-docs).
