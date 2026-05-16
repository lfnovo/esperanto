# Multimodal Input (Images)

## Overview

Esperanto supports **multimodal input** — sending images alongside text in `chat_complete` / `achat_complete` calls. The canonical format is **OpenAI's content array**, which Esperanto translates to each provider's native format automatically.

Define your message once, and use it with OpenAI, Anthropic, Google, Vertex AI, Ollama, and any OpenAI-compatible provider — identical code.

> **Image input only.** Audio, PDF, and video input to the LLM are not yet supported. File extraction (FFmpeg frame extraction, PDF page rendering, etc.) belongs in higher-level libraries like [content-core](https://github.com/lfnovo/content-core) — Esperanto is a transport layer, not a media pipeline.

## Quick Start

```python
from esperanto import AIFactory

model = AIFactory.create_language("openai", "gpt-4o")

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    ],
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

The same `messages` list works with any multimodal-capable provider — just change the `provider` argument:

```python
model = AIFactory.create_language("anthropic", "claude-sonnet-4-20250514")
model = AIFactory.create_language("google", "gemini-2.0-flash")
model = AIFactory.create_language("ollama", "llava")
# ... same chat_complete(messages) call
```

## Canonical Format

`Message.content` accepts either a string (existing behavior, unchanged) or a list of content-block dicts in OpenAI's format:

```python
content = [
    {"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {"url": "..."}},
]
```

### URL Forms

`image_url.url` accepts:

- **`data:` URLs** with inline base64 — universal; works on every provider.

  ```python
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
  ```

- **`https://` URLs** — provider-dependent. OpenAI/Anthropic accept arbitrary public URLs. Google/Vertex/Ollama don't, so Esperanto downloads and inlines them automatically.

  ```python
  {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
  ```

### Why Dicts Instead of Typed Classes?

The list-of-dicts form passes through to OpenAI verbatim. New OpenAI block types (e.g. additions to the content array spec) work immediately without requiring an Esperanto release.

## Helper Utilities

For convenience, Esperanto exposes three helpers in `esperanto.utils.vision` (also re-exported from the package root):

### `encode_image_base64(file_path)`

Read an image file and return `(base64_data, mime_type)` with automatic MIME detection.

```python
from esperanto import encode_image_base64

b64, mime = encode_image_base64("/path/to/photo.png")
# b64 = "iVBORw0KGgo..."
# mime = "image/png"
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`. Raises `FileNotFoundError` if the path doesn't exist, `ValueError` for unsupported formats.

### `image_to_content_part(source, mime_type=None, detail=None)`

Convert any image source to an OpenAI-format content part. Accepts a file path, an `http(s)://` URL, or a raw base64 string.

```python
from esperanto import image_to_content_part

# From a file path - MIME type auto-detected
part = image_to_content_part("/path/to/photo.png")
# {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

# From a URL - passed through
part = image_to_content_part("https://example.com/photo.jpg")
# {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}

# From raw base64 - mime_type required
part = image_to_content_part("iVBORw0KGgo...", mime_type="image/png")
# {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

# OpenAI-only "detail" hint
part = image_to_content_part("/path/to/photo.png", detail="high")
# {"type": "image_url", "image_url": {"url": "data:...", "detail": "high"}}
```

### `create_image_message(image_source, prompt, ...)`

Build a complete `{"role": "user", "content": [text, image]}` message in one call.

```python
from esperanto import AIFactory, create_image_message

model = AIFactory.create_language("openai", "gpt-4o")

message = create_image_message(
    "/path/to/photo.png",
    prompt="Describe this image in detail.",
)
response = model.chat_complete([message])
```

## Multiple Images

Add as many `image_url` blocks as the model supports:

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Compare these two images."},
        {"type": "image_url", "image_url": {"url": url_a}},
        {"type": "image_url", "image_url": {"url": url_b}},
    ],
}]
```

## Provider Translation

Esperanto translates the OpenAI content array to each provider's native format internally. Users do not need to format images differently per provider.

### OpenAI / Azure / OpenAI-Compatible

Pass-through. The content array is the native format for OpenAI, Groq, Mistral, xAI, Perplexity, OpenRouter, DeepSeek, and any OpenAI-compatible profile.

### Anthropic

Translates `image_url` blocks to Anthropic's content blocks:

```python
# Input (OpenAI format):
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}

# Sent to Anthropic:
{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/..."}}
```

For `https://` URLs, the URL source form is used where supported.

### Google / Vertex AI

Translates to Gemini parts. Since Gemini doesn't accept arbitrary public URLs, `https://` URLs are downloaded and inlined automatically:

```python
# Input (OpenAI format):
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}

# Sent to Gemini:
{"inline_data": {"mime_type": "image/jpeg", "data": "/9j/..."}}
```

### Ollama

Splits text and images into separate fields per Ollama's API:

```python
# Input (OpenAI format):
{"role": "user", "content": [
    {"type": "text", "text": "What's in this?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}},
]}

# Sent to Ollama:
{"role": "user", "content": "What's in this?", "images": ["/9j/..."]}
```

## Capability Detection

**Esperanto does not curate per-model capability flags.** If you send images to a non-multimodal model, the provider's API will return an error and Esperanto surfaces it verbatim. Use a multimodal model (e.g. `gpt-4o`, `claude-sonnet-4-*`, `gemini-2.0-flash`, `llava`).

## Async Usage

The async API works identically:

```python
import asyncio
from esperanto import AIFactory, create_image_message

async def main():
    model = AIFactory.create_language("openai", "gpt-4o")
    message = create_image_message("/path/to/photo.png", prompt="Describe this.")
    response = await model.achat_complete([message])
    print(response.choices[0].message.content)

asyncio.run(main())
```

## Provider Support

| Provider          | Image Input | Notes                                                                |
|-------------------|:-----------:|----------------------------------------------------------------------|
| OpenAI            |     Yes     | Native pass-through; supports `detail` hint.                         |
| Azure             |     Yes     | Native pass-through (uses OpenAI format).                            |
| Anthropic         |     Yes     | Translated to `{type: "image", source: {...}}`.                      |
| Google (Gemini)   |     Yes     | Translated to `inline_data`; `https://` URLs downloaded.             |
| Vertex AI         |     Yes     | Same as Google.                                                      |
| Ollama            |   Yes\*     | Split into `images` field. Requires a multimodal model (e.g. llava). |
| Groq              |     Yes     | Native pass-through (model-dependent).                               |
| Mistral           |     Yes     | Native pass-through (model-dependent).                               |
| xAI               |     Yes     | Native pass-through (model-dependent).                               |
| Perplexity        |     Yes     | Native pass-through (model-dependent).                               |
| OpenRouter        |     Yes     | Native pass-through (model-dependent).                               |
| DeepSeek          |     Yes     | Native pass-through (model-dependent).                               |
| OpenAI-compatible |     Yes     | Native pass-through (model-dependent).                               |

\* Only multimodal Ollama models accept images. The non-multimodal models will reject the request.

## Out of Scope

These are deferred to follow-up issues with their own demand signals:

- **Audio input** to LLMs (`{"type": "input_audio", ...}`)
- **PDF input** (Anthropic native PDF blocks, OpenAI Files API)
- **Video input** (Gemini video parts)
- **Image / video generation endpoints** — separate provider type
- **Capability detection** — per-model multimodal flags

For PDF / video extraction *into images* that you can then feed through this image-input surface, see [content-core](https://github.com/lfnovo/content-core).

## See Also

- [Tool Calling](tool-calling.md) — function calling across providers
- [Language Models (LLM)](../capabilities/llm.md) — core LLM documentation
- [Issue #82](https://github.com/lfnovo/esperanto/issues/82) — design discussion
