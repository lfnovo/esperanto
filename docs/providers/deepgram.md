# Deepgram

## Overview

Deepgram provides high-quality text-to-speech via its Aura model family. Aura-2 supports English, Spanish, French, German, Italian, Japanese, and Dutch voices; legacy Aura-1 voices are also available for English.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ❌ | Not available |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ✅ | Aura-2 and Aura-1 voice families |

**Official Documentation:** https://developers.deepgram.com/docs/tts-overview

## Prerequisites

### Account Requirements
- Deepgram account (sign up at https://deepgram.com)
- API key with TTS access

### Getting API Keys
1. Visit https://console.deepgram.com
2. Navigate to "API Keys"
3. Create a new key or copy an existing one

## Environment Variables

```bash
# Deepgram API key (required)
DEEPGRAM_API_KEY="your-api-key"
```

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

model = AIFactory.create_text_to_speech("deepgram", "aura-2-thalia-en")
response = model.generate_speech(
    text="Hello from Deepgram!",
    voice="aura-2-thalia-en",
)
with open("output.mp3", "wb") as f:
    f.write(response.audio_data)
```

### Direct Instantiation

```python
from esperanto.providers.tts.deepgram import DeepgramTextToSpeechModel

model = DeepgramTextToSpeechModel(
    api_key="your-api-key",
    model_name="aura-2-thalia-en",
)
```

## Async Usage

```python
import asyncio
from esperanto.factory import AIFactory

async def main():
    model = AIFactory.create_text_to_speech("deepgram", "aura-2-thalia-en")
    response = await model.agenerate_speech(
        text="Hello from Deepgram async!",
        voice="aura-2-thalia-en",
    )
    with open("output.mp3", "wb") as f:
        f.write(response.audio_data)

asyncio.run(main())
```

## Parameters

All extra keyword arguments to `generate_speech` / `agenerate_speech` are forwarded as Deepgram query parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `encoding` | str | Output encoding: `mp3` (default), `wav`, `flac`, `ogg`, `aac`, `opus`, `linear16` |
| `container` | str | File format container: `mp3`, `wav`, `ogg`, `none` |
| `sample_rate` | int | Sample rate in Hz: `8000`, `16000`, `22050`, `24000`, `48000` |
| `bit_rate` | int | Bitrate in bits/sec (e.g. `128000`) |
| `speed` | float | Speaking rate multiplier, range 0.7–1.5 (default 1.0) |

Example with extra parameters:

```python
response = model.generate_speech(
    text="Slow and clear.",
    voice="aura-2-thalia-en",
    encoding="wav",
    sample_rate=16000,
    speed=0.85,
)
```

## Input Limit

Deepgram TTS accepts up to **2000 characters** per request. Split longer texts into chunks if needed.

## Available Voices

Voices are available as the `DEEPGRAM_VOICES` constant in `esperanto.providers.tts.deepgram`, and also via the `available_voices` property on the model instance.

### Aura-2 English (41 voices)

| Voice ID | Name | Gender |
|----------|------|--------|
| aura-2-thalia-en | Thalia | Female |
| aura-2-andromeda-en | Andromeda | Female |
| aura-2-helena-en | Helena | Female |
| aura-2-apollo-en | Apollo | Male |
| aura-2-arcas-en | Arcas | Male |
| aura-2-aries-en | Aries | Male |
| aura-2-amalthea-en | Amalthea | Female |
| aura-2-asteria-en | Asteria | Female |
| aura-2-athena-en | Athena | Female |
| aura-2-atlas-en | Atlas | Male |
| aura-2-aurora-en | Aurora | Female |
| aura-2-callista-en | Callista | Female |
| aura-2-cora-en | Cora | Female |
| aura-2-cordelia-en | Cordelia | Female |
| aura-2-delia-en | Delia | Female |
| aura-2-draco-en | Draco | Male |
| aura-2-electra-en | Electra | Female |
| aura-2-harmonia-en | Harmonia | Female |
| aura-2-hera-en | Hera | Female |
| aura-2-hermes-en | Hermes | Male |
| aura-2-hyperion-en | Hyperion | Male |
| aura-2-iris-en | Iris | Female |
| aura-2-janus-en | Janus | Male |
| aura-2-juno-en | Juno | Female |
| aura-2-jupiter-en | Jupiter | Male |
| aura-2-luna-en | Luna | Female |
| aura-2-mars-en | Mars | Male |
| aura-2-minerva-en | Minerva | Female |
| aura-2-neptune-en | Neptune | Male |
| aura-2-odysseus-en | Odysseus | Male |
| aura-2-ophelia-en | Ophelia | Female |
| aura-2-orion-en | Orion | Male |
| aura-2-orpheus-en | Orpheus | Male |
| aura-2-pandora-en | Pandora | Female |
| aura-2-phoebe-en | Phoebe | Female |
| aura-2-pluto-en | Pluto | Male |
| aura-2-saturn-en | Saturn | Male |
| aura-2-selene-en | Selene | Female |
| aura-2-theia-en | Theia | Female |
| aura-2-vesta-en | Vesta | Female |
| aura-2-zeus-en | Zeus | Male |

### Aura-2 Other Languages

- **Spanish** (`-es`): celeste, estrella, nestor, sirio, carina, alvaro, diana, aquila, selena, javier, agustina, antonia, gloria, luciano, olivia, silvia, valerio
- **French** (`-fr`): agathe, hector
- **German** (`-de`): julius, viktoria, elara, aurelia, lara, fabian, kara
- **Italian** (`-it`): livia, dionisio, melia, elio, flavio, maia, cinzia, cesare, perseo, demetra
- **Japanese** (`-ja`): fujin, izanami, uzume, ebisu, ama
- **Dutch** (`-nl`): rhea, sander, beatrix, daphne, cornelia, hestia, lars, roman, leda

### Aura-1 English (Legacy)

`aura-asteria-en`, `aura-luna-en`, `aura-stella-en`, `aura-athena-en`, `aura-hera-en`, `aura-orion-en`, `aura-arcas-en`, `aura-perseus-en`, `aura-angus-en`, `aura-orpheus-en`, `aura-helios-en`, `aura-zeus-en`

## See Also

- [Text-to-Speech Guide](../capabilities/text-to-speech.md)
- [OpenAI Provider](./openai.md)
- [ElevenLabs Provider](./elevenlabs.md)
