# Google (GenAI)

## Overview

Google provides access to Gemini models for language tasks, embeddings with native task optimization, and text-to-speech through the Generative AI API.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Gemini 2.0 Flash, Gemini 1.5 Pro |
| Embeddings | ✅ | text-embedding-004 with native task types |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ✅ | 30+ unique voices with personalities |

**Official Documentation:** https://ai.google.dev/docs

## Prerequisites

### Account Requirements
- Google account
- Google AI Studio access (https://makersuite.google.com/app/apikey)
- API key for Generative AI

### Getting API Keys
1. Visit https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy and store the key securely

## Environment Variables

```bash
# Google API key (required) - either variable works
GOOGLE_API_KEY="AIza..."
# or
GEMINI_API_KEY="AIza..."

# Optional: Override base URL (useful for network restrictions or proxies)
# GEMINI_API_BASE_URL="https://generativelanguage.googleapis.com"
```

**Custom Base URL:**
The `GEMINI_API_BASE_URL` environment variable allows you to override the default Gemini API endpoint. This is useful when:
- The default endpoint is not accessible in your network
- You need to use a proxy or alternative routing
- You're testing with a mock or staging environment

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Language model
model = AIFactory.create_language("google", "gemini-2.0-flash")

# Embedding model
embedder = AIFactory.create_embedding("google", "text-embedding-004")

# Text-to-speech
speaker = AIFactory.create_text_to_speech("google", "gemini-2.5-flash-preview-tts")
```

### Direct Instantiation

```python
from esperanto.providers.llm.google import GoogleLanguageModel
from esperanto.providers.embedding.google import GoogleEmbeddingModel
from esperanto.providers.text_to_speech.google import GoogleTextToSpeech

# Language model
llm = GoogleLanguageModel(
    api_key="your-api-key",
    model_name="gemini-2.0-flash"
)

# Embedding model
embedder = GoogleEmbeddingModel(
    api_key="your-api-key",
    model_name="text-embedding-004"
)

# Text-to-speech
tts = GoogleTextToSpeech(
    api_key="your-api-key",
    model_name="gemini-2.5-flash-preview-tts"
)
```

## Capabilities

### Language Models (LLM)

**Available Models:**
- **gemini-2.0-flash** - Latest, fast and capable model
- **gemini-1.5-pro** - Most capable Gemini 1.5 model
- **gemini-1.5-flash** - Fast and efficient

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "google",
    "gemini-2.0-flash",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 2.0)
        "max_tokens": 1000,           # Maximum response length
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode
        "timeout": 60.0               # Request timeout
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create model
model = AIFactory.create_language("google", "gemini-2.0-flash")

# Chat completion
messages = [
    {"role": "user", "content": "Explain machine learning in simple terms."}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Streaming:**

```python
# Synchronous streaming
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Example - JSON Mode:**

```python
model = AIFactory.create_language(
    "google",
    "gemini-2.0-flash",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three cities with their populations as JSON"
}]

response = model.chat_complete(messages)
# Response will be valid JSON
```

**Example - Custom Base URL:**

```python
# Use custom endpoint (for proxies or network restrictions)
import os
os.environ["GEMINI_API_BASE_URL"] = "https://custom-proxy.com"

model = AIFactory.create_language("google", "gemini-2.0-flash")
# Will use custom base URL
```

### Embeddings

**Available Models:**

| Model | Dimensions | Best For |
|-------|------------|----------|
| **text-embedding-004** | 768 | Latest, highest quality |
| **embedding-001** | 768 | General purpose, stable |

**Native Task Types:**

Google has **native API support** for task optimization. The following task types map directly to Gemini API parameters:

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Task type mappings (native Gemini API support)
EmbeddingTaskType.RETRIEVAL_QUERY       → "RETRIEVAL_QUERY"
EmbeddingTaskType.RETRIEVAL_DOCUMENT    → "RETRIEVAL_DOCUMENT"
EmbeddingTaskType.SIMILARITY            → "SEMANTIC_SIMILARITY"
EmbeddingTaskType.CLASSIFICATION        → "CLASSIFICATION"
EmbeddingTaskType.CLUSTERING            → "CLUSTERING"
EmbeddingTaskType.CODE_RETRIEVAL        → "CODE_RETRIEVAL_QUERY"
EmbeddingTaskType.QUESTION_ANSWERING    → "QUESTION_ANSWERING"
EmbeddingTaskType.FACT_VERIFICATION     → "FACT_VERIFICATION"
```

**Configuration:**

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

model = AIFactory.create_embedding(
    "google",
    "text-embedding-004",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,  # Native optimization
        "timeout": 60.0
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
model = AIFactory.create_embedding("google", "text-embedding-004")

# Generate embeddings
texts = ["Hello, world!", "Another text"]
response = model.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
```

**Example - Task-Optimized Embeddings:**

```python
from esperanto.common_types.task_type import EmbeddingTaskType

# Optimize for search queries (native API support)
query_model = AIFactory.create_embedding(
    "google",
    "text-embedding-004",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

# Optimize for document storage (native API support)
document_model = AIFactory.create_embedding(
    "google",
    "text-embedding-004",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

# Generate optimized embeddings
query_embedding = query_model.embed(["search query"])
doc_embeddings = document_model.embed(["document 1", "document 2"])
```

**Example - Fact Verification:**

```python
# Google-specific task type for fact-checking
fact_model = AIFactory.create_embedding(
    "google",
    "text-embedding-004",
    config={"task_type": EmbeddingTaskType.FACT_VERIFICATION}
)

statements = [
    "The Earth orbits the Sun",
    "Water boils at 100°C at sea level"
]

embeddings = fact_model.embed(statements)
```

**Example - Question Answering:**

```python
# Optimize for Q&A tasks
qa_model = AIFactory.create_embedding(
    "google",
    "text-embedding-004",
    config={"task_type": EmbeddingTaskType.QUESTION_ANSWERING}
)

questions = ["What is Python?", "How does photosynthesis work?"]
embeddings = qa_model.embed(questions)
```

### Text-to-Speech

**Available Voices:**

Google TTS provides 30+ unique voices with distinct personalities:

| Voice | Gender | Personality | Example Use Case |
|-------|--------|-------------|------------------|
| **achernar** | Female | Upbeat | Energetic content |
| **charon** | Male | Upbeat | Engaging narration |
| **kore** | Female | Informative | Educational content |
| **puck** | Male | Bright | Lively presentations |
| **sulafat** | Female | Knowledgeable | Expert explanations |
| **umbriel** | Male | Professional | Business content |
| **gacrux** | Female | Clear | Professional narration |
| **despina** | Female | Smooth/Gentle | Calm narration |
| **sadachbia** | Male | Smooth/Gentle | Soothing content |
| **algenib** | Male | Clear | Technical content |
| **enceladus** | Male | Bright/Energetic | Dynamic content |
| **leda** | Female | Engaging | Interactive content |
| **laomedeia** | Female | Informative | Tutorial content |
| **erinome** | Female | Smooth | Relaxing content |

**Configuration:**

```python
from esperanto.factory import AIFactory

speaker = AIFactory.create_text_to_speech(
    "google",
    "gemini-2.5-flash-preview-tts",
    config={
        "timeout": 300.0  # 5 minutes timeout
    }
)
```

**Example - Basic Speech Generation:**

```python
from esperanto.factory import AIFactory

# Create text-to-speech model
model = AIFactory.create_text_to_speech("google", "gemini-2.5-flash-preview-tts")

# Generate speech
response = model.generate_speech(
    text="Hello from Google Text-to-Speech!",
    voice="charon",  # Upbeat male voice
    output_file="greeting.wav"
)

print(f"Generated {len(response.audio_data)} bytes of audio")
```

**Example - Voice Customization:**

```python
# Professional female voice for business content
response = model.generate_speech(
    text="Welcome to our quarterly earnings call.",
    voice="gacrux",  # Clear, professional female
    output_file="business.wav"
)

# Energetic male voice for dynamic content
response = model.generate_speech(
    text="Get ready for an exciting journey!",
    voice="enceladus",  # Bright, energetic male
    output_file="energetic.wav"
)
```

**Example - Multi-Speaker Conversations:**

Google TTS supports creating dialogues with different voices for each speaker:

```python
# Define conversation with speaker names
conversation_text = """
Joe: Hi there! How are you doing today?
Jane: I'm doing great, thanks for asking! How about you?
Joe: I'm wonderful. Did you see the latest AI developments?
Jane: Yes! The multi-speaker TTS technology is really impressive.
"""

# Configure speakers with different voices
speaker_configs = [
    {"speaker": "Joe", "voice": "charon"},    # Male, upbeat voice
    {"speaker": "Jane", "voice": "kore"}      # Female, informative voice
]

# Generate multi-speaker audio
response = model.generate_multi_speaker_speech(
    text=conversation_text,
    speaker_configs=speaker_configs,
    output_file="conversation.wav"
)

print(f"Generated multi-speaker dialogue with {len(speaker_configs)} voices")
```

**Example - Async Multi-Speaker:**

```python
async def create_dialogue():
    model = AIFactory.create_text_to_speech("google", "gemini-2.5-flash-preview-tts")

    interview_text = """
    Interviewer: Welcome to our tech podcast. Today we're discussing AI.
    Expert: Thank you for having me. AI is transforming every industry.
    Interviewer: What's the most exciting development you've seen recently?
    Expert: Multi-modal AI that can understand and generate text, images, and audio.
    """

    speaker_configs = [
        {"speaker": "Interviewer", "voice": "puck"},     # Bright, engaging male
        {"speaker": "Expert", "voice": "sulafat"}        # Knowledgeable female
    ]

    response = await model.agenerate_multi_speaker_speech(
        text=interview_text,
        speaker_configs=speaker_configs,
        output_file="ai_interview.wav"
    )

    return response
```

## Advanced Features

### Native Task Optimization
Google embeddings have native API support for task types - no emulation needed:

```python
# All these task types are sent directly to Google's API
task_types = [
    EmbeddingTaskType.RETRIEVAL_QUERY,
    EmbeddingTaskType.RETRIEVAL_DOCUMENT,
    EmbeddingTaskType.SIMILARITY,
    EmbeddingTaskType.CLASSIFICATION,
    EmbeddingTaskType.CLUSTERING,
    EmbeddingTaskType.CODE_RETRIEVAL,
    EmbeddingTaskType.QUESTION_ANSWERING,
    EmbeddingTaskType.FACT_VERIFICATION
]

for task_type in task_types:
    model = AIFactory.create_embedding(
        "google",
        "text-embedding-004",
        config={"task_type": task_type}
    )
```

### Custom Base URL
Override the default API endpoint:

```python
import os

# Set custom base URL
os.environ["GEMINI_API_BASE_URL"] = "https://custom-proxy.com"

# All Google providers will use this URL
model = AIFactory.create_language("google", "gemini-2.0-flash")
embedder = AIFactory.create_embedding("google", "text-embedding-004")
```

### Timeout Configuration
Customize request timeouts:

```python
# LLM with custom timeout
model = AIFactory.create_language(
    "google",
    "gemini-2.0-flash",
    config={"timeout": 120.0}  # 2 minutes
)

# Embedding with custom timeout
embedder = AIFactory.create_embedding(
    "google",
    "text-embedding-004",
    config={"timeout": 90.0}  # 1.5 minutes
)
```

### LangChain Integration
Convert to LangChain models:

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("google", "gemini-2.0-flash")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key is correct and active in Google AI Studio.

**Rate Limit Error:**
```
Error: Quota exceeded
```
**Solution:** Check your quota in Google AI Studio and request an increase if needed.

**Network Error with Base URL:**
```
Error: Failed to connect to endpoint
```
**Solution:** Check `GEMINI_API_BASE_URL` is correctly set or remove it to use default.

**Timeout Error:**
```
Error: Request timed out
```
**Solution:** Increase the timeout configuration.

### Best Practices

1. **Use Native Task Types:** Take advantage of Google's native task optimization for embeddings.

2. **Choose Right Voice:** Select voices that match your content's tone and audience.

3. **Multi-Speaker Content:** Use multi-speaker feature for dialogues and conversations.

4. **API Key Security:** Always use environment variables for API keys in production.

5. **Custom Base URL:** Only set `GEMINI_API_BASE_URL` when you need to use a proxy or alternative routing.

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [Embeddings Guide](../capabilities/embedding.md)
- [Text-to-Speech Guide](../capabilities/text-to-speech.md)
- [OpenAI Provider](./openai.md)
- [Anthropic Provider](./anthropic.md)
- [Azure Provider](./azure.md)
