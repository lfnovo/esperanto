# Vertex AI

## Overview

Vertex AI is Google Cloud's enterprise AI platform, providing access to Gemini models, embeddings, and text-to-speech capabilities with enterprise-grade security and scalability.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Gemini 1.5 Pro, Gemini 2.0 Flash |
| Embeddings | ✅ | text-embedding-004, text-multilingual-embedding-002 |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ✅ | Multi-speaker support, multiple voices |

**Official Documentation:** https://cloud.google.com/vertex-ai/docs

## Prerequisites

### Account Requirements
- Google Cloud account with billing enabled
- Vertex AI API enabled in your project
- Service account with appropriate permissions

### Getting Started
1. Create a Google Cloud Project at https://console.cloud.google.com
2. Enable the Vertex AI API
3. Create a service account with Vertex AI User role
4. Download the service account key JSON file
5. Note your project ID and preferred location (region)

## Environment Variables

```bash
# Google Application Credentials (required)
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Vertex AI project configuration (required)
VERTEX_PROJECT="your-project-id"
VERTEX_LOCATION="us-east5"  # or us-central1, europe-west1, etc.
```

**Variable Priority:**
1. Direct parameters in config dictionary
2. Environment variables (`GOOGLE_APPLICATION_CREDENTIALS`, `VERTEX_PROJECT`, `VERTEX_LOCATION`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Language model
model = AIFactory.create_language("vertex", "gemini-1.5-pro")

# Embedding model
embedder = AIFactory.create_embedding("vertex", "text-embedding-004")

# Text-to-speech
speaker = AIFactory.create_text_to_speech("vertex", "default")
```

### Direct Instantiation

```python
from esperanto.providers.llm.vertex import VertexLanguageModel
from esperanto.providers.embedding.vertex import VertexEmbeddingModel
from esperanto.providers.text_to_speech.vertex import VertexTextToSpeech

# Language model
llm = VertexLanguageModel(
    model_name="gemini-1.5-pro",
    config={
        "project": "your-project-id",
        "location": "us-east5",
        "credentials_path": "/path/to/service-account-key.json"
    }
)

# Embedding model
embedder = VertexEmbeddingModel(
    model_name="text-embedding-004",
    config={
        "project": "your-project-id",
        "location": "us-east5"
    }
)

# Text-to-speech
tts = VertexTextToSpeech(
    model_name="default",
    config={
        "project": "your-project-id",
        "location": "us-east5"
    }
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
    "vertex",
    "gemini-1.5-pro",
    config={
        "temperature": 0.7,           # Randomness (0.0 - 2.0)
        "max_tokens": 8000,           # Maximum response length
        "top_p": 0.9,                 # Nucleus sampling
        "streaming": True,            # Enable streaming
        "structured": {"type": "json"}, # JSON mode
        "project": "your-project-id",
        "location": "us-east5"
    }
)
```

**Example - Basic Chat:**

```python
from esperanto.factory import AIFactory

# Create model
model = AIFactory.create_language("vertex", "gemini-1.5-pro")

# Chat completion
messages = [
    {"role": "user", "content": "Explain cloud computing in simple terms."}
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
    "vertex",
    "gemini-2.0-flash",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List three cloud services with their descriptions as JSON"
}]

response = model.chat_complete(messages)
# Response will be valid JSON
```

### Embeddings

**Available Models:**

| Model | Dimensions | Best For |
|-------|------------|----------|
| **text-embedding-004** | 768 | Latest, highest quality |
| **text-multilingual-embedding-002** | 768 | Multilingual content |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_embedding(
    "vertex",
    "text-embedding-004",
    config={
        "project": "your-project-id",
        "location": "us-east5",
        "timeout": 60.0
    }
)
```

**Example - Basic Embeddings:**

```python
from esperanto.factory import AIFactory

# Create embedding model
model = AIFactory.create_embedding("vertex", "text-embedding-004")

# Generate embeddings
texts = ["Cloud computing is scalable", "Enterprise AI solutions"]
response = model.embed(texts)

# Access embeddings
for i, embedding_obj in enumerate(response.data):
    print(f"Text {i}: {len(embedding_obj.embedding)} dimensions")
```

**Example - Multilingual Embeddings:**

```python
# Use multilingual model for international content
model = AIFactory.create_embedding("vertex", "text-multilingual-embedding-002")

texts = [
    "Hello, world!",           # English
    "Bonjour le monde!",       # French
    "Hola, mundo!",            # Spanish
    "こんにちは世界!"            # Japanese
]

response = model.embed(texts)
print(f"Generated {len(response.data)} multilingual embeddings")
```

**Example - Batch Processing:**

```python
# Process large batches efficiently
model = AIFactory.create_embedding("vertex", "text-embedding-004")

# Large document corpus
documents = [f"Document {i} content..." for i in range(1000)]

# Process in batches (Vertex handles batching automatically)
response = model.embed(documents)
print(f"Processed {len(response.data)} document embeddings")
```

### Text-to-Speech

**Configuration:**

```python
from esperanto.factory import AIFactory

speaker = AIFactory.create_text_to_speech(
    "vertex",
    "default",
    config={
        "project": "your-project-id",
        "location": "us-east5",
        "timeout": 300.0  # 5 minutes timeout
    }
)
```

**Example - Basic Speech Generation:**

```python
from esperanto.factory import AIFactory

# Create text-to-speech model
model = AIFactory.create_text_to_speech("vertex", "default")

# Generate speech
response = model.generate_speech(
    text="Welcome to Vertex AI Text-to-Speech!",
    voice="en-US-Wavenet-D",
    output_file="greeting.mp3"
)

print(f"Generated {len(response.audio_data)} bytes of audio")
```

**Example - Voice Customization:**

```python
# Professional male voice
response = model.generate_speech(
    text="Welcome to our enterprise platform.",
    voice="en-US-Wavenet-D",  # Professional male
    output_file="professional.mp3"
)

# Natural female voice
response = model.generate_speech(
    text="Thank you for choosing our service.",
    voice="en-US-Wavenet-F",  # Natural female
    output_file="natural.mp3"
)
```

**Example - Multi-Speaker Conversations:**

Vertex AI TTS supports creating dialogues with different voices for each speaker:

```python
# Define conversation with speaker names
conversation_text = """
Alice: Good morning! How is the cloud migration going?
Bob: Great progress! We've moved 80% of our workloads to Vertex AI.
Alice: That's excellent. What about the AI models?
Bob: All deployed on Vertex AI with automatic scaling.
"""

# Configure speakers with different voices
speaker_configs = [
    {"speaker": "Alice", "voice": "en-US-Wavenet-F"},  # Female voice
    {"speaker": "Bob", "voice": "en-US-Wavenet-D"}     # Male voice
]

# Generate multi-speaker audio
response = model.generate_multi_speaker_speech(
    text=conversation_text,
    speaker_configs=speaker_configs,
    output_file="conversation.mp3"
)

print(f"Generated multi-speaker dialogue with {len(speaker_configs)} voices")
```

**Example - Async Multi-Speaker:**

```python
async def create_enterprise_dialogue():
    model = AIFactory.create_text_to_speech("vertex", "default")

    interview_text = """
    Interviewer: Tell us about Vertex AI's capabilities.
    Expert: Vertex AI provides enterprise-grade machine learning at scale.
    Interviewer: What makes it different from other platforms?
    Expert: Integrated security, compliance, and Google Cloud's infrastructure.
    """

    speaker_configs = [
        {"speaker": "Interviewer", "voice": "en-US-Wavenet-A"},
        {"speaker": "Expert", "voice": "en-US-Wavenet-D"}
    ]

    response = await model.agenerate_multi_speaker_speech(
        text=interview_text,
        speaker_configs=speaker_configs,
        output_file="enterprise_interview.mp3"
    )

    return response
```

**Available Voices:**

Vertex AI provides multiple voice options:
- **en-US-Wavenet-A**: Male, professional
- **en-US-Wavenet-B**: Male, warm
- **en-US-Wavenet-C**: Female, professional
- **en-US-Wavenet-D**: Male, authoritative
- **en-US-Wavenet-F**: Female, natural
- **en-US-Neural2-A**: Neural male voice
- **en-US-Neural2-C**: Neural female voice
- Plus many more regional and language-specific voices

## Advanced Features

### Enterprise Security

Vertex AI provides enterprise-grade security features:

```python
# Use service account with minimal permissions
model = AIFactory.create_language(
    "vertex",
    "gemini-1.5-pro",
    config={
        "project": "production-project",
        "location": "us-central1",
        "credentials_path": "/secure/path/service-account.json"
    }
)

# All data stays within your Google Cloud project
response = model.chat_complete(messages)
```

### Regional Deployment

Choose regions for data residency and compliance:

```python
# EU data residency
eu_model = AIFactory.create_language(
    "vertex",
    "gemini-1.5-pro",
    config={
        "project": "eu-project",
        "location": "europe-west1"  # Data stays in EU
    }
)

# US deployment
us_model = AIFactory.create_language(
    "vertex",
    "gemini-2.0-flash",
    config={
        "project": "us-project",
        "location": "us-central1"  # US region
    }
)
```

### Timeout Configuration

Customize request timeouts:

```python
# LLM with custom timeout
model = AIFactory.create_language(
    "vertex",
    "gemini-1.5-pro",
    config={"timeout": 180.0}  # 3 minutes
)

# Embedding with custom timeout
embedder = AIFactory.create_embedding(
    "vertex",
    "text-embedding-004",
    config={"timeout": 120.0}  # 2 minutes
)
```

### LangChain Integration

Convert to LangChain models:

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("vertex", "gemini-1.5-pro")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Region Availability

**Recommended Regions:**
- **us-central1** - Iowa, USA (lowest latency for US)
- **us-east5** - Ohio, USA
- **europe-west1** - Belgium (EU data residency)
- **europe-west4** - Netherlands
- **asia-southeast1** - Singapore
- **asia-northeast1** - Tokyo

**Choosing a Region:**
- Consider data residency requirements
- Choose regions close to your users for lower latency
- Check Vertex AI pricing by region
- Ensure your required models are available in the region

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Could not load credentials
```
**Solution:** Verify `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account JSON file.

**Permission Error:**
```
Error: Permission denied
```
**Solution:** Ensure your service account has the "Vertex AI User" role in the project.

**Project Not Found:**
```
Error: Project not found
```
**Solution:** Verify `VERTEX_PROJECT` is set correctly and the project exists.

**Region Not Available:**
```
Error: Model not available in region
```
**Solution:** Check if your model is available in the specified `VERTEX_LOCATION`. Try a different region.

**API Not Enabled:**
```
Error: Vertex AI API not enabled
```
**Solution:** Enable the Vertex AI API in your Google Cloud Console.

**Quota Exceeded:**
```
Error: Quota exceeded
```
**Solution:** Check your quotas in Google Cloud Console and request increases if needed.

### Best Practices

1. **Use Service Accounts:** Always use service accounts with minimal required permissions.

2. **Regional Selection:** Choose regions based on data residency and latency requirements.

3. **Credentials Security:** Store service account keys securely and use environment variables.

4. **Project Organization:** Use separate projects for development, staging, and production.

5. **Monitor Usage:** Use Google Cloud Console to monitor API usage and costs.

6. **Timeout Configuration:** Set appropriate timeouts based on your use case.

7. **Error Handling:** Implement proper error handling for production applications.

## Cost Optimization

### Cost-Effective Strategies

**Use Appropriate Models:**
```python
# For simple tasks, use Gemini 2.0 Flash (cheaper)
simple_model = AIFactory.create_language("vertex", "gemini-2.0-flash")

# For complex tasks, use Gemini 1.5 Pro
complex_model = AIFactory.create_language("vertex", "gemini-1.5-pro")
```

**Batch Embeddings:**
```python
# Process in larger batches for better efficiency
model = AIFactory.create_embedding("vertex", "text-embedding-004")

# Batch processing is more cost-effective
large_batch = [f"Document {i}" for i in range(1000)]
response = model.embed(large_batch)
```

**Monitor and Budget:**
- Set up billing alerts in Google Cloud Console
- Use Cloud Monitoring to track API usage
- Implement request caching where appropriate
- Consider committed use discounts for high volume

## Use Cases

### Enterprise RAG Systems

```python
# Secure enterprise knowledge base
embedder = AIFactory.create_embedding("vertex", "text-embedding-004")
llm = AIFactory.create_language("vertex", "gemini-1.5-pro")

# Embed company documents (data stays in your Google Cloud)
documents = ["Company policy 1", "Product documentation", "Training materials"]
doc_embeddings = embedder.embed(documents)

# Query with data residency compliance
query = "What is our remote work policy?"
response = llm.chat_complete([{"role": "user", "content": query}])
```

### Multilingual Support

```python
# Global application with multilingual content
embedder = AIFactory.create_embedding("vertex", "text-multilingual-embedding-002")
llm = AIFactory.create_language("vertex", "gemini-1.5-pro")

# Handle multiple languages seamlessly
texts = {
    "en": "Welcome to our platform",
    "fr": "Bienvenue sur notre plateforme",
    "de": "Willkommen auf unserer Plattform",
    "ja": "私たちのプラットフォームへようこそ"
}

embeddings = embedder.embed(list(texts.values()))
```

### Compliance-First Deployment

```python
# GDPR-compliant EU deployment
eu_llm = AIFactory.create_language(
    "vertex",
    "gemini-1.5-pro",
    config={
        "project": "eu-compliant-project",
        "location": "europe-west1"  # Data stays in EU
    }
)

# Process sensitive data with regional constraints
response = eu_llm.chat_complete(messages)
# Data never leaves EU region
```

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [Embeddings Guide](../capabilities/embedding.md)
- [Text-to-Speech Guide](../capabilities/text-to-speech.md)
- [Google Provider](./google.md)
- [Azure Provider](./azure.md)
