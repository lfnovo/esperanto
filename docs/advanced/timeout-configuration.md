# Timeout Configuration

## Overview

Esperanto provides flexible timeout configuration across all provider types with intelligent defaults and multiple configuration methods. This allows you to optimize for different use cases, from real-time applications requiring quick responses to batch processing with longer-running operations.

## Default Timeouts

Different provider types have optimized default timeouts based on typical operation duration:

| Provider Type | Default Timeout | Typical Use Case |
|--------------|-----------------|------------------|
| **Language Models** | 60 seconds | Chat completions, text generation |
| **Embedding** | 60 seconds | Text embeddings, semantic search |
| **Reranking** | 60 seconds | Search result reranking |
| **Speech-to-Text** | 300 seconds (5 min) | Audio transcription |
| **Text-to-Speech** | 300 seconds (5 min) | Audio generation |

## Configuration Methods

Configure timeouts using three methods with clear priority hierarchy:

### 1. Config Dictionary (Highest Priority)

The most explicit and recommended method - set timeout in the config dictionary:

```python
from esperanto.factory import AIFactory

# LLM with custom timeout
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"timeout": 120.0}  # 2 minutes
)

# Embedding with custom timeout
embedder = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-small",
    config={"timeout": 90.0}  # 1.5 minutes
)

# Speech-to-Text with longer timeout for large files
transcriber = AIFactory.create_speech_to_text(
    "openai",
    config={"timeout": 600.0}  # 10 minutes
)
```

### 2. Direct Parameters (STT/TTS Only)

For Speech-to-Text and Text-to-Speech providers, you can pass timeout as a direct parameter:

```python
# Text-to-Speech with direct timeout parameter
speaker = AIFactory.create_text_to_speech(
    "elevenlabs",
    timeout=180.0  # 3 minutes
)

# Speech-to-Text with direct timeout parameter
transcriber = AIFactory.create_speech_to_text(
    "openai",
    timeout=450.0  # 7.5 minutes
)
```

**Note**: This method is only available for STT and TTS providers. For consistency, using the config dictionary is recommended for all provider types.

### 3. Environment Variables

Set global defaults for all instances of a provider type:

```bash
# Set environment variables
export ESPERANTO_LLM_TIMEOUT=90          # 90 seconds for all LLM providers
export ESPERANTO_EMBEDDING_TIMEOUT=120   # 2 minutes for all embedding providers
export ESPERANTO_RERANKER_TIMEOUT=75     # 75 seconds for all reranker providers
export ESPERANTO_STT_TIMEOUT=600         # 10 minutes for all STT providers
export ESPERANTO_TTS_TIMEOUT=400         # 6.5 minutes for all TTS providers
```

```python
# These will use environment variable defaults
model = AIFactory.create_language("openai", "gpt-4")  # Uses ESPERANTO_LLM_TIMEOUT
embedder = AIFactory.create_embedding("voyage", "voyage-2")  # Uses ESPERANTO_EMBEDDING_TIMEOUT
```

## Priority Order

Configuration resolves in this priority order (highest to lowest):

1. **Config parameter** - Explicit timeout in config dict
2. **Environment variable** - Global default for provider type
3. **Provider type default** - Built-in default (60s or 300s)

```python
# Example: Final timeout will be 150 seconds (config overrides env var)
# Even if ESPERANTO_LLM_TIMEOUT=90 is set
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"timeout": 150.0}  # This takes precedence
)
```

## Validation

All timeout values are validated with clear error messages to prevent common mistakes:

### Valid Range

- **Minimum**: 1 second
- **Maximum**: 3600 seconds (1 hour)

### Type Checking

Timeouts must be numeric (int or float):

```python
# ✅ Valid timeouts
AIFactory.create_language("openai", "gpt-4", config={"timeout": 30})      # int
AIFactory.create_language("openai", "gpt-4", config={"timeout": 30.5})    # float
AIFactory.create_language("openai", "gpt-4", config={"timeout": 120.0})   # float

# ❌ Invalid timeouts - will raise ValueError
AIFactory.create_language("openai", "gpt-4", config={"timeout": "30"})    # string
AIFactory.create_language("openai", "gpt-4", config={"timeout": -1})      # negative
AIFactory.create_language("openai", "gpt-4", config={"timeout": 0})       # zero
AIFactory.create_language("openai", "gpt-4", config={"timeout": 4000})    # too large
```

### Error Messages

```python
# TypeError for non-numeric values
try:
    model = AIFactory.create_language("openai", "gpt-4", config={"timeout": "invalid"})
except ValueError as e:
    print(e)  # "Timeout must be a number (int or float), got str"

# ValueError for out-of-range values
try:
    model = AIFactory.create_language("openai", "gpt-4", config={"timeout": -1})
except ValueError as e:
    print(e)  # "Timeout must be between 1 and 3600 seconds, got -1"
```

## Production Use Cases

### Real-time Applications

Short timeouts for interactive applications:

```python
# Chatbot with quick response requirement
chatbot = AIFactory.create_language(
    "groq",
    "mixtral-8x7b-32768",  # Fast inference
    config={"timeout": 30.0}  # 30 seconds max
)

# Real-time embedding for instant search
search_embedder = AIFactory.create_embedding(
    "voyage",
    "voyage-2",
    config={"timeout": 20.0}  # 20 seconds
)
```

### Batch Processing

Longer timeouts for bulk operations:

```python
# Batch embedding of large datasets
batch_embedder = AIFactory.create_embedding(
    "openai",
    "text-embedding-3-large",
    config={"timeout": 300.0}  # 5 minutes for large batches
)

# Process thousands of documents
documents = load_documents()  # Large dataset
embeddings = batch_embedder.embed(documents)
```

### Audio Processing

Extended timeouts for long audio files:

```python
# Transcribe hour-long meetings
transcriber = AIFactory.create_speech_to_text(
    "openai",
    config={"timeout": 900.0}  # 15 minutes
)

# Transcribe large audio file
with open("meeting.mp3", "rb") as audio_file:
    transcript = transcriber.transcribe(audio_file)

# Generate long-form audio content
speaker = AIFactory.create_text_to_speech(
    "elevenlabs",
    timeout=600.0  # 10 minutes for long narrations
)
```

### Background Jobs

Maximum timeout for async background processing:

```python
# Long-running document analysis
analyzer = AIFactory.create_language(
    "anthropic",
    "claude-3-5-sonnet-20241022",
    config={"timeout": 3600.0}  # 1 hour max
)

# Process in background
async def process_document(doc):
    messages = [{"role": "user", "content": f"Analyze this document: {doc}"}]
    result = await analyzer.achat_complete(messages)
    return result
```

### Microservices Architecture

Different timeouts for different service tiers:

```python
# API Gateway Configuration
class AIServiceConfig:
    """Centralized timeout configuration"""

    # Quick operations
    QUICK_TIMEOUT = 15.0

    # Standard operations
    STANDARD_TIMEOUT = 60.0

    # Long operations
    LONG_TIMEOUT = 300.0

    @staticmethod
    def create_quick_llm(provider, model):
        return AIFactory.create_language(
            provider, model,
            config={"timeout": AIServiceConfig.QUICK_TIMEOUT}
        )

    @staticmethod
    def create_standard_embedder(provider, model):
        return AIFactory.create_embedding(
            provider, model,
            config={"timeout": AIServiceConfig.STANDARD_TIMEOUT}
        )

    @staticmethod
    def create_batch_transcriber(provider):
        return AIFactory.create_speech_to_text(
            provider,
            config={"timeout": AIServiceConfig.LONG_TIMEOUT}
        )

# Usage in services
quick_chat = AIServiceConfig.create_quick_llm("groq", "llama3-8b-8192")
standard_embedder = AIServiceConfig.create_standard_embedder("openai", "text-embedding-3-small")
batch_transcriber = AIServiceConfig.create_batch_transcriber("openai")
```

### Load Testing

Adjust timeouts for different load scenarios:

```python
# Normal load
normal_model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 60.0}
)

# High load - shorter timeout to fail fast
high_load_model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 30.0}  # Fail faster under load
)

# Low priority background tasks - longer timeout
background_model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 180.0}  # More patient for background work
)
```

## Environment-Specific Configuration

### Development

More generous timeouts for debugging:

```bash
# .env.development
ESPERANTO_LLM_TIMEOUT=300
ESPERANTO_EMBEDDING_TIMEOUT=180
ESPERANTO_STT_TIMEOUT=900
```

### Production

Tighter timeouts for reliability:

```bash
# .env.production
ESPERANTO_LLM_TIMEOUT=60
ESPERANTO_EMBEDDING_TIMEOUT=45
ESPERANTO_STT_TIMEOUT=300
```

### Testing

Very short timeouts to catch issues early:

```bash
# .env.test
ESPERANTO_LLM_TIMEOUT=10
ESPERANTO_EMBEDDING_TIMEOUT=10
ESPERANTO_STT_TIMEOUT=30
```

### Loading Environment-Specific Config

```python
import os
from dotenv import load_dotenv

# Load environment-specific config
env = os.getenv("ENVIRONMENT", "development")
load_dotenv(f".env.{env}")

# Models automatically use environment timeouts
model = AIFactory.create_language("openai", "gpt-4")
# Uses timeout from ESPERANTO_LLM_TIMEOUT in loaded env file
```

## Timeout Handling Strategies

### Retry with Exponential Backoff

```python
import time

def create_model_with_retry(provider, model_name, max_retries=3):
    """Create model with retry logic for timeout errors"""

    for attempt in range(max_retries):
        try:
            # Increase timeout with each retry
            timeout = 60.0 * (2 ** attempt)  # 60s, 120s, 240s

            model = AIFactory.create_language(
                provider, model_name,
                config={"timeout": min(timeout, 3600.0)}  # Cap at 1 hour
            )

            # Test the model
            test_messages = [{"role": "user", "content": "test"}]
            model.chat_complete(test_messages)

            return model

        except TimeoutError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

# Usage
model = create_model_with_retry("openai", "gpt-4")
```

### Fallback to Faster Model

```python
def create_model_with_fallback(primary_config, fallback_config):
    """Try primary model, fall back to faster model on timeout"""

    try:
        # Try primary model with shorter timeout
        model = AIFactory.create_language(
            **primary_config,
            config={"timeout": 30.0}
        )
        return model, "primary"

    except TimeoutError:
        # Fall back to faster model
        print("Primary model timeout, using fallback...")
        model = AIFactory.create_language(
            **fallback_config,
            config={"timeout": 60.0}
        )
        return model, "fallback"

# Usage
primary = {"provider": "anthropic", "model_name": "claude-3-5-sonnet-20241022"}
fallback = {"provider": "groq", "model_name": "mixtral-8x7b-32768"}

model, used = create_model_with_fallback(primary, fallback)
print(f"Using {used} model")
```

### Circuit Breaker Pattern

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    """Circuit breaker for timeout protection"""

    def __init__(self, failure_threshold=5, timeout_duration=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_duration):
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except TimeoutError:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = "closed"

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

# Usage
breaker = CircuitBreaker(failure_threshold=3, timeout_duration=60)

model = AIFactory.create_language("openai", "gpt-4", config={"timeout": 30.0})

try:
    response = breaker.call(
        model.chat_complete,
        [{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    print(f"Circuit breaker prevented call: {e}")
```

## Monitoring and Logging

### Track Timeout Performance

```python
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimedModel:
    """Wrapper to track model performance"""

    def __init__(self, model):
        self.model = model
        self.call_times = []

    def chat_complete(self, messages):
        start_time = time.time()
        try:
            response = self.model.chat_complete(messages)
            elapsed = time.time() - start_time

            self.call_times.append(elapsed)
            logger.info(f"Call completed in {elapsed:.2f}s")

            # Warn if approaching timeout
            if hasattr(self.model, 'timeout') and elapsed > self.model.timeout * 0.8:
                logger.warning(f"Call took {elapsed:.2f}s, approaching timeout of {self.model.timeout}s")

            return response

        except TimeoutError as e:
            elapsed = time.time() - start_time
            logger.error(f"Timeout after {elapsed:.2f}s")
            raise

    def get_stats(self):
        if not self.call_times:
            return None

        return {
            "count": len(self.call_times),
            "avg": sum(self.call_times) / len(self.call_times),
            "min": min(self.call_times),
            "max": max(self.call_times)
        }

# Usage
base_model = AIFactory.create_language("openai", "gpt-4", config={"timeout": 60.0})
timed_model = TimedModel(base_model)

# Make calls
timed_model.chat_complete([{"role": "user", "content": "Hello"}])

# Get statistics
stats = timed_model.get_stats()
print(f"Average response time: {stats['avg']:.2f}s")
```

## Best Practices

### 1. Set Appropriate Defaults

```python
# ✅ Good - Reasonable defaults for use case
chat_model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 60.0}  # Standard chat timeout
)

# ❌ Bad - Unnecessarily long timeout
chat_model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 3600.0}  # 1 hour for simple chat?
)
```

### 2. Use Environment Variables for Global Defaults

```bash
# Set sensible defaults in environment
export ESPERANTO_LLM_TIMEOUT=60
export ESPERANTO_EMBEDDING_TIMEOUT=45
```

```python
# Override only when needed
standard_model = AIFactory.create_language("openai", "gpt-4")  # Uses env default

long_running_model = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 300.0}  # Override for specific use case
)
```

### 3. Match Timeout to Provider Speed

```python
# Fast providers - shorter timeout
groq_model = AIFactory.create_language(
    "groq", "mixtral-8x7b-32768",
    config={"timeout": 30.0}  # Groq is very fast
)

# Slower providers - longer timeout
anthropic_model = AIFactory.create_language(
    "anthropic", "claude-3-5-sonnet-20241022",
    config={"timeout": 120.0}  # Claude can be slower
)
```

### 4. Consider Content Length

```python
# Short prompts - short timeout
quick_chat = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 30.0}
)

# Long documents or complex analysis - longer timeout
document_analyzer = AIFactory.create_language(
    "openai", "gpt-4",
    config={"timeout": 180.0}
)
```

### 5. Document Your Timeout Strategy

```python
class ModelFactory:
    """
    Centralized model creation with documented timeout strategy:

    - Quick operations (chat, simple queries): 30s
    - Standard operations (analysis, generation): 60s
    - Long operations (batch, complex tasks): 300s
    - Audio operations: 600s
    """

    TIMEOUT_QUICK = 30.0
    TIMEOUT_STANDARD = 60.0
    TIMEOUT_LONG = 300.0
    TIMEOUT_AUDIO = 600.0

    @staticmethod
    def create_chat_model():
        return AIFactory.create_language(
            "openai", "gpt-4",
            config={"timeout": ModelFactory.TIMEOUT_QUICK}
        )

    @staticmethod
    def create_analysis_model():
        return AIFactory.create_language(
            "anthropic", "claude-3-5-sonnet-20241022",
            config={"timeout": ModelFactory.TIMEOUT_STANDARD}
        )
```

## See Also

- [Language Model Capabilities](/docs/capabilities/language.md) - LLM features and usage
- [Embedding Capabilities](/docs/capabilities/embeddings.md) - Embedding features
- [Speech-to-Text Capabilities](/docs/capabilities/speech-to-text.md) - Transcription features
- [Text-to-Speech Capabilities](/docs/capabilities/text-to-speech.md) - Audio generation features
- [OpenAI Provider](/docs/providers/openai.md) - Provider-specific details
