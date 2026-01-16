# Common Types

Shared type definitions and response models used across all provider types.

## Files

- **`model.py`**: `Model` dataclass representing AI model metadata (id, owner, context_window)
- **`response.py`**: Chat completion response types (`ChatCompletion`, `ChatCompletionChunk`, `Message`, `Choice`, `Usage`, etc.)
- **`task_type.py`**: `EmbeddingTaskType` enum for task-aware embeddings
- **`stt.py`**: `TranscriptionResponse` for speech-to-text results
- **`tts.py`**: `AudioResponse` and `Voice` for text-to-speech
- **`reranker.py`**: `RerankResponse` and `RerankResult` for document reranking

## Patterns

### Pydantic Models

All response types use Pydantic BaseModel for:

- **Validation**: Automatic type checking and conversion
- **Serialization**: `model_dump()` for dict conversion
- **Immutability**: Most use `frozen=True` config
- **Dict-like access**: Some implement `__getitem__` for backward compatibility

Example:

```python
from pydantic import BaseModel, Field

class Message(BaseModel):
    content: Optional[str] = Field(default=None, description="The content of the message")
    role: Optional[str] = Field(default=None, description="The role of the message sender")
```

### Response Standardization

All providers convert their API responses to Esperanto's common types:

- **LLM**: Returns `ChatCompletion` (non-streaming) or yields `ChatCompletionChunk` (streaming)
- **Embedding**: Returns `List[List[float]]` (not a custom type)
- **Reranker**: Returns `RerankResponse` with list of `RerankResult`
- **STT**: Returns `TranscriptionResponse`
- **TTS**: Returns `AudioResponse`

### Message Structure

Chat messages follow OpenAI-style format (response.py:31):

```python
Message(
    content="Hello, world!",
    role="user",  # or "assistant", "system"
    function_call=None,  # Optional function call
    tool_calls=None,     # Optional tool calls
)
```

### Usage Tracking

Token usage is standardized in `Usage` class (response.py:19):

```python
Usage(
    prompt_tokens=10,
    completion_tokens=20,
    total_tokens=30
)
```

All counts must be >= 0 (enforced by Pydantic).

### Streaming vs Non-Streaming

- **Non-streaming**: `ChatCompletion` with `choices` list containing full `Message`
- **Streaming**: `ChatCompletionChunk` with `choices` containing `DeltaMessage` (partial content)

Providers yield chunks for streaming:

```python
def chat_complete(self, messages, stream=True):
    if stream:
        for chunk in api_stream:
            yield ChatCompletionChunk(...)
    else:
        return ChatCompletion(...)
```

### Task Type Enum

`EmbeddingTaskType` (task_type.py:7) defines universal task types:

- **Retrieval**: `RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`
- **Similarity**: `SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`
- **Code**: `CODE_RETRIEVAL`
- **Q&A**: `QUESTION_ANSWERING`, `FACT_VERIFICATION`
- **Default**: `DEFAULT` (no optimization)

Use for task-aware embeddings:

```python
from esperanto.common_types import EmbeddingTaskType

config = {"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
model = AIFactory.create_embedding("jina", "jina-embeddings-v2-base-en", config=config)
```

### Audio Response

TTS providers return `AudioResponse` (tts.py):

- `audio_data`: bytes (raw audio content)
- `format`: str (e.g., "mp3", "wav")
- `voice_used`: str (voice ID used)
- `model_used`: str (model name used)
- `output_file`: Optional[str] (path if saved)
- `duration`: Optional[float] (duration in seconds)

STT providers return `TranscriptionResponse` (stt.py):

- `text`: str (transcribed text)
- `language`: Optional[str] (detected/specified language)
- `duration`: Optional[float] (audio duration)
- `segments`: Optional[List] (timestamped segments)
- `words`: Optional[List] (word-level timestamps)

### Reranker Response

Reranker providers return `RerankResponse` (reranker.py):

- `model`: str (model used)
- `results`: List[RerankResult] (ranked results)
- `usage`: Optional[Dict] (token usage if available)

Each `RerankResult` contains:

- `index`: int (original index in input documents)
- `document`: str (the document text)
- `relevance_score`: float (relevance score, typically 0-1)

## Integration

- Imported by all provider implementations
- Used for type hints in provider methods
- Ensures consistency across different providers
- Enables seamless provider switching

## Gotchas

- **Frozen models**: Most models use `frozen=True` - create new instances instead of modifying
- **Optional fields**: Many fields are Optional - always check for None before use
- **Dict conversion**: Use `model_dump()` not `dict()` for Pydantic v2
- **Backward compatibility**: `Message` implements `__getitem__` for dict-like access - avoid in new code
- **Enum string conversion**: `EmbeddingTaskType` has custom `__str__()` returning `.value` not `.name`
- **Validation errors**: Pydantic raises ValidationError for invalid data - catch and handle
- **Model validators**: `Message` and `Choice` have custom validators - be aware when constructing
- **Choice vs StreamChoice**: Different types for non-streaming vs streaming (both in response.py)
- **Content can be None**: Message.content is Optional - providers may return None for tool calls
- **Function calls vs tool calls**: Both exist for backward compatibility - use tool_calls for new code

## When Adding New Response Types

1. Create new file or add to existing file in this directory
2. Use Pydantic BaseModel for validation
3. Add clear Field descriptions
4. Make fields Optional if they might not be present
5. Add to `__init__.py` exports
6. Consider frozen=True for immutability
7. Add custom validators if needed
8. Document in this CLAUDE.md file

## Common Type Usage Examples

### Creating a ChatCompletion

```python
from esperanto.common_types import ChatCompletion, Choice, Message, Usage

completion = ChatCompletion(
    id="chatcmpl-123",
    choices=[
        Choice(
            index=0,
            message=Message(content="Hello!", role="assistant"),
            finish_reason="stop"
        )
    ],
    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    model="gpt-4",
    created=1234567890
)
```

### Creating a StreamChunk

```python
from esperanto.common_types import ChatCompletionChunk, StreamChoice, DeltaMessage

chunk = ChatCompletionChunk(
    id="chatcmpl-123",
    choices=[
        StreamChoice(
            index=0,
            delta=DeltaMessage(content="Hello", role="assistant"),
            finish_reason=None
        )
    ],
    model="gpt-4",
    created=1234567890
)
```

### Using EmbeddingTaskType

```python
from esperanto.common_types import EmbeddingTaskType

# String value
task_type = EmbeddingTaskType.RETRIEVAL_QUERY
print(task_type.value)  # "retrieval.query"
print(str(task_type))   # "retrieval.query"

# Enum name
print(task_type.name)   # "RETRIEVAL_QUERY"
```

### Accessing Message Fields

```python
from esperanto.common_types import Message

msg = Message(content="Hello", role="user")

# Pydantic way (preferred)
print(msg.content)  # "Hello"

# Dict-like way (backward compatibility)
print(msg["content"])  # "Hello"

# Convert to dict
msg_dict = msg.model_dump()  # {"content": "Hello", "role": "user", ...}
```
