# Connection Resource Management
## HTTP Client Management

All Esperanto providers that use HTTP connections (LLM, Embedding, Reranker, STT, TTS) automatically manage HTTP client resources. There are two options for managing these resources: **automatic cleanup** (default) or **explicit management** (recommended for long-running applications).

### Default Behavior (Automatic Cleanup)

By default, HTTP clients are automatically cleaned up when the model object is destroyed:

```python
from esperanto.factory import AIFactory

# Clients are automatically created and managed
model = AIFactory.create_language("openai", "gpt-4")
response = model.chat_complete(messages)

# Clients are automatically closed when model goes out of scope
# No manual cleanup needed for short-lived instances
```
> NOTE: THe default auto-cleanup approach only works for synchronous cases and cannot be used for asynchronous clients.


### Explicit Resource Management

For long-running applications, web servers, or when you want explicit control, use one of these methods:

#### Option 1: Context Manager (Recommended)

Use Python's context manager for automatic cleanup:

**Synchronous:**
```python
from esperanto.factory import AIFactory

# Sync context manager
with AIFactory.create_language("openai", "gpt-4") as model:
    response = model.chat_complete(messages)
    # Client is automatically closed when exiting the context
```

**Asynchronous:**
```python
import asyncio
from esperanto.factory import AIFactory

async def main():
    # Async context manager
    async with AIFactory.create_language("openai", "gpt-4") as model:
        response = await model.achat_complete(messages)
        # Async client is automatically closed when exiting the context

asyncio.run(main())
```

**Benefits:**
- ✅ Automatic cleanup even if exceptions occur
- ✅ Works with both sync and async code
- ✅ Pythonic

#### Option 2: Manual Close

Explicitly close clients when done:

**Synchronous:**
```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("openai", "gpt-4")
try:
    response = model.chat_complete(messages)
finally:
    model.close()  # Close sync client
```

**Asynchronous:**
```python
import asyncio
from esperanto.factory import AIFactory

async def main():
    model = AIFactory.create_language("openai", "gpt-4")
    try:
        response = await model.achat_complete(messages)
    finally:
        await model.aclose()  # Close async client

asyncio.run(main())
```

**Closing Both Clients:**
```python
model = AIFactory.create_language("openai", "gpt-4")

# Use both sync and async methods
response = model.chat_complete(messages)
async_response = await model.achat_complete(messages)

# Close both clients
model.close()      # Closes sync client
await model.aclose()  # Closes async client
```

**Benifits**:
- ✅ Clear, well-defined
- ✅ Support for **connection resource re-use** for huge amount of requests

### When to Use Explicit Management

Use explicit resource management in these scenarios:

**Long-running applications:**
```python
# Web server or daemon
from esperanto.factory import AIFactory

class AIService:
    def __init__(self):
        self.model = AIFactory.create_language("openai", "gpt-4")

    def process(self, messages):
        return self.model.chat_complete(messages)

    def shutdown(self):
        self.model.close()  # Explicitly close when shutting down
```

**Multiple model instances:**
```python
# Create and manage multiple models
models = [
    AIFactory.create_language("openai", "gpt-4"),
    AIFactory.create_embedding("jina", "jina-embeddings-v3"),
    AIFactory.create_reranker("jina", "jina-reranker-v2-base-multilingual")
]

try:
    # Use models...
    pass
finally:
    # Clean up all models
    for model in models:
        model.close()
```

**Note:** Transformers provider (local models) does not use HTTP clients and manages resources in another way.