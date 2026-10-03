# LangChain Integration

## Overview

Esperanto converts any language model provider into a LangChain chat model with `.to_langchain()`. You configure the model once through Esperanto's unified interface and then use it anywhere LangChain expects a chat model: LCEL chains, agents, retrieval pipelines and tools.

The examples on this page target LangChain 1.x (`langchain-core` 1.x).

## Prerequisites

Install `langchain-core` plus the LangChain integration package for each provider you convert:

```bash
pip install langchain-core langchain-openai   # OpenAI, Azure, OpenAI-compatible, OpenRouter, xAI, ...
pip install langchain-anthropic               # Anthropic
pip install langchain-google-genai            # Google (Gemini)
pip install langchain-groq                    # Groq
pip install langchain-ollama                  # Ollama
pip install langchain-mistralai               # Mistral
pip install langchain                         # only for agents (create_agent)
```

`.to_langchain()` raises an `ImportError` naming the missing package when one is needed.

## Quick Start

Convert any Esperanto language model to LangChain format:

```python
from esperanto import AIFactory

# Create an Esperanto model (reads OPENAI_API_KEY from the environment)
model = AIFactory.create_language("openai", "gpt-4o-mini")

# Convert to a LangChain chat model
langchain_model = model.to_langchain()

# Use it like any LangChain chat model
response = langchain_model.invoke("Hello! How are you?")
print(response.content)
```

Pass an API key explicitly through `config`:

```python
model = AIFactory.create_language(
    "openai", "gpt-4o-mini", config={"api_key": "your-api-key"}
)
```

## Supported Providers

The `.to_langchain()` method works with all language model providers in Esperanto. The model, temperature, max tokens and base URL you set in Esperanto carry over to the LangChain model, and so does structured output (except on Cohere). Timeouts carry over for OpenAI, Azure, Groq, Ollama, Perplexity and OpenAI-compatible providers; for the others, set the timeout on the LangChain model.

### OpenAI

```python
from esperanto.providers.llm.openai import OpenAILanguageModel

model = OpenAILanguageModel(model_name="gpt-4o-mini")
langchain_model = model.to_langchain()
```

### Anthropic (Claude)

```python
from esperanto import AIFactory

model = AIFactory.create_language("anthropic", "claude-sonnet-5")
langchain_model = model.to_langchain()
```

### Google (Gemini)

```python
from esperanto import AIFactory

model = AIFactory.create_language("google", "gemini-2.5-flash")
langchain_model = model.to_langchain()
```

### Groq

```python
from esperanto import AIFactory

model = AIFactory.create_language("groq", "openai/gpt-oss-20b")
langchain_model = model.to_langchain()
```

### OpenAI-Compatible Endpoints

```python
from esperanto import AIFactory

# Works with LM Studio, vLLM, LocalAI, etc.
model = AIFactory.create_language(
    "openai-compatible",
    "local-model-name",
    config={
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed-for-local"
    }
)

langchain_model = model.to_langchain()
```

### Ollama

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "ollama",
    "llama3.2",
    config={"base_url": "http://localhost:11434"}
)

langchain_model = model.to_langchain()
```

## Use Cases

### Prompt Templates and Chains (LCEL)

Compose a prompt, the model and an output parser with the `|` operator:

```python
from esperanto import AIFactory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

langchain_model = AIFactory.create_language("anthropic", "claude-sonnet-5").to_langchain()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You translate {input_language} to {output_language}."),
    ("human", "{text}"),
])

chain = prompt | langchain_model | StrOutputParser()

result = chain.invoke({
    "input_language": "English",
    "output_language": "Spanish",
    "text": "Hello, how are you?",
})
print(result)  # "Hola, ¿cómo estás?"
```

### Conversation with Memory

Keep the conversation as a list of messages and send the whole history on each turn:

```python
from esperanto import AIFactory
from langchain_core.messages import HumanMessage, SystemMessage

langchain_model = AIFactory.create_language("openai", "gpt-4o-mini").to_langchain()

history = [SystemMessage("You are a helpful assistant.")]

def chat(text: str) -> str:
    history.append(HumanMessage(text))
    reply = langchain_model.invoke(history)
    history.append(reply)
    return reply.content

print(chat("My name is Alice"))
print(chat("What's my name?"))  # "Your name is Alice"
```

For persistent, multi-session memory, use an agent built with `create_agent` and a LangGraph checkpointer (see the LangChain docs on short-term memory).

### Sequential Chains

Feed the output of one step into the next:

```python
from esperanto import AIFactory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

langchain_model = AIFactory.create_language("google", "gemini-2.5-flash").to_langchain()

topic_chain = (
    ChatPromptTemplate.from_template("Generate a single interesting topic about {subject}. Reply with the topic only.")
    | langchain_model
    | StrOutputParser()
)

article_chain = (
    ChatPromptTemplate.from_template("Write a short paragraph about: {topic}")
    | langchain_model
    | StrOutputParser()
)

overall_chain = {"topic": topic_chain} | article_chain

result = overall_chain.invoke({"subject": "artificial intelligence"})
print(result)
```

### Agents with Tools

Build a tool-calling agent with `create_agent` (requires `pip install langchain`):

```python
from esperanto import AIFactory
from langchain.agents import create_agent
from langchain_core.tools import tool

langchain_model = AIFactory.create_language("openai", "gpt-4o-mini").to_langchain()

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

agent = create_agent(langchain_model, tools=[search, multiply])

result = agent.invoke({"messages": [{"role": "user", "content": "What is 25 * 17?"}]})
print(result["messages"][-1].content)  # "25 * 17 = 425"
```

### RAG (Retrieval-Augmented Generation)

Use Esperanto for both embeddings and generation. A small adapter exposes any Esperanto embedding model through LangChain's `Embeddings` interface:

```python
from esperanto import AIFactory
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore


class EsperantoEmbeddings(Embeddings):
    """Expose an Esperanto embedding model to LangChain."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.embed(texts)

    def embed_query(self, text):
        return self.model.embed([text])[0]


embeddings = EsperantoEmbeddings(AIFactory.create_embedding("openai", "text-embedding-3-small"))
langchain_model = AIFactory.create_language("openai", "gpt-4o-mini").to_langchain()

documents = [
    Document(page_content="Esperanto is a unified interface for AI models."),
    Document(page_content="It supports multiple providers like OpenAI, Anthropic, and Google."),
    Document(page_content="You can switch providers without changing your code."),
]
vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    "Answer using only this context:\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | langchain_model
    | StrOutputParser()
)

print(rag_chain.invoke("What is Esperanto?"))
```

Swap either provider (for example `create_embedding("voyage", ...)` or `create_language("anthropic", ...)`) without touching the rest of the chain.

## Advanced Configuration

### Streaming with LangChain

LangChain chat models stream with `.stream()` / `.astream()`:

```python
from esperanto import AIFactory

langchain_model = AIFactory.create_language("openai", "gpt-4o-mini").to_langchain()

for chunk in langchain_model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

Chains stream too: `chain.stream({...})` yields the parsed output as it arrives.

### Structured Output with LangChain

Use `with_structured_output()` on the converted model to get a validated Pydantic object:

```python
from esperanto import AIFactory
from pydantic import BaseModel, Field


class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief summary")


langchain_model = AIFactory.create_language("openai", "gpt-4o-mini").to_langchain()
reviewer = langchain_model.with_structured_output(MovieReview)

review = reviewer.invoke("Review the movie The Matrix.")
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Summary: {review.summary}")
```

Alternatively, set `config={"structured": {"type": "json_schema", "schema": MovieReview}}` on the Esperanto model: every provider except Cohere carries it into the converted model, which then returns JSON text matching the schema (see [Language Models](../capabilities/llm.md#structured-output)).

### Multi-Provider Pipeline

Use different providers for different steps of one chain:

```python
from esperanto import AIFactory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Fast model for initial processing
fast_model = AIFactory.create_language("groq", "openai/gpt-oss-20b").to_langchain()

# More capable model for the final output
powerful_model = AIFactory.create_language("anthropic", "claude-sonnet-5").to_langchain()

analysis_chain = (
    ChatPromptTemplate.from_template("Quickly analyze this text and extract key points: {text}")
    | fast_model
    | StrOutputParser()
)

response_chain = (
    ChatPromptTemplate.from_template("Based on this analysis, write a detailed response: {analysis}")
    | powerful_model
    | StrOutputParser()
)

pipeline = {"analysis": analysis_chain} | response_chain

print(pipeline.invoke({"text": "Your input text here"}))
```

## Best Practices

### 1. Choose the Right Model for the Task

```python
from esperanto import AIFactory
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Summarize: {text}")

# Fast models for simple steps
fast_chain = prompt | AIFactory.create_language("groq", "openai/gpt-oss-20b").to_langchain()

# More capable models for complex reasoning
complex_chain = prompt | AIFactory.create_language("anthropic", "claude-sonnet-5").to_langchain()
```

### 2. Leverage Esperanto's Factory Pattern

```python
from esperanto import AIFactory

def create_langchain_model(provider="openai", model_name="gpt-4o-mini"):
    return AIFactory.create_language(provider, model_name).to_langchain()

# Switch providers with a config change
langchain_model = create_langchain_model("anthropic", "claude-haiku-4-5-20251001")
```

### 3. Fall Back to Another Provider

LangChain runnables support fallbacks directly:

```python
from esperanto import AIFactory
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Summarize: {text}")

primary = AIFactory.create_language("openai", "gpt-4o-mini").to_langchain()
backup = AIFactory.create_language("anthropic", "claude-haiku-4-5-20251001").to_langchain()

model_with_fallback = primary.with_fallbacks([backup])
chain = prompt | model_with_fallback
```

### 4. Use Caching for Repeated Queries

```python
from esperanto import AIFactory
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

# Enable caching
set_llm_cache(InMemoryCache())

langchain_model = AIFactory.create_language("openai", "gpt-4o-mini").to_langchain()

# First call - hits the API
result1 = langchain_model.invoke("What is AI?")

# Second call - returns the cached result
result2 = langchain_model.invoke("What is AI?")
```

## Async

Converted models support LangChain's async API (`ainvoke`, `astream`, `abatch`), and so do chains built from them:

```python
import asyncio

from esperanto import AIFactory

langchain_model = AIFactory.create_language("openai", "gpt-4o-mini").to_langchain()


async def main():
    response = await langchain_model.ainvoke("Hello")
    print(response.content)

    async for chunk in langchain_model.astream("Tell me a story"):
        print(chunk.content, end="", flush=True)


asyncio.run(main())
```

Esperanto's native `achat_complete()` remains available on the original model when you want Esperanto's normalized response types instead of LangChain messages.

## Migration from Native LangChain Providers

If you're migrating from native LangChain providers to Esperanto:

### Before (Native LangChain)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key="your-api-key"
)
```

### After (Esperanto + LangChain)

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "openai",
    "gpt-4o-mini",
    config={"temperature": 0.7, "api_key": "your-api-key"},
)

llm = model.to_langchain()
```

### Benefits of Migration

- **Provider flexibility**: Easily switch between OpenAI, Anthropic, Google, etc.
- **Unified interface**: Same code works across providers
- **Advanced features**: Access Esperanto-specific features
- **Consistent configuration**: Timeouts, SSL and structured output configured the same way for every provider

## Troubleshooting

### ImportError when calling `.to_langchain()`

Install the LangChain package for that provider, for example:

```bash
pip install langchain-openai      # or langchain-anthropic, langchain-google-genai, ...
```

### `ModuleNotFoundError: No module named 'langchain.chains'`

`ConversationChain`, `LLMChain`, `SequentialChain`, `RetrievalQA` and `initialize_agent` were removed in LangChain 1.0. Use the LCEL patterns on this page (`prompt | model | parser`, message lists for memory, `create_agent`).

### Passing Esperanto-Style Messages

LangChain chat models accept OpenAI-style message dicts directly:

```python
messages = [{"role": "user", "content": "Your prompt"}]
response = langchain_model.invoke(messages)
```

## See Also

- [Language Model Capabilities](../capabilities/llm.md) - Overview of LLM features
- [OpenAI Provider](../providers/openai.md) - OpenAI-specific features
- [Anthropic Provider](../providers/anthropic.md) - Claude-specific features
- [Timeout Configuration](./timeout-configuration.md) - Configure timeouts
- [LangChain Documentation](https://python.langchain.com/) - Official LangChain docs
