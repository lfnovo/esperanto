# LangChain Integration

## Overview

Esperanto provides seamless integration with LangChain, allowing you to convert any language model provider into a LangChain-compatible chat model. This enables you to leverage Esperanto's unified interface while benefiting from LangChain's rich ecosystem of chains, agents, and tools.

## Prerequisites

LangChain must be installed to use this feature:

```bash
pip install langchain
# Or with LangChain community packages
pip install langchain langchain-community
```

## Quick Start

Convert any Esperanto language model to LangChain format:

```python
from esperanto import AIFactory

# Create an Esperanto model
model = AIFactory.create_language("openai", "gpt-4", api_key="your-api-key")

# Convert to LangChain chat model
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)

response = chain.run("Hello! How are you?")
print(response)
```

## Supported Providers

The `.to_langchain()` method works with all language model providers in Esperanto:

### OpenAI

```python
from esperanto.providers.llm.openai import OpenAILanguageModel

model = OpenAILanguageModel(
    api_key="your-api-key",
    model_name="gpt-4"
)

langchain_model = model.to_langchain()
```

### Anthropic (Claude)

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "anthropic",
    "claude-3-5-sonnet-20241022",
    api_key="your-api-key"
)

langchain_model = model.to_langchain()
```

### Google (Gemini)

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "google",
    "gemini-1.5-pro",
    api_key="your-api-key"
)

langchain_model = model.to_langchain()
```

### Groq

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "groq",
    "mixtral-8x7b-32768",
    api_key="your-api-key"
)

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

### Conversation Chain

Create a conversational agent with memory:

```python
from esperanto import AIFactory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Create Esperanto model
model = AIFactory.create_language("openai", "gpt-4")
langchain_model = model.to_langchain()

# Create conversation chain with memory
conversation = ConversationChain(
    llm=langchain_model,
    memory=ConversationBufferMemory()
)

# Have a conversation
response1 = conversation.run("My name is Alice")
print(response1)  # "Nice to meet you, Alice!"

response2 = conversation.run("What's my name?")
print(response2)  # "Your name is Alice"
```

### LLM Chain with Prompt Templates

Use LangChain's prompt templating:

```python
from esperanto import AIFactory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create model
model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
langchain_model = model.to_langchain()

# Create prompt template
template = """You are a helpful assistant that translates {input_language} to {output_language}.

Text to translate: {text}

Translation:"""

prompt = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template=template
)

# Create chain
chain = LLMChain(llm=langchain_model, prompt=prompt)

# Use chain
result = chain.run(
    input_language="English",
    output_language="Spanish",
    text="Hello, how are you?"
)
print(result)  # "Hola, ¿cómo estás?"
```

### Sequential Chains

Combine multiple chains:

```python
from esperanto import AIFactory
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate

# Create model
model = AIFactory.create_language("google", "gemini-1.5-pro")
langchain_model = model.to_langchain()

# First chain: Generate a topic
topic_template = "Generate a single interesting topic about {subject}"
topic_prompt = PromptTemplate(input_variables=["subject"], template=topic_template)
topic_chain = LLMChain(llm=langchain_model, prompt=topic_prompt)

# Second chain: Write about the topic
article_template = "Write a short paragraph about: {topic}"
article_prompt = PromptTemplate(input_variables=["topic"], template=article_template)
article_chain = LLMChain(llm=langchain_model, prompt=article_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[topic_chain, article_chain],
    verbose=True
)

# Execute
result = overall_chain.run("artificial intelligence")
print(result)
```

### Agents with Tools

Create agents that can use tools:

```python
from esperanto import AIFactory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Create model
model = AIFactory.create_language("openai", "gpt-4")
langchain_model = model.to_langchain()

# Define tools
def search_tool(query: str) -> str:
    """Simulated search tool"""
    return f"Search results for: {query}"

def calculator_tool(expression: str) -> str:
    """Simple calculator"""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for searching information"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for mathematical calculations"
    )
]

# Create agent
agent = initialize_agent(
    tools,
    langchain_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use agent
result = agent.run("What is 25 * 17?")
print(result)
```

### RAG (Retrieval-Augmented Generation)

Combine with vector stores for RAG:

```python
from esperanto import AIFactory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Create language model
llm = AIFactory.create_language("openai", "gpt-4")
langchain_llm = llm.to_langchain()

# Prepare documents
documents = [
    "Esperanto is a unified interface for AI models.",
    "It supports multiple providers like OpenAI, Anthropic, and Google.",
    "You can easily switch between providers without changing your code."
]

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.create_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=langchain_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
response = qa_chain.run("What is Esperanto?")
print(response)
```

## Advanced Configuration

### Streaming with LangChain

Esperanto's streaming capabilities work through LangChain:

```python
from esperanto import AIFactory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create streaming model
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"streaming": True}
)

langchain_model = model.to_langchain()

# Use with streaming callback
response = langchain_model.invoke(
    "Tell me a story",
    config={"callbacks": [StreamingStdOutCallbackHandler()]}
)
```

### Structured Output with LangChain

Combine Esperanto's structured output with LangChain:

```python
from esperanto import AIFactory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Define output structure
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief summary")

# Create model with JSON output
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"structured": {"type": "json"}}
)

langchain_model = model.to_langchain()

# Create parser
parser = PydanticOutputParser(pydantic_object=MovieReview)

# Create prompt with format instructions
template = """Review the following movie and provide a structured response.

{format_instructions}

Movie: {movie_name}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["movie_name"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Use chain
from langchain.chains import LLMChain
chain = LLMChain(llm=langchain_model, prompt=prompt)

result = chain.run(movie_name="The Matrix")
review = parser.parse(result)
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Summary: {review.summary}")
```

### Multi-Provider Strategy

Use different providers for different parts of your pipeline:

```python
from esperanto import AIFactory
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate

# Fast model for initial processing
fast_model = AIFactory.create_language("groq", "mixtral-8x7b-32768")
fast_langchain = fast_model.to_langchain()

# Powerful model for final output
powerful_model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
powerful_langchain = powerful_model.to_langchain()

# Chain 1: Fast initial analysis
analysis_prompt = PromptTemplate(
    input_variables=["text"],
    template="Quickly analyze this text and extract key points: {text}"
)
analysis_chain = LLMChain(
    llm=fast_langchain,
    prompt=analysis_prompt,
    output_key="analysis"
)

# Chain 2: Detailed response with powerful model
response_prompt = PromptTemplate(
    input_variables=["analysis"],
    template="Based on this analysis, write a detailed response: {analysis}"
)
response_chain = LLMChain(
    llm=powerful_langchain,
    prompt=response_prompt,
    output_key="response"
)

# Combine chains
overall_chain = SequentialChain(
    chains=[analysis_chain, response_chain],
    input_variables=["text"],
    output_variables=["response"],
    verbose=True
)

result = overall_chain({"text": "Your input text here"})
print(result["response"])
```

## Best Practices

### 1. Choose the Right Model for the Task

```python
# Use fast models for simple tasks
fast_model = AIFactory.create_language("groq", "llama3-8b-8192")
fast_chain = LLMChain(llm=fast_model.to_langchain(), prompt=simple_prompt)

# Use powerful models for complex reasoning
powerful_model = AIFactory.create_language("anthropic", "claude-3-5-sonnet-20241022")
complex_chain = LLMChain(llm=powerful_model.to_langchain(), prompt=complex_prompt)
```

### 2. Leverage Esperanto's Factory Pattern

```python
# Easy provider switching
def create_langchain_model(provider="openai", model_name="gpt-4"):
    esperanto_model = AIFactory.create_language(provider, model_name)
    return esperanto_model.to_langchain()

# Switch providers with a config change
langchain_model = create_langchain_model("anthropic", "claude-3-5-haiku-20241022")
```

### 3. Handle Errors Gracefully

```python
from langchain.chains import LLMChain

model = AIFactory.create_language("openai", "gpt-4")
langchain_model = model.to_langchain()

chain = LLMChain(llm=langchain_model, prompt=prompt)

try:
    result = chain.run(input_text)
except Exception as e:
    print(f"Error: {e}")
    # Fallback to different provider
    backup_model = AIFactory.create_language("groq", "mixtral-8x7b-32768")
    backup_chain = LLMChain(llm=backup_model.to_langchain(), prompt=prompt)
    result = backup_chain.run(input_text)
```

### 4. Use Caching for Repeated Queries

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(InMemoryCache())

model = AIFactory.create_language("openai", "gpt-4")
langchain_model = model.to_langchain()

# First call - hits API
result1 = langchain_model.invoke("What is AI?")

# Second call - returns cached result
result2 = langchain_model.invoke("What is AI?")
```

## Limitations

### Not Supported

The following Esperanto features are not directly available through the LangChain interface:

- **Async methods**: Use Esperanto's native `achat_complete()` instead
- **Direct streaming control**: Use LangChain's callback system
- **Custom response handling**: LangChain handles response parsing

### Workarounds

For features not available through LangChain, use Esperanto directly:

```python
# For async operations
model = AIFactory.create_language("openai", "gpt-4")

# Use Esperanto directly
messages = [{"role": "user", "content": "Hello"}]
async_result = await model.achat_complete(messages)

# For LangChain compatibility
langchain_model = model.to_langchain()
langchain_result = langchain_model.invoke("Hello")
```

## Migration from Native LangChain Providers

If you're migrating from native LangChain providers to Esperanto:

### Before (Native LangChain)

```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    openai_api_key="your-api-key"
)
```

### After (Esperanto + LangChain)

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"temperature": 0.7},
    api_key="your-api-key"
)

llm = model.to_langchain()
```

### Benefits of Migration

- **Provider flexibility**: Easily switch between OpenAI, Anthropic, Google, etc.
- **Unified interface**: Same code works across providers
- **Advanced features**: Access Esperanto-specific features
- **Better error handling**: Consistent error handling across providers

## Troubleshooting

### Import Error: langchain not found

```bash
pip install langchain
```

### Type Compatibility Issues

Some LangChain features expect specific types. Convert as needed:

```python
# If LangChain expects string input
langchain_model.invoke("Your prompt here")

# If you have Esperanto messages format
messages = [{"role": "user", "content": "Your prompt"}]
# Convert to string for LangChain
prompt_text = messages[0]["content"]
langchain_model.invoke(prompt_text)
```

### Streaming Not Working

Ensure streaming is enabled in the Esperanto model:

```python
model = AIFactory.create_language(
    "openai",
    "gpt-4",
    config={"streaming": True}
)

langchain_model = model.to_langchain()
```

## See Also

- [Language Model Capabilities](/docs/capabilities/language.md) - Overview of LLM features
- [OpenAI Provider](/docs/providers/openai.md) - OpenAI-specific features
- [Anthropic Provider](/docs/providers/anthropic.md) - Claude-specific features
- [Timeout Configuration](/docs/advanced/timeout-configuration.md) - Configure timeouts
- [LangChain Documentation](https://python.langchain.com/) - Official LangChain docs
