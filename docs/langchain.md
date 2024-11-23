# LangChain Integration

Esperanto provides seamless integration with LangChain, allowing you to use all providers within the LangChain ecosystem. This guide explains how to use Esperanto models with LangChain.

## Installation

To use LangChain features, install Esperanto with the LangChain extra:

```bash
# Using pip
pip install "esperanto[langchain]"

# Using poetry
poetry install --with langchain
```

## Language Models

All Esperanto language models can be converted to LangChain chat models:

```python
from esperanto.factory import AIFactory

# Create an Esperanto LLM
llm = AIFactory.create_llm("openai", "gpt-4")

# Convert to LangChain model
langchain_model = llm.to_langchain()

# Use with LangChain components
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | langchain_model
result = await chain.ainvoke({"input": "What is quantum computing?"})
```

## Embedding Models

Esperanto embedding models can be used with LangChain for vector operations:

```python
# Create an Esperanto embedding model
embeddings = AIFactory.create_embeddings("openai", "text-embedding-3-small")

# Convert to LangChain embeddings
langchain_embeddings = embeddings.to_langchain()

# Use with LangChain vector stores
from langchain.vectorstores import Chroma

texts = ["Hello world", "Bye world", "Hello there"]
vectorstore = await Chroma.afrom_texts(
    texts=texts,
    embedding=langchain_embeddings,
    collection_name="my_collection"
)

# Search similar documents
query = "Hi world"
docs = await vectorstore.asimilarity_search(query)
```

## Retrieval Chains

Combine Esperanto models with LangChain for sophisticated retrieval:

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Create components
llm = AIFactory.create_llm("openai", "gpt-4").to_langchain()
embeddings = AIFactory.create_embeddings("openai", "text-embedding-3-small").to_langchain()

# Create vector store
vectorstore = await Chroma.afrom_texts(
    texts=texts,
    embedding=embeddings,
    collection_name="my_collection"
)
retriever = vectorstore.as_retriever()

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context:\n\n{context}"),
    ("human", "{input}")
])

# Create chains
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Run chain
response = await retrieval_chain.ainvoke({"input": "What is the meaning of life?"})
```

## Agents

Use Esperanto models with LangChain agents:

```python
from langchain.agents import create_openai_functions_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor

# Create Esperanto LLM
llm = AIFactory.create_llm("openai", "gpt-4").to_langchain()

# Create tools
tools = [DuckDuckGoSearchRun()]

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human", "{input}")
])

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run agent
result = await agent_executor.ainvoke({"input": "What's the weather in San Francisco?"})
```

## Memory

Integrate Esperanto models with LangChain memory systems:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain
llm = AIFactory.create_llm("openai", "gpt-4").to_langchain()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Have a conversation
await conversation.ainvoke({"input": "Hi! My name is Bob"})
await conversation.ainvoke({"input": "What's my name?"})  # The model remembers "Bob"
```

## Best Practices

1. **Model Selection**:
   - Choose appropriate models for your use case
   - Consider cost and performance trade-offs
   - Use streaming for better user experience

2. **Error Handling**:
   - Implement proper error handling for both Esperanto and LangChain
   - Handle rate limits and quotas
   - Consider fallback options

3. **Memory Management**:
   - Clear memory when appropriate
   - Use appropriate memory types
   - Consider persistence requirements

4. **Performance**:
   - Cache embeddings when possible
   - Use batching for multiple operations
   - Implement proper retry mechanisms

## Examples

### Question Answering with Sources

```python
from langchain.chains import create_retrieval_qa_chain
from langchain.vectorstores import Chroma

# Create components
llm = AIFactory.create_llm("openai", "gpt-4").to_langchain()
embeddings = AIFactory.create_embeddings("openai", "text-embedding-3-small").to_langchain()

# Create vector store
docs = [
    "The sky is blue because of Rayleigh scattering.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth orbits around the Sun."
]
vectorstore = await Chroma.afrom_texts(docs, embeddings)

# Create chain
chain = create_retrieval_qa_chain(llm, vectorstore.as_retriever())

# Ask questions
result = await chain.ainvoke({
    "query": "Why is the sky blue?"
})
print(result["answer"])  # Explains Rayleigh scattering with source
```

### Chatbot with Memory

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

# Create components
llm = AIFactory.create_llm("openai", "gpt-4").to_langchain()
memory = ConversationBufferWindowMemory(k=5)  # Remember last 5 exchanges

# Create chain
chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat
responses = []
questions = [
    "Hi! I'm learning about physics.",
    "Can you explain quantum entanglement?",
    "That's interesting! How does it relate to quantum computing?",
    "Can you remind me what we were talking about at the start?"
]

for question in questions:
    response = await chain.ainvoke({"input": question})
    responses.append(response["response"])
```
