# LLM Providers

Esperanto supports various Large Language Model (LLM) providers through a unified interface.

## Supported Providers

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3 family)
- OpenRouter (Multiple models)
- xAI (Grok)
- Groq (Mixtral, Llama)
- Gemini
- Vertex AI (Google Cloud)
- Ollama (Local deployment)

## Supported Platforms

- Langchain
- Llamaindex *(coming soon)*

## Usage Examples

### Using AI Factory

```python
from esperanto.factory import AIFactory

# Create an LLM instance
model = AIFactory.create_llm("openai", "gpt-3.5-turbo")

# Synchronous usage
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
]
response = model.chat_complete(messages)

# Asynchronous usage
async def get_response():
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Streaming usage
model = AIFactory.create_llm("openai", "gpt-3.5-turbo", streaming=True)
for chunk in model.chat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Structured Output (JSON)

```python
model = AIFactory.create_llm("openai", "gpt-3.5-turbo", structured="json")

messages = [
    {"role": "user", "content": "List three European capitals"}
]

response = model.chat_complete(messages)
# Response will be in JSON format
```

### LangChain Integration

```python
from esperanto.factory import AIFactory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Create LLM and convert to LangChain
model = AIFactory.create_llm("openai", "gpt-3.5-turbo")
langchain_model = model.to_langchain()

# Use with LangChain chains
chain = ConversationChain(llm=langchain_model)
response = chain.invoke("What's the capital of Paris?")

# Use with custom prompts
prompt = PromptTemplate.from_template("Tell me a {adjective} story about {subject}")
formatted_prompt = prompt.format(adjective="funny", subject="a talking cat")
response = langchain_model.invoke(formatted_prompt)
```

### Using Provider-Specific Classes

### OpenAI
```python
model = OpenAILanguageModel(
    api_key="your-api-key",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=850,
    streaming=False,
    structured="json"
)
```

### Groq
```python
model = GroqLanguageModel(
    api_key="your-api-key",
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=850,
    streaming=False,
    structured="json"
)
```

### Ollama
```python
from esperanto.providers.llm.ollama import OllamaLanguageModel

model = OllamaLanguageModel(
    model_name="llama2",  # or any other supported model
    base_url="http://localhost:11434"  # default Ollama server
)
```

### Anthropic
```python
from esperanto.providers.llm.anthropic import AnthropicLanguageModel

model = AnthropicLanguageModel(
    api_key="your-api-key",
    model_name="claude-3-opus-20240229"
)
```

### Gemini
```python
from esperanto.providers.llm.gemini import GeminiLanguageModel

model = GeminiLanguageModel(
    api_key="your-api-key",
    model_name="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=850,
    streaming=False,
    structured="json"
)
```
