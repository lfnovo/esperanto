# LLM Providers

Esperanto supports various Large Language Model (LLM) providers through a unified interface.

## Supported Providers

- OpenAI (GPT-4, GPT-3.5, o1)
- Anthropic (Claude 3 family)
- OpenRouter (Multiple models)
- xAI (Grok)
- Groq (Mixtral, Llama)
- Google GenAI (Gemini)
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
model = AIFactory.create_language("openai", "gpt-3.5-turbo")

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
model = AIFactory.create_language("openai", "gpt-3.5-turbo", streaming=True)
for chunk in model.chat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Structured Output (JSON)

```python
model = AIFactory.create_language(
    "openai", 
    "gpt-3.5-turbo", 
    structured={"type": "json"}
)

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
model = AIFactory.create_language("openai", "gpt-3.5-turbo")
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
    structured={"type": "json"}
)
```

### OpenAI o1 Model

When using OpenAI's o1 model, Esperanto automatically handles the required transformations:
- Replaces `max_tokens` with `max_completion_tokens`
- Sets temperature to 1.0 (required by the model)
- Removes `top_p` parameter
- Converts system messages to user messages

```python
model = AIFactory.create_language("openai", "o1-model")
response = model.chat_complete([
    {"role": "system", "content": "You are a helpful assistant."},  # Will be converted to user role
    {"role": "user", "content": "Hello!"}
])
```

### Groq
```python
model = GroqLanguageModel(
    api_key="your-api-key",
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=850,
    streaming=False,
    structured={"type": "json"}
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

### Goggle Gen AI
```python
from esperanto.providers.llm.google import GoogleLanguageModel

model = GoogleLanguageModel(
    api_key="your-api-key",
    model_name="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=850,
    streaming=False,
    structured={"type": "json"}
)
```
