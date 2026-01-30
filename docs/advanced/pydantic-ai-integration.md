# Pydantic AI Integration

## Overview

Esperanto provides seamless integration with [Pydantic AI](https://ai.pydantic.dev/), allowing you to use any Esperanto language model provider with Pydantic AI agents. This enables you to leverage Esperanto's unified interface and provider flexibility while benefiting from Pydantic AI's powerful agent framework, structured outputs, and tool calling capabilities.

## Why Use Esperanto with Pydantic AI?

**Provider Flexibility**: Pydantic AI has built-in support for OpenAI, Anthropic, and a few other providers, but each requires provider-specific configuration. With Esperanto, you configure your model once and use it with any of 15+ providers without changing your agent code.

**Single Source of Truth**: Configure model parameters (temperature, max_tokens, API keys, base URLs) in one place through Esperanto, then use the same configuration across your entire application.

**Hot-Swap Providers**: Switch between providers (OpenAI → Anthropic → Google → local Ollama) by changing a single parameter, without touching your Pydantic AI agent code.

## Prerequisites

Install pydantic-ai alongside esperanto:

```bash
pip install 'pydantic-ai>=1.50.0'
# Or with uv
uv add 'pydantic-ai>=1.50.0'
```

## Quick Start

Convert any Esperanto language model to Pydantic AI format:

```python
from esperanto import AIFactory
from pydantic_ai import Agent

# Create an Esperanto model
model = AIFactory.create_language("openai", "gpt-4o")

# Convert to Pydantic AI model
pydantic_model = model.to_pydantic_ai()

# Use with Pydantic AI Agent
agent = Agent(pydantic_model)
result = await agent.run("What is the capital of France?")
print(result.output)
```

## Supported Providers

The `.to_pydantic_ai()` method works with all language model providers in Esperanto:

### OpenAI

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "openai",
    "gpt-4o",
    config={"temperature": 0.7}
)

agent = Agent(model.to_pydantic_ai())
```

### Anthropic (Claude)

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "anthropic",
    "claude-sonnet-4-20250514",
    config={"temperature": 0.7}
)

agent = Agent(model.to_pydantic_ai())
```

### Google (Gemini)

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "google",
    "gemini-2.5-flash",
    config={"temperature": 0.7}
)

agent = Agent(model.to_pydantic_ai())
```

### Azure OpenAI

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "azure",
    "gpt-4o",
    config={
        "api_key": "your-azure-key",
        "base_url": "https://your-resource.openai.azure.com",
        "api_version": "2024-02-15-preview"
    }
)

agent = Agent(model.to_pydantic_ai())
```

### Groq

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "groq",
    "llama-3.3-70b-versatile",
    config={"temperature": 0.7}
)

agent = Agent(model.to_pydantic_ai())
```

### Mistral

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "mistral",
    "mistral-large-latest",
    config={"temperature": 0.7}
)

agent = Agent(model.to_pydantic_ai())
```

### xAI (Grok)

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "xai",
    "grok-3",
    config={"temperature": 0.7}
)

agent = Agent(model.to_pydantic_ai())
```

### DeepSeek

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "deepseek",
    "deepseek-chat",
    config={"temperature": 0.7}
)

agent = Agent(model.to_pydantic_ai())
```

### Ollama (Local)

```python
from esperanto import AIFactory

model = AIFactory.create_language(
    "ollama",
    "llama3.2",
    config={
        "base_url": "http://localhost:11434",
        "timeout": 120  # Local models may need longer timeout
    }
)

agent = Agent(model.to_pydantic_ai())
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

agent = Agent(model.to_pydantic_ai())
```

## Use Cases

### Tool Calling

Pydantic AI's tool calling works seamlessly with Esperanto models:

```python
import random
from esperanto import AIFactory
from pydantic_ai import Agent, RunContext

model = AIFactory.create_language("openai", "gpt-4o").to_pydantic_ai()

agent = Agent(
    model,
    deps_type=str,
    instructions=(
        "You're a dice game. Roll the die and check if it matches "
        "the user's guess. Use the player's name in your response."
    ),
)

@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps

result = await agent.run("My guess is 4", deps="Alice")
print(result.output)
# "I rolled a 3, Alice. Unfortunately, that doesn't match your guess of 4!"
```

### Structured Output

Use Pydantic models for type-safe structured outputs:

```python
from esperanto import AIFactory
from pydantic_ai import Agent
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

model = AIFactory.create_language("anthropic", "claude-sonnet-4-20250514").to_pydantic_ai()

agent = Agent(model, output_type=MovieReview)

result = await agent.run("Review the movie 'Inception'")
review = result.output  # MovieReview instance
print(f"{review.title}: {review.rating}/10")
print(review.summary)
```

### Streaming Responses

Stream responses for real-time output:

```python
from esperanto import AIFactory
from pydantic_ai import Agent

model = AIFactory.create_language("openai", "gpt-4o").to_pydantic_ai()
agent = Agent(model)

async with agent.run_stream("Tell me a story") as response:
    async for chunk in response.stream_text():
        print(chunk, end="", flush=True)
```

### Provider Switching

The same agent code works with any provider:

```python
from esperanto import AIFactory
from pydantic_ai import Agent

def create_agent(provider: str, model_name: str) -> Agent:
    """Create an agent with any provider."""
    model = AIFactory.create_language(provider, model_name)
    return Agent(model.to_pydantic_ai())

# Use with different providers - same agent code!
agent_openai = create_agent("openai", "gpt-4o")
agent_anthropic = create_agent("anthropic", "claude-sonnet-4-20250514")
agent_google = create_agent("google", "gemini-2.5-flash")
agent_local = create_agent("ollama", "llama3.2")

# All agents work identically
for agent in [agent_openai, agent_anthropic, agent_google, agent_local]:
    result = await agent.run("Hello!")
    print(result.output)
```

### Conversation with Memory

Multi-turn conversations with context:

```python
from esperanto import AIFactory
from pydantic_ai import Agent

model = AIFactory.create_language("openai", "gpt-4o").to_pydantic_ai()
agent = Agent(model)

# First turn
result1 = await agent.run("My name is Alice")
print(result1.output)

# Continue conversation with history
result2 = await agent.run(
    "What's my name?",
    message_history=result1.all_messages()
)
print(result2.output)  # "Your name is Alice"
```

### System Instructions

Customize agent behavior with instructions:

```python
from esperanto import AIFactory
from pydantic_ai import Agent

model = AIFactory.create_language("anthropic", "claude-sonnet-4-20250514").to_pydantic_ai()

agent = Agent(
    model,
    instructions="""You are a helpful coding assistant.
    Always provide code examples in Python.
    Explain your reasoning step by step."""
)

result = await agent.run("How do I read a file?")
print(result.output)
```

## Advanced Configuration

### Model Settings Override

Override settings per-request:

```python
from esperanto import AIFactory
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

model = AIFactory.create_language(
    "openai",
    "gpt-4o",
    config={"temperature": 0.7}  # Default temperature
).to_pydantic_ai()

agent = Agent(model)

# Override temperature for this request
result = await agent.run(
    "Generate a creative story",
    model_settings=ModelSettings(temperature=1.0)
)
```

### Timeout Configuration

Configure timeouts for slow providers (like local models):

```python
from esperanto import AIFactory
from pydantic_ai import Agent

# Set timeout at Esperanto level
model = AIFactory.create_language(
    "ollama",
    "llama3.2",
    config={"timeout": 180}  # 3 minutes
).to_pydantic_ai()

agent = Agent(model)
```

Or via environment variable:

```bash
export ESPERANTO_LLM_TIMEOUT=180
```

### Accessing the Original Esperanto Model

The Pydantic AI adapter maintains a reference to the original Esperanto model:

```python
from esperanto import AIFactory
from pydantic_ai import Agent

model = AIFactory.create_language("openai", "gpt-4o", config={"temperature": 0.5})
pydantic_model = model.to_pydantic_ai()

# Access original Esperanto model
print(pydantic_model._esperanto_model.temperature)  # 0.5
print(pydantic_model.model_name)  # "gpt-4o"
print(pydantic_model.system)  # "openai"
```

## Provider-Specific Notes

### Google (Gemini)

Google's API has stricter JSON schema requirements. Esperanto automatically cleans schemas by removing unsupported fields like `additionalProperties`.

### Ollama / Local Models

- Set appropriate timeouts (models need time to load)
- First request may be slow as the model loads into memory
- Use environment variable `OLLAMA_BASE_URL` for custom endpoints

### OpenAI-Compatible (LM Studio, vLLM)

Tool calling support depends on the model:
- **Native support**: Qwen2.5, Llama-3.1/3.2, Mistral models
- **Limited support**: Other models may not reliably call tools

### DeepSeek

DeepSeek supports function calling but may require explicit prompting for reliable tool use.

## Troubleshooting

### ImportError: pydantic-ai not found

```bash
pip install 'pydantic-ai>=1.50.0'
```

### Model Not Calling Tools

Some models don't reliably use function calling. Try:
1. Using a more capable model (GPT-4o, Claude Sonnet, etc.)
2. Being more explicit in your prompts
3. Testing the tools directly with Esperanto first

### Timeout Errors

Increase the timeout:

```python
model = AIFactory.create_language(
    "provider",
    "model",
    config={"timeout": 120}
)
```

### Google Schema Errors

If you see errors about `additionalProperties`, ensure you're using the latest Esperanto version which automatically cleans schemas.

## Migration from Native Pydantic AI

### Before (Native Pydantic AI)

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel("gpt-4o", api_key="your-key")
agent = Agent(model)
```

### After (Esperanto + Pydantic AI)

```python
from esperanto import AIFactory
from pydantic_ai import Agent

model = AIFactory.create_language("openai", "gpt-4o")
agent = Agent(model.to_pydantic_ai())
```

### Benefits of Migration

- **Provider flexibility**: Switch providers without code changes
- **Unified configuration**: Same config pattern for all providers
- **Additional providers**: Access to 15+ providers not natively supported by Pydantic AI
- **Consistent interface**: Same API across all providers

## See Also

- [Language Model Capabilities](../capabilities/llm.md) - Overview of LLM features
- [Tool Calling](../features/tool-calling.md) - Esperanto's native tool calling
- [LangChain Integration](./langchain-integration.md) - Alternative integration
- [Provider Setup Guides](../providers/) - Provider-specific configuration
- [Pydantic AI Documentation](https://ai.pydantic.dev/) - Official Pydantic AI docs
