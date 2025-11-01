# Perplexity

## Overview

Perplexity provides AI models with integrated web search capabilities, enabling access to up-to-date information and real-time data. Their Sonar models combine language understanding with live web search for current, factual responses.

**Supported Capabilities:**

| Capability | Supported | Notes |
|------------|-----------|-------|
| Language Models (LLM) | ✅ | Sonar models with web search integration |
| Embeddings | ❌ | Not available |
| Reranking | ❌ | Not available |
| Speech-to-Text | ❌ | Not available |
| Text-to-Speech | ❌ | Not available |

**Official Documentation:** https://docs.perplexity.ai

## Prerequisites

### Account Requirements
- Perplexity account (sign up at https://www.perplexity.ai)
- API key with credits or billing enabled

### Getting API Keys
1. Visit https://www.perplexity.ai/settings/api
2. Generate a new API key
3. Copy and store the key securely

## Environment Variables

```bash
# Perplexity API key (required)
PERPLEXITY_API_KEY="pplx-..."
```

**Variable Priority:**
1. Direct parameter in code (`api_key="..."`)
2. Environment variable (`PERPLEXITY_API_KEY`)

## Quick Start

### Via Factory (Recommended)

```python
from esperanto.factory import AIFactory

# Create Perplexity model
model = AIFactory.create_language("perplexity", "llama-3.1-sonar-large-128k-online")

# Ask a question with real-time web search
messages = [{"role": "user", "content": "What are the latest AI developments?"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

### Direct Instantiation

```python
from esperanto.providers.llm.perplexity import PerplexityLanguageModel

# Create model instance
model = PerplexityLanguageModel(
    api_key="your-api-key",
    model_name="llama-3.1-sonar-large-128k-online"
)

# Use the model
messages = [{"role": "user", "content": "Current Bitcoin price?"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

## Capabilities

### Language Models (LLM)

**Available Models:**

| Model | Context Window | Web Search | Best For |
|-------|----------------|------------|----------|
| **llama-3.1-sonar-small-128k-online** | 128K tokens | ✅ | Fast, current information |
| **llama-3.1-sonar-large-128k-online** | 128K tokens | ✅ | Comprehensive research |
| **llama-3.1-sonar-huge-128k-online** | 128K tokens | ✅ | Most capable, detailed answers |

**Configuration:**

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "temperature": 0.7,                      # Randomness (0.0 - 1.0)
        "max_tokens": 1000,                      # Maximum response length
        "top_p": 0.9,                            # Nucleus sampling
        "streaming": True,                       # Enable streaming
        "search_domain_filter": ["news.com"],   # Limit search domains
        "return_images": True,                   # Include images in results
        "return_related_questions": True,        # Get related questions
        "search_recency_filter": "week"          # Filter by time period
    }
)
```

**Example - Basic Web Search:**

```python
from esperanto.factory import AIFactory

# Create Perplexity model with web search
model = AIFactory.create_language("perplexity", "llama-3.1-sonar-large-128k-online")

# Ask about current events
messages = [{"role": "user", "content": "What are today's top tech news?"}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - With Search Domain Filter:**

```python
# Filter search to specific domains
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "search_domain_filter": ["techcrunch.com", "theverge.com", "arstechnica.com"]
    }
)

messages = [{"role": "user", "content": "Latest AI breakthroughs?"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Exclude Domains:**

```python
# Exclude specific domains (prefix with -)
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "search_domain_filter": ["-spam.com", "-unreliable.org"]
    }
)

messages = [{"role": "user", "content": "Climate change research"}]
response = model.chat_complete(messages)
```

**Example - Search Recency Filter:**

```python
# Get only recent information
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "search_recency_filter": "day"  # Options: "day", "week", "month", "year"
    }
)

messages = [{"role": "user", "content": "Today's stock market performance"}]
response = model.chat_complete(messages)
```

**Example - With Related Questions:**

```python
# Request related questions
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={"return_related_questions": True}
)

messages = [{"role": "user", "content": "How does quantum computing work?"}]
response = model.chat_complete(messages)
print(response.choices[0].message.content)
# Related questions will be included in the response
```

**Example - Include Images:**

```python
# Request image results
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={"return_images": True}
)

messages = [{"role": "user", "content": "Show me the latest SpaceX launches"}]
response = model.chat_complete(messages)
# Images will be included in the response
```

**Example - Streaming:**

```python
model = AIFactory.create_language("perplexity", "llama-3.1-sonar-large-128k-online")

messages = [{"role": "user", "content": "Explain recent developments in AI"}]

# Synchronous streaming
for chunk in model.chat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async for chunk in model.achat_complete(messages, stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

**Example - Multi-turn Conversation:**

```python
# Build conversation with context
messages = [
    {"role": "user", "content": "What are the latest AI models released this month?"},
    {"role": "assistant", "content": "This month saw releases of..."},
    {"role": "user", "content": "Which one is best for coding?"}
]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
```

**Example - Async Chat:**

```python
async def research_async():
    model = AIFactory.create_language("perplexity", "llama-3.1-sonar-large-128k-online")

    messages = [{"role": "user", "content": "Latest findings on renewable energy"}]
    response = await model.achat_complete(messages)
    print(response.choices[0].message.content)

# Run async
# await research_async()
```

**Example - JSON Mode:**

```python
# Enable JSON output
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={"structured": {"type": "json"}}
)

messages = [{
    "role": "user",
    "content": "List the top 5 tech companies by market cap as JSON with name and value"
}]

response = model.chat_complete(messages)
print(response.choices[0].message.content)
# Response will be valid JSON
```

## Advanced Features

### Search Domain Filtering

Control which domains are searched:

```python
# Include specific domains
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "search_domain_filter": [
            "arxiv.org",           # Include research papers
            "github.com",          # Include code repositories
            "stackoverflow.com"    # Include developer Q&A
        ]
    }
)

# Exclude unreliable sources
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "search_domain_filter": [
            "-tabloid.com",        # Exclude tabloids
            "-spam.org",           # Exclude spam sites
            "news.com",            # Include trusted news
            "research.edu"         # Include research sites
        ]
    }
)
```

### Search Recency Control

Filter results by time period:

```python
# Get very recent information (last 24 hours)
day_model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={"search_recency_filter": "day"}
)

# Get recent information (last 7 days)
week_model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={"search_recency_filter": "week"}
)

# Get information from last month
month_model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={"search_recency_filter": "month"}
)

# Get information from last year
year_model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={"search_recency_filter": "year"}
)
```

### Web Search Options

Control search context and behavior:

```python
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "web_search_options": {
            "context_size": "large",  # Amount of context to use
            "num_results": 10          # Number of search results
        }
    }
)
```

### Enhanced Results

Get additional information with responses:

```python
# Get images and related questions
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "return_images": True,
        "return_related_questions": True
    }
)

messages = [{"role": "user", "content": "Latest Mars rover discoveries"}]
response = model.chat_complete(messages)
# Response includes images and suggested follow-up questions
```

### LangChain Integration

```python
from esperanto.factory import AIFactory

model = AIFactory.create_language("perplexity", "llama-3.1-sonar-large-128k-online")
langchain_model = model.to_langchain()

# Use with LangChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=langchain_model)
```

## Model Selection Guide

### Sonar Small (Recommended for Speed)
**Best for:** Quick answers, high-volume queries
- Fastest response times
- Most cost-effective
- Still provides accurate, current information
- Good for simple questions

```python
model = AIFactory.create_language("perplexity", "llama-3.1-sonar-small-128k-online")
```

### Sonar Large (Recommended for Quality)
**Best for:** Balanced performance, comprehensive answers
- Excellent balance of speed and quality
- More detailed responses
- Better reasoning capabilities
- Good for most use cases

```python
model = AIFactory.create_language("perplexity", "llama-3.1-sonar-large-128k-online")
```

### Sonar Huge (Best Quality)
**Best for:** Complex research, detailed analysis
- Most comprehensive responses
- Best reasoning and analysis
- Highest quality answers
- Use when quality is paramount

```python
model = AIFactory.create_language("perplexity", "llama-3.1-sonar-huge-128k-online")
```

## Use Cases

### Real-Time Information

```python
# Current events
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={"search_recency_filter": "day"}
)

messages = [{"role": "user", "content": "What happened in tech today?"}]
```

### Research and Analysis

```python
# Academic research
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-huge-128k-online",
    config={
        "search_domain_filter": ["arxiv.org", "scholar.google.com"],
        "return_related_questions": True
    }
)

messages = [{"role": "user", "content": "Latest quantum computing research"}]
```

### Market Intelligence

```python
# Financial information
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "search_domain_filter": ["bloomberg.com", "reuters.com", "wsj.com"],
        "search_recency_filter": "day"
    }
)

messages = [{"role": "user", "content": "Today's market analysis for tech stocks"}]
```

### Content Verification

```python
# Fact-checking with credible sources
model = AIFactory.create_language(
    "perplexity",
    "llama-3.1-sonar-large-128k-online",
    config={
        "search_domain_filter": [
            "reuters.com",
            "apnews.com",
            "factcheck.org",
            "-tabloid.com",
            "-conspiracy.org"
        ]
    }
)

messages = [{"role": "user", "content": "Verify this claim: [claim to verify]"}]
```

## Troubleshooting

### Common Errors

**Authentication Error:**
```
Error: Invalid API key
```
**Solution:** Verify your API key is correct and active at https://www.perplexity.ai/settings/api

**Rate Limit Error:**
```
Error: Rate limit exceeded
```
**Solution:** Implement retry logic with exponential backoff or upgrade your plan

**Context Length Exceeded:**
```
Error: Request exceeds token limit
```
**Solution:** Reduce message history or content length (max 128K tokens)

**Invalid Domain Filter:**
```
Error: Invalid search_domain_filter
```
**Solution:** Check domain format (e.g., "domain.com" not "https://domain.com")

**Invalid Recency Filter:**
```
Error: Invalid search_recency_filter
```
**Solution:** Use valid values: "day", "week", "month", or "year"

### Best Practices

1. **Use Domain Filtering:** Always filter domains for focused, reliable results

2. **Set Recency Appropriately:** Use "day" for current events, "month" or "year" for historical

3. **Choose Right Model:** Small for speed, Large for balance, Huge for quality

4. **Enable Related Questions:** Helps users explore topics more deeply

5. **Stream for UX:** Use streaming for better user experience with longer responses

6. **Exclude Unreliable Sources:** Use "-domain.com" to exclude low-quality sites

## Performance Characteristics

### Response Times
- **Small**: 1-3 seconds (with web search)
- **Large**: 2-5 seconds (with web search)
- **Huge**: 3-8 seconds (with web search)

### Context Window
All Sonar models support 128K token context:
- Approximately 96,000 words
- Extensive conversation history
- Long document processing

### Web Search Integration
- Real-time web search included
- Automatic citation of sources
- Up-to-date information (as recent as today)
- Configurable search parameters

## See Also

- [Language Models Guide](../capabilities/llm.md)
- [OpenAI Provider](./openai.md)
- [Anthropic Provider](./anthropic.md)
- [Google Provider](./google.md)
