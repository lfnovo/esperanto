# Embeddings in Esperanto

Transform text into powerful vector representations that capture semantic meaning. Whether you're building search engines, recommendation systems, or AI assistants, embeddings are your gateway to understanding text at scale.

## 🚀 Quick Start

```python
from esperanto.factory import AIFactory

# Create an embedding model
model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Transform text into vectors
texts = ["I love programming", "Coding is amazing", "The weather is nice"]
response = model.embed(texts)

# Use the vectors
for i, data in enumerate(response.data):
    print(f"Text: {texts[i]}")
    print(f"Vector dimensions: {len(data.embedding)}")
```

## 📖 Documentation Guide

### 🌟 [Getting Started Guide](guide.md)
**Start here if you're new to embeddings**
- What are embeddings and why use them?
- Common use cases with practical examples
- Choosing the right approach for your needs

### 🔧 [Provider Reference](providers.md) 
**Technical details for each provider**
- Complete provider comparison
- Configuration options
- When to use each provider

### ⚡ [Advanced Features](advanced.md)
**Unlock the full power of modern embeddings**
- Task-aware embeddings (NEW!)
- Performance optimization
- Production considerations

## 💡 Popular Use Cases

### 🔍 **Semantic Search**
Find documents by meaning, not just keywords
```python
# Search millions of documents by semantic similarity
query = "machine learning applications"
results = find_similar_documents(query, document_vectors)
```

### 🤖 **RAG (Retrieval-Augmented Generation)**
Build AI assistants with accurate, up-to-date knowledge
```python
# Find relevant context for AI responses
context = retrieve_relevant_context(user_question, knowledge_base)
```

### 📊 **Content Classification**
Automatically categorize text at scale
```python
# Classify customer support tickets
category = classify_text(support_ticket, category_embeddings)
```

### 💬 **Recommendation Systems**
Suggest content based on semantic similarity
```python
# Recommend similar articles
recommendations = find_similar_content(current_article, all_articles)
```

## 🌟 What Makes Esperanto Special?

### **Universal Interface**
```python
# Same code, any provider - switch without changing your application
openai_model = AIFactory.create_embedding("openai", "text-embedding-3-small")
google_model = AIFactory.create_embedding("google", "text-embedding-004")
local_model = AIFactory.create_embedding("transformers", "all-MiniLM-L6-v2")

# Identical usage across all providers
for model in [openai_model, google_model, local_model]:
    embeddings = model.embed(["Hello world"])
```

### **Advanced Features Made Simple**
```python
# Task-aware embeddings - optimize for your specific use case
search_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": "retrieval.query"}  # Optimized for search queries
)

document_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3", 
    config={
        "task_type": "retrieval.document",  # Optimized for documents
        "late_chunking": True               # Better long-text handling
    }
)
```

### **Privacy Options**
```python
# Local processing - your data never leaves your machine
local_model = AIFactory.create_embedding("transformers", "all-MiniLM-L6-v2")
embeddings = local_model.embed(sensitive_documents)  # 100% private
```

## 🎯 Choose Your Path

| **I want to...** | **Start with** | **Provider** | **Why** |
|-------------------|----------------|--------------|---------|
| **Learn embeddings basics** | [Getting Started Guide](guide.md) | OpenAI | Reliable, well-documented |
| **Build production search** | [Advanced Features](advanced.md) | Jina | Task optimization, best performance |
| **Process sensitive data** | [Provider Reference](providers.md) | Transformers | Complete privacy, local processing |
| **Minimize costs** | [Provider Reference](providers.md) | Ollama/Transformers | Free after setup |
| **Enterprise deployment** | [Advanced Features](advanced.md) | Azure/Google | Enterprise compliance |

## 🛠 Supported Providers

- **🔥 OpenAI** - Reliable, high-quality embeddings
- **⚡ Jina** - Advanced task optimization (recommended for production)
- **🌐 Google** - Gemini models with native task support
- **🏠 Transformers** - Local processing, complete privacy
- **☁️ Azure** - Enterprise integration
- **📦 Ollama** - Local models, easy setup
- **🚀 Voyage** - Optimized for retrieval tasks
- **⭐ Mistral** - Multilingual excellence

## 🆘 Need Help?

- **Quick questions**: Check the [Getting Started Guide](guide.md)
- **Provider issues**: See [Provider Reference](providers.md)  
- **Performance optimization**: Read [Advanced Features](advanced.md)
- **Bugs or features**: [GitHub Issues](https://github.com/lfnovo/esperanto/issues)

---

**Ready to transform text into intelligence?** Start with the [Getting Started Guide](guide.md) and build something amazing! 🚀