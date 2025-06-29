# Embeddings Guide: From Zero to Production

This guide will take you from "what are embeddings?" to building production-ready systems. No prior knowledge required.

## 📚 What Are Embeddings?

Think of embeddings as a **universal translator for text**. They convert human language into numbers that computers can understand and compare.

### The Magic of Semantic Understanding

```python
# Traditional keyword search fails here:
query = "car"
documents = ["automobile repair", "vehicle maintenance", "driving tips"]
# No keyword matches! 😞

# But embeddings understand meaning:
from esperanto.factory import AIFactory

model = AIFactory.create_embedding("openai", "text-embedding-3-small")

query_vector = model.embed(["car"])
doc_vectors = model.embed(documents)

# Now we can find semantic matches! ✨
# "automobile" and "vehicle" are similar to "car"
```

### Real-World Impact

**Before embeddings:**
- Search: "red shoes" only finds documents with those exact words
- Recommendations: Based on simple keyword matching
- Classification: Manual rule creation

**With embeddings:**
- Search: "red shoes" finds "crimson sneakers", "scarlet footwear", etc.
- Recommendations: Understand content meaning, not just surface words
- Classification: Automatic understanding of text categories

## 🎯 When to Use Embeddings

### ✅ Perfect For:

#### **🔍 Semantic Search**
*Find content by meaning, not keywords*

```python
# User searches for "fast cars"
# Finds: "speedy vehicles", "quick automobiles", "rapid sports cars"
search_results = semantic_search("fast cars", car_database)
```

**Why it works:** Embeddings understand that "fast" = "speedy" = "quick"

#### **🤖 RAG (Retrieval-Augmented Generation)**
*Give AI assistants accurate, current knowledge*

```python
# User asks: "What's our refund policy?"
# System finds relevant policy documents
# AI generates accurate answer using that context
context = find_relevant_docs(user_question, company_docs)
answer = ai_model.generate(user_question, context)
```

**Why it works:** Embeddings find the most relevant context for any question

#### **📊 Content Classification**
*Automatically categorize text*

```python
# Classify customer emails
email = "I want to return this broken item"
category = classify_email(email)  # → "Returns"
```

**Why it works:** Embeddings learn patterns in text categories

#### **💡 Recommendation Systems**
*Suggest similar content*

```python
# User reads article about "Python programming"
# System recommends "JavaScript tutorials", "Coding best practices"
recommendations = find_similar_articles(current_article, all_articles)
```

**Why it works:** Embeddings understand content similarity

### ❌ Not Ideal For:

- **Exact keyword matching** (use traditional search)
- **Grammar/spelling checking** (use specialized tools)
- **Very short text** (< 3 words, limited semantic info)
- **Real-time chat** (embeddings add latency)

## 🚀 Your First Embedding System

Let's build a simple but powerful document search system:

### Step 1: Basic Setup

```python
from esperanto.factory import AIFactory
import numpy as np

# Create embedding model
model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Sample documents (imagine these are your company docs)
documents = [
    "Our return policy allows returns within 30 days of purchase",
    "Technical support is available Monday through Friday, 9 AM to 5 PM",
    "Shipping takes 3-5 business days for standard delivery",
    "Premium members get free shipping on all orders",
    "We offer installation services for enterprise customers"
]

print("✅ Setup complete!")
```

### Step 2: Create Document Index

```python
# Convert all documents to embeddings (vectors)
print("🔄 Creating document embeddings...")
doc_response = model.embed(documents)
doc_embeddings = [data.embedding for data in doc_response.data]

print(f"✅ Created {len(doc_embeddings)} document embeddings")
print(f"📏 Each embedding has {len(doc_embeddings[0])} dimensions")
```

### Step 3: Search Function

```python
def semantic_search(query, documents, doc_embeddings, top_k=3):
    """Find the most relevant documents for a query"""
    
    # Convert query to embedding
    query_response = model.embed([query])
    query_embedding = query_response.data[0].embedding
    
    # Calculate similarity with all documents
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    similarities = [
        cosine_similarity(query_embedding, doc_emb) 
        for doc_emb in doc_embeddings
    ]
    
    # Get top results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx],
            'rank': len(results) + 1
        })
    
    return results

print("🔍 Search function ready!")
```

### Step 4: Test Your System

```python
# Test different queries
test_queries = [
    "How do I return an item?",
    "When can I get help?", 
    "How long does delivery take?",
    "Do I get free shipping?"
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    results = semantic_search(query, documents, doc_embeddings, top_k=2)
    
    for result in results:
        print(f"  📄 Rank {result['rank']}: {result['document']}")
        print(f"  📊 Similarity: {result['similarity']:.3f}")
```

**Expected Results:**
- "How do I return an item?" → finds return policy doc
- "When can I get help?" → finds technical support doc
- Etc.

The magic: No exact keyword matches needed! ✨

## 🎨 Choosing Task Types (Advanced)

Esperanto's task-aware embeddings can optimize for your specific use case:

### 🔍 Search & Retrieval

```python
# For search queries (what users type)
query_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": "retrieval.query"}
)

# For documents (what gets searched)
document_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3", 
    config={"task_type": "retrieval.document"}
)
```

**When to use:** Building search engines, RAG systems, Q&A platforms

**Why it matters:** 15-20% improvement in search relevance

### 📊 Classification

```python
classification_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": "classification"}
)
```

**When to use:** Categorizing emails, content moderation, sentiment analysis

**Why it matters:** Better separation between categories

### 🔗 Similarity & Clustering

```python
similarity_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": "similarity"}
)
```

**When to use:** Recommendation systems, duplicate detection, content clustering

**Why it matters:** More accurate similarity scores

### 💻 Code Search

```python
code_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": "code.retrieval"}
)
```

**When to use:** Code search engines, documentation systems, API discovery

**Why it matters:** Understands programming concepts and syntax

## 🏠 Local vs Cloud: When to Use What

### ☁️ Cloud Providers (OpenAI, Google, Jina)

**✅ Choose when:**
- Getting started (easiest setup)
- Need latest/best models
- Variable usage patterns
- Want someone else to handle infrastructure

**💰 Cost:** Pay per embedding (usually $0.0001-0.002 per 1K tokens)

```python
# Cloud example - no setup required
model = AIFactory.create_embedding("openai", "text-embedding-3-small")
embeddings = model.embed(texts)  # Instant results
```

### 🏠 Local Providers (Transformers, Ollama)

**✅ Choose when:**
- Processing sensitive data
- High volume/predictable usage
- Want complete control
- Cost optimization for large scale

**💰 Cost:** Hardware + electricity (often cheaper at scale)

```python
# Local example - runs on your machine
model = AIFactory.create_embedding("transformers", "all-MiniLM-L6-v2")
embeddings = model.embed(sensitive_docs)  # 100% private
```

### 🔄 When to Use Both

```python
# Development: Use cloud for experimentation
dev_model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Production: Switch to local for cost/privacy
prod_model = AIFactory.create_embedding("transformers", "all-MiniLM-L6-v2")

# Same interface, different backends!
embeddings = dev_model.embed(texts)  # Development
embeddings = prod_model.embed(texts)  # Production
```

## 🔧 Understanding Model Types (Transformers)

When using local models, choose based on your needs:

### 🎯 Sentence Transformers (Recommended)
*Pre-trained for semantic similarity*

```python
# Best general choice
model = AIFactory.create_embedding("transformers", "all-MiniLM-L6-v2")
```

**Use for:** Search, recommendations, RAG, general semantic tasks
**Why:** Trained specifically for creating good sentence-level embeddings

### 🧠 BERT Models
*Good for classification and understanding*

```python
# For classification tasks
model = AIFactory.create_embedding(
    "transformers", "bert-base-uncased",
    config={"pooling_strategy": "cls"}  # Use [CLS] token for classification
)
```

**Use for:** Text classification, sentiment analysis, named entity recognition
**Why:** BERT's [CLS] token is designed for classification tasks

### 🌍 Multilingual Models

```python
# For non-English text
model = AIFactory.create_embedding(
    "transformers", 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

**Use for:** Multi-language applications, international content
**Why:** Trained on multiple languages, understands cross-language similarity

### 📏 Choosing Model Size

| Model Size | Speed | Quality | Memory | Use Case |
|------------|-------|---------|--------|----------|
| **Mini** (22MB) | ⚡⚡⚡ | ⭐⭐ | 💾 | Development, testing |
| **Small** (80MB) | ⚡⚡ | ⭐⭐⭐ | 💾💾 | Production, real-time |
| **Base** (420MB) | ⚡ | ⭐⭐⭐⭐ | 💾💾💾 | High-quality results |
| **Large** (1.2GB) | 🐌 | ⭐⭐⭐⭐⭐ | 💾💾💾💾 | Best quality, batch processing |

## 🚀 Advanced Features Explained

### 🔄 Late Chunking
*Better handling of long documents*

```python
model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": "retrieval.document",
        "late_chunking": True  # Magic for long texts
    }
)
```

**What it does:** Instead of truncating long text, it processes in chunks then combines intelligently

**When to use:** Documents > 512 tokens, articles, research papers, documentation

**Why it matters:** Preserves context that would be lost in simple truncation

### 📐 Output Dimensions
*Control embedding size*

```python
model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"output_dimensions": 512}  # Smaller = faster + less memory
)
```

**Trade-offs:**
- **Smaller (256-512):** Faster similarity search, less memory, slightly lower quality
- **Larger (1024-1536):** Better quality, more memory, slower search

**When to optimize:** High-volume production systems, memory-constrained environments

## 📈 Production Considerations

### 🏃‍♂️ Performance Tips

```python
# Batch processing for efficiency
texts = ["doc1", "doc2", "doc3", ...]
embeddings = model.embed(texts)  # Much faster than one-by-one

# Async for I/O-bound workloads
embeddings = await model.aembed(texts)

# Use appropriate dimensions
model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"output_dimensions": 512}  # Balance quality vs speed
)
```

### 💾 Storage & Indexing

```python
# Store embeddings efficiently
import numpy as np

# Save embeddings
embeddings_array = np.array([data.embedding for data in response.data])
np.save('document_embeddings.npy', embeddings_array)

# For production: Use vector databases
# - Pinecone (cloud)
# - Weaviate (self-hosted)
# - Chroma (lightweight)
# - FAISS (Facebook's library)
```

### 🔄 Caching Strategy

```python
# Cache embeddings to avoid recomputation
import hashlib
import json

def get_cached_embedding(text, model):
    # Create cache key
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache first
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    # Generate and cache
    embedding = model.embed([text]).data[0].embedding
    embedding_cache[cache_key] = embedding
    return embedding
```

## 🎓 Next Steps

Now that you understand the fundamentals:

1. **🔧 Try the examples** in this guide with your own data
2. **📖 Read [Provider Reference](providers.md)** to choose the best provider for your needs
3. **⚡ Explore [Advanced Features](advanced.md)** for production optimization
4. **🛠 Build something awesome** and share it with the community!

## ❓ Common Questions

**Q: Which provider should I start with?**
A: OpenAI for learning, Jina for production, Transformers for privacy.

**Q: How many dimensions do I need?**
A: Start with default, optimize later. 512 is usually enough.

**Q: Can I use different providers together?**
A: Yes! Use cloud for development, local for production, or specialize by task.

**Q: What about costs?**
A: Cloud: ~$0.10-2.00 per million tokens. Local: hardware cost only.

**Q: How do I handle rate limits?**
A: Use async processing, implement exponential backoff, or switch to local models.

---

Ready to build something amazing with embeddings? The world of semantic AI awaits! 🚀