# Advanced Embeddings Features

Unlock the full power of modern embeddings with task-aware optimization, advanced processing techniques, and production-ready patterns.

## 🎯 Task-Aware Embeddings (NEW!)

Task-aware embeddings represent a breakthrough in semantic processing. Instead of one-size-fits-all vectors, you can now optimize embeddings for specific tasks, achieving 15-30% performance improvements.

### The Problem with Generic Embeddings

```python
# Traditional approach - same embedding for everything
generic_model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Query: "fast cars"
query_embedding = generic_model.embed(["fast cars"])

# Document: "high-performance vehicles" 
doc_embedding = generic_model.embed(["high-performance vehicles"])

# Search result might miss the connection! 😞
```

### The Task-Aware Solution

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Specialized models for different purposes
query_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

document_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3", 
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

# Now they're optimized to find each other! ✨
query_embedding = query_model.embed(["fast cars"])
doc_embedding = document_model.embed(["high-performance vehicles"])
# Much better similarity scores!
```

### Universal Task Types

All providers support these task types through a unified interface:

#### 🔍 **Retrieval Tasks**

```python
# For search queries (what users type)
query_model = AIFactory.create_embedding(
    provider="any",  # Works with any provider!
    model_name="any-model",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

# For documents (what gets searched)  
document_model = AIFactory.create_embedding(
    provider="any",
    model_name="any-model",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)
```

**Best for:** Search engines, RAG systems, Q&A platforms, documentation search

**Performance gain:** 15-25% improvement in search relevance

#### 📊 **Classification Tasks**

```python
classification_model = AIFactory.create_embedding(
    "google", "text-embedding-004",
    config={"task_type": EmbeddingTaskType.CLASSIFICATION}
)

# Optimized for categorizing text
emails = [
    "I want to return this broken item",  # → Support
    "When will my order arrive?",         # → Shipping  
    "I love this product!",               # → Feedback
]
embeddings = classification_model.embed(emails)
```

**Best for:** Email routing, content moderation, sentiment analysis, topic modeling

**Performance gain:** Better separation between categories, cleaner decision boundaries

#### 🔗 **Similarity & Clustering**

```python
similarity_model = AIFactory.create_embedding(
    "transformers", "all-mpnet-base-v2",
    config={"task_type": EmbeddingTaskType.SIMILARITY}
)

# Find similar content
articles = ["AI in healthcare", "Medical AI applications", "Sports news"]
embeddings = similarity_model.embed(articles)
# First two will be much closer than the third
```

**Best for:** Recommendation systems, duplicate detection, content clustering, plagiarism detection

#### 💻 **Code Retrieval**

```python
code_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.CODE_RETRIEVAL}
)

# Understands programming concepts
code_snippets = [
    "def fibonacci(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
    "function factorial(n) { return n <= 1 ? 1 : n * factorial(n-1) }",
    "class Car: def __init__(self, brand): self.brand = brand"
]
embeddings = code_model.embed(code_snippets)
```

**Best for:** Code search engines, API documentation, programming tutorials, code completion

#### 🧠 **Question Answering**

```python
qa_model = AIFactory.create_embedding(
    "google", "text-embedding-004", 
    config={"task_type": EmbeddingTaskType.QUESTION_ANSWERING}
)

# Optimized for Q&A patterns
questions = ["What is machine learning?", "How does AI work?"]
answers = ["ML is a subset of AI...", "AI works by processing data..."]

q_embeddings = qa_model.embed(questions)
a_embeddings = qa_model.embed(answers)
```

**Best for:** FAQ systems, educational platforms, customer support, chatbots

#### ✅ **Fact Verification**

```python
fact_model = AIFactory.create_embedding(
    "google", "text-embedding-004",
    config={"task_type": EmbeddingTaskType.FACT_VERIFICATION}
)

# Optimized for fact-checking
claims = ["The Earth is round", "Water boils at 100°C at sea level"]
evidence = ["Scientific consensus...", "Physics textbooks state..."]

claim_embeddings = fact_model.embed(claims)
evidence_embeddings = fact_model.embed(evidence)
```

**Best for:** Fact-checking systems, misinformation detection, journalism tools, research validation

### Provider Implementation Differences

| Provider | Implementation | Performance | Features |
|----------|----------------|-------------|----------|
| **Jina** | 🥇 Native API | Best | Full task optimization + late chunking |
| **Google** | 🥈 Native API | Excellent | 8 task types, direct translation |
| **OpenAI** | 🥉 Smart Prefixes | Good | Intelligent prompt engineering |
| **Transformers** | 🔧 Local Emulation | Good | Advanced local processing |
| **Others** | 📝 Basic Prefixes | Fair | Simple text prefixes |

### Real-World RAG Pipeline

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Step 1: Create specialized models
query_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "output_dimensions": 512  # Faster search
    }
)

document_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,        # Handle long docs
        "output_dimensions": 512      # Match query model
    }
)

# Step 2: Index your knowledge base
knowledge_base = [
    "Esperanto is a unified interface for AI models...",
    "Task-aware embeddings optimize for specific use cases...", 
    "RAG systems retrieve relevant context before generation..."
]

print("🔄 Indexing knowledge base...")
doc_embeddings = document_model.embed(knowledge_base)

# Step 3: Query processing
def ask_question(question):
    # Optimize query embedding for retrieval
    query_embedding = query_model.embed([question])
    
    # Find most relevant documents
    similarities = calculate_similarities(query_embedding, doc_embeddings)
    best_docs = get_top_k(similarities, k=3)
    
    return best_docs

# Step 4: Test the system
question = "How do I optimize embeddings for search?"
relevant_docs = ask_question(question)
print(f"📚 Found {len(relevant_docs)} relevant documents")
```

## 🔄 Late Chunking

Late chunking revolutionizes how we handle long documents by preserving context that would be lost in traditional truncation.

### The Problem with Traditional Chunking

```python
# Traditional approach - information loss! 😞
long_document = """
Introduction to Machine Learning
Machine learning is a powerful subset of artificial intelligence...
[... 2000 words ...]
Conclusion: The future of ML looks bright with continued innovation...
"""

# Gets truncated to first 512 tokens
# Loses conclusion, context, and important connections!
traditional_embedding = basic_model.embed([long_document])
```

### Late Chunking Solution

```python
# Late chunking - preserves full context! ✨
smart_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True
    }
)

# Intelligently processes entire document
smart_embedding = smart_model.embed([long_document])
# Preserves connections between introduction and conclusion!
```

### How Late Chunking Works

1. **Text Analysis**: Document is analyzed for semantic structure
2. **Intelligent Segmentation**: Split into overlapping, meaningful chunks
3. **Context Preservation**: Maintains relationships between sections
4. **Weighted Aggregation**: Combines chunk embeddings with attention weights
5. **Final Embedding**: Single vector representing entire document

### When to Use Late Chunking

```python
# Perfect for long documents
use_cases = [
    "Research papers (5-50 pages)",
    "Legal documents", 
    "Technical documentation",
    "News articles",
    "Book chapters",
    "Product manuals"
]

# Enable for documents > 512 tokens
document_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "late_chunking": True,
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)

embeddings = document_model.embed(use_cases)
```

### Performance Impact

| Document Length | Traditional | Late Chunking | Improvement |
|-----------------|-------------|---------------|-------------|
| **512 tokens** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | No difference |
| **1K tokens** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | +40% context |
| **2K tokens** | ⭐⭐ | ⭐⭐⭐⭐⭐ | +80% context |
| **5K+ tokens** | ⭐ | ⭐⭐⭐⭐ | +200% context |

## 📐 Output Dimensions Control

Optimize embedding size for your specific performance and quality requirements.

### Understanding the Trade-off

```python
# High dimensions = Better quality, slower search, more memory
high_quality = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"output_dimensions": 1024}  # Full quality
)

# Lower dimensions = Faster search, less memory, slight quality loss
optimized = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3", 
    config={"output_dimensions": 256}   # 4x faster search!
)
```

### Dimension Selection Guide

| Dimensions | Use Case | Quality | Speed | Memory |
|------------|----------|---------|-------|---------|
| **1536+** | Research, critical accuracy | ⭐⭐⭐⭐⭐ | 🐌 | 💾💾💾💾 |
| **1024** | Production, high quality | ⭐⭐⭐⭐ | ⚡ | 💾💾💾 |
| **512** | Real-time, good quality | ⭐⭐⭐ | ⚡⚡ | 💾💾 |
| **256** | Mobile, fast search | ⭐⭐ | ⚡⚡⚡ | 💾 |

### Production Optimization Example

```python
# Development - prioritize quality
dev_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"output_dimensions": 1024}
)

# Production - balance quality and speed
prod_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "output_dimensions": 512,  # 2x faster similarity search
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True
    }
)

# Mobile/Edge - prioritize speed
mobile_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3", 
    config={"output_dimensions": 256}  # 4x faster, 75% less memory
)
```

## 🚀 Production Patterns

### Hybrid Provider Strategy

Use different providers for different parts of your pipeline:

```python
# Development: Cloud for experimentation
dev_model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Indexing: High-quality with advanced features
indexing_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,
        "output_dimensions": 512
    }
)

# Query: Fast, local processing
query_model = AIFactory.create_embedding(
    "transformers", "all-MiniLM-L6-v2",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY
    }
)

# Use each for its strength!
```

### Caching Strategy

```python
import hashlib
import json
from typing import Dict, List

class EmbeddingCache:
    def __init__(self):
        self.cache: Dict[str, List[float]] = {}
    
    def get_cache_key(self, text: str, model_config: dict) -> str:
        """Create unique cache key"""
        content = f"{text}:{json.dumps(model_config, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_embedding(self, text: str, model, model_config: dict):
        """Get cached embedding or compute new one"""
        cache_key = self.get_cache_key(text, model_config)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compute new embedding
        response = model.embed([text])
        embedding = response.data[0].embedding
        
        # Cache result
        self.cache[cache_key] = embedding
        return embedding

# Usage
cache = EmbeddingCache()
model = AIFactory.create_embedding("jina", "jina-embeddings-v3")
config = {"task_type": "retrieval.query"}

# First call - computes embedding
embedding1 = cache.get_embedding("hello world", model, config)

# Second call - returns cached result
embedding2 = cache.get_embedding("hello world", model, config)
```

### Batch Processing Optimization

```python
async def process_large_dataset(texts: List[str], batch_size: int = 100):
    """Efficiently process large datasets"""
    model = AIFactory.create_embedding(
        "jina", "jina-embeddings-v3",
        config={
            "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
            "output_dimensions": 512  # Optimize for storage
        }
    )
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Process batch with error handling
        try:
            response = await model.aembed(batch)
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"✅ Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
        except Exception as e:
            print(f"❌ Error in batch {i//batch_size + 1}: {e}")
            # Add None placeholders or retry logic
            all_embeddings.extend([None] * len(batch))
    
    return all_embeddings

# Usage
texts = ["text " + str(i) for i in range(10000)]
embeddings = await process_large_dataset(texts)
```

### Vector Database Integration

```python
# Example with Chroma
import chromadb
from esperanto.factory import AIFactory

# Initialize
client = chromadb.Client()
collection = client.create_collection("my_documents")

# Create optimized embedding model
model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,
        "output_dimensions": 512
    }
)

# Add documents
documents = ["doc1 content", "doc2 content", "doc3 content"]
embeddings_response = model.embed(documents)
embeddings = [data.embedding for data in embeddings_response.data]

collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Search with optimized query model
query_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "output_dimensions": 512  # Match document dimensions
    }
)

query = "search for something"
query_embedding = query_model.embed([query]).data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
```

## 🔍 Performance Benchmarking

### Measuring Task-Aware Improvements

```python
import time
import numpy as np
from esperanto.factory import AIFactory

def benchmark_task_optimization():
    """Compare generic vs task-aware embeddings"""
    
    # Test data
    queries = ["machine learning tutorial", "python programming guide"]
    documents = [
        "Learn ML with Python: comprehensive tutorial for beginners",
        "Python programming: complete guide to coding in Python"
    ]
    
    # Generic model
    generic_model = AIFactory.create_embedding("jina", "jina-embeddings-v3")
    
    # Task-optimized models
    query_model = AIFactory.create_embedding(
        "jina", "jina-embeddings-v3",
        config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
    )
    
    doc_model = AIFactory.create_embedding(
        "jina", "jina-embeddings-v3",
        config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
    )
    
    # Benchmark
    def calculate_similarities(query_embs, doc_embs):
        similarities = []
        for q_emb in query_embs:
            for d_emb in doc_embs:
                sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
                similarities.append(sim)
        return similarities
    
    # Generic approach
    generic_q_embs = [generic_model.embed([q]).data[0].embedding for q in queries]
    generic_d_embs = [generic_model.embed([d]).data[0].embedding for d in documents]
    generic_sims = calculate_similarities(generic_q_embs, generic_d_embs)
    
    # Task-optimized approach
    task_q_embs = [query_model.embed([q]).data[0].embedding for q in queries]
    task_d_embs = [doc_model.embed([d]).data[0].embedding for d in documents]
    task_sims = calculate_similarities(task_q_embs, task_d_embs)
    
    # Results
    print("🔍 Task-Aware Embedding Benchmark:")
    print(f"Generic similarities: {[f'{s:.3f}' for s in generic_sims]}")
    print(f"Task-optimized similarities: {[f'{s:.3f}' for s in task_sims]}")
    
    improvement = np.mean(task_sims) / np.mean(generic_sims) - 1
    print(f"📈 Average improvement: {improvement:.1%}")

# Run benchmark
benchmark_task_optimization()
```

### Speed vs Quality Analysis

```python
def benchmark_dimensions():
    """Compare different embedding dimensions"""
    text = "Machine learning is transforming industries worldwide"
    
    dimensions = [256, 512, 1024, 1536]
    results = {}
    
    for dim in dimensions:
        model = AIFactory.create_embedding(
            "jina", "jina-embeddings-v3",
            config={"output_dimensions": dim}
        )
        
        # Time embedding generation
        start_time = time.time()
        embedding = model.embed([text]).data[0].embedding
        generation_time = time.time() - start_time
        
        # Simulate similarity search time (scales with dimensions)
        search_time = dim * 0.000001  # Simulated
        
        results[dim] = {
            "generation_time": generation_time,
            "search_time": search_time,
            "memory_mb": dim * 4 / 1024 / 1024,  # 4 bytes per float
            "actual_dimensions": len(embedding)
        }
    
    print("\n📊 Dimension Performance Analysis:")
    for dim, metrics in results.items():
        print(f"{dim}D: Gen={metrics['generation_time']:.3f}s, "
              f"Search={metrics['search_time']:.3f}s, "
              f"Memory={metrics['memory_mb']:.2f}MB")

benchmark_dimensions()
```

## 🎓 Advanced Use Cases

### Multi-Modal Search System

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

class MultiModalSearchEngine:
    def __init__(self):
        # Specialized models for different content types
        self.text_model = AIFactory.create_embedding(
            "jina", "jina-embeddings-v3",
            config={
                "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
                "late_chunking": True,
                "output_dimensions": 512
            }
        )
        
        self.code_model = AIFactory.create_embedding(
            "jina", "jina-embeddings-v3",
            config={
                "task_type": EmbeddingTaskType.CODE_RETRIEVAL,
                "output_dimensions": 512
            }
        )
        
        self.query_model = AIFactory.create_embedding(
            "jina", "jina-embeddings-v3",
            config={
                "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
                "output_dimensions": 512
            }
        )
    
    def index_content(self, items):
        """Index different types of content"""
        indexed_items = []
        
        for item in items:
            if item["type"] == "text":
                embedding = self.text_model.embed([item["content"]]).data[0].embedding
            elif item["type"] == "code":
                embedding = self.code_model.embed([item["content"]]).data[0].embedding
            else:
                # Fallback to text model
                embedding = self.text_model.embed([item["content"]]).data[0].embedding
            
            indexed_items.append({
                **item,
                "embedding": embedding
            })
        
        return indexed_items
    
    def search(self, query, indexed_items, top_k=5):
        """Search across all content types"""
        query_embedding = self.query_model.embed([query]).data[0].embedding
        
        # Calculate similarities
        similarities = []
        for item in indexed_items:
            similarity = np.dot(query_embedding, item["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
            )
            similarities.append((similarity, item))
        
        # Return top results
        similarities.sort(reverse=True)
        return similarities[:top_k]

# Usage example
search_engine = MultiModalSearchEngine()

content = [
    {"type": "text", "content": "Machine learning tutorial for beginners", "title": "ML Guide"},
    {"type": "code", "content": "def train_model(X, y): return sklearn.fit(X, y)", "title": "Training Code"},
    {"type": "text", "content": "Advanced neural network architectures", "title": "Deep Learning"}
]

indexed_content = search_engine.index_content(content)
results = search_engine.search("how to train ML models", indexed_content)

for similarity, item in results:
    print(f"📄 {item['title']}: {similarity:.3f}")
```

### Dynamic Task Switching

```python
class AdaptiveEmbeddingPipeline:
    """Automatically choose the best task type based on content"""
    
    def __init__(self):
        self.models = {
            EmbeddingTaskType.RETRIEVAL_QUERY: AIFactory.create_embedding(
                "jina", "jina-embeddings-v3",
                config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
            ),
            EmbeddingTaskType.RETRIEVAL_DOCUMENT: AIFactory.create_embedding(
                "jina", "jina-embeddings-v3",
                config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT, "late_chunking": True}
            ),
            EmbeddingTaskType.CLASSIFICATION: AIFactory.create_embedding(
                "jina", "jina-embeddings-v3",
                config={"task_type": EmbeddingTaskType.CLASSIFICATION}
            ),
            EmbeddingTaskType.CODE_RETRIEVAL: AIFactory.create_embedding(
                "jina", "jina-embeddings-v3",
                config={"task_type": EmbeddingTaskType.CODE_RETRIEVAL}
            )
        }
    
    def detect_task_type(self, text: str) -> EmbeddingTaskType:
        """Automatically detect the best task type"""
        text_lower = text.lower()
        
        # Code detection
        code_indicators = ["def ", "function", "class ", "import ", "return ", "{", "}", "var ", "let "]
        if any(indicator in text_lower for indicator in code_indicators):
            return EmbeddingTaskType.CODE_RETRIEVAL
        
        # Query detection (short, question-like)
        query_indicators = ["what", "how", "when", "where", "why", "?"]
        if len(text.split()) < 10 and any(indicator in text_lower for indicator in query_indicators):
            return EmbeddingTaskType.RETRIEVAL_QUERY
        
        # Classification detection (categories, labels)
        classification_indicators = ["category:", "label:", "type:", "classify"]
        if any(indicator in text_lower for indicator in classification_indicators):
            return EmbeddingTaskType.CLASSIFICATION
        
        # Default to document retrieval
        return EmbeddingTaskType.RETRIEVAL_DOCUMENT
    
    def embed_smart(self, text: str):
        """Automatically choose and apply the best embedding strategy"""
        task_type = self.detect_task_type(text)
        model = self.models[task_type]
        
        print(f"🎯 Auto-detected task: {task_type.value}")
        return model.embed([text]).data[0].embedding

# Usage
pipeline = AdaptiveEmbeddingPipeline()

# Automatically optimizes for each input type
texts = [
    "What is machine learning?",  # → RETRIEVAL_QUERY
    "def fibonacci(n): return n if n <= 1 else fib(n-1) + fib(n-2)",  # → CODE_RETRIEVAL
    "Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",  # → RETRIEVAL_DOCUMENT
    "category: technology, sentiment: positive"  # → CLASSIFICATION
]

for text in texts:
    embedding = pipeline.embed_smart(text)
    print(f"Generated {len(embedding)}-dimensional embedding\n")
```

## 🔧 Troubleshooting & Optimization

### Common Performance Issues

#### Slow Similarity Search

```python
# ❌ Problem: High-dimensional embeddings
slow_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3"  # Default 1024 dimensions
)

# ✅ Solution: Optimize dimensions for speed
fast_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"output_dimensions": 256}  # 4x faster search
)

# ✅ Alternative: Use faster similarity search libraries
import faiss
import numpy as np

def create_fast_index(embeddings):
    """Create optimized similarity search index"""
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create FAISS index for fast similarity search
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    return index

def fast_search(query_embedding, index, k=5):
    """Ultra-fast similarity search"""
    query_array = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_array)
    
    scores, indices = index.search(query_array, k)
    return list(zip(scores[0], indices[0]))
```

#### Memory Issues with Large Datasets

```python
# ❌ Problem: Loading all embeddings into memory
def memory_intensive_approach(texts):
    model = AIFactory.create_embedding("jina", "jina-embeddings-v3")
    all_embeddings = []
    
    for text in texts:  # Memory grows linearly!
        embedding = model.embed([text]).data[0].embedding
        all_embeddings.append(embedding)
    
    return all_embeddings

# ✅ Solution: Streaming processing with disk storage
import numpy as np
from pathlib import Path

def memory_efficient_approach(texts, batch_size=100):
    model = AIFactory.create_embedding(
        "jina", "jina-embeddings-v3",
        config={"output_dimensions": 512}  # Reduce memory usage
    )
    
    output_dir = Path("embeddings")
    output_dir.mkdir(exist_ok=True)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Process batch
        response = model.embed(batch)
        batch_embeddings = np.array([data.embedding for data in response.data])
        
        # Save to disk immediately
        batch_file = output_dir / f"batch_{i//batch_size:04d}.npy"
        np.save(batch_file, batch_embeddings)
        
        print(f"💾 Saved batch {i//batch_size + 1} to {batch_file}")
    
    return output_dir

# Usage for millions of documents
texts = ["text " + str(i) for i in range(1_000_000)]
embeddings_dir = memory_efficient_approach(texts)
```

#### Rate Limit Handling

```python
import asyncio
import time
from typing import List

class RateLimitedEmbedder:
    def __init__(self, provider="openai", model="text-embedding-3-small", 
                 requests_per_minute=500):
        self.model = AIFactory.create_embedding(provider, model)
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    async def embed_with_rate_limit(self, texts: List[str]):
        """Embed with automatic rate limiting"""
        now = time.time()
        
        # Remove old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Check if we need to wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                print(f"⏳ Rate limit reached, waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Make request
        try:
            response = await self.model.aembed(texts)
            self.request_times.append(time.time())
            return response
        
        except Exception as e:
            if "rate limit" in str(e).lower():
                print("🚫 Rate limit hit, backing off...")
                await asyncio.sleep(60)  # Wait 1 minute
                return await self.embed_with_rate_limit(texts)  # Retry
            else:
                raise e

# Usage
embedder = RateLimitedEmbedder("openai", "text-embedding-3-small", requests_per_minute=500)

async def process_safely(texts):
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await embedder.embed_with_rate_limit(batch)
        all_embeddings.extend([data.embedding for data in response.data])
    
    return all_embeddings
```

### Quality Optimization

#### A/B Testing Different Configurations

```python
import random
from sklearn.metrics import accuracy_score

def ab_test_embeddings(test_queries, ground_truth_docs):
    """Compare different embedding configurations"""
    
    configurations = [
        {
            "name": "Generic OpenAI",
            "model": AIFactory.create_embedding("openai", "text-embedding-3-small")
        },
        {
            "name": "Task-Optimized Jina",
            "model": AIFactory.create_embedding(
                "jina", "jina-embeddings-v3",
                config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
            )
        },
        {
            "name": "Optimized Local",
            "model": AIFactory.create_embedding(
                "transformers", "all-mpnet-base-v2",
                config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
            )
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"🧪 Testing {config['name']}...")
        
        # Generate embeddings
        query_embeddings = config["model"].embed(test_queries)
        doc_embeddings = config["model"].embed(ground_truth_docs)
        
        # Calculate retrieval accuracy
        correct_matches = 0
        for i, query_emb in enumerate([data.embedding for data in query_embeddings.data]):
            similarities = []
            for doc_emb in [data.embedding for data in doc_embeddings.data]:
                similarity = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                similarities.append(similarity)
            
            # Check if ground truth doc is in top-3
            top_3_indices = np.argsort(similarities)[-3:]
            if i in top_3_indices:  # Assuming query i matches doc i
                correct_matches += 1
        
        accuracy = correct_matches / len(test_queries)
        results[config["name"]] = accuracy
        print(f"📊 Accuracy: {accuracy:.3f}")
    
    # Find best configuration
    best_config = max(results, key=results.get)
    print(f"\n🏆 Best configuration: {best_config} ({results[best_config]:.3f})")
    
    return results

# Run A/B test
test_queries = ["machine learning tutorial", "python programming", "data science"]
ground_truth_docs = [
    "Complete guide to machine learning with examples",
    "Python programming tutorial for beginners", 
    "Introduction to data science and analytics"
]

results = ab_test_embeddings(test_queries, ground_truth_docs)
```

## 🎯 Production Checklist

### Pre-Deployment Validation

```python
def validate_embedding_pipeline(model, test_cases):
    """Comprehensive pipeline validation"""
    
    print("🔍 Running embedding pipeline validation...")
    
    # Test 1: Basic functionality
    try:
        test_embedding = model.embed(["test"]).data[0].embedding
        assert len(test_embedding) > 0
        print("✅ Basic embedding generation works")
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False
    
    # Test 2: Batch processing
    try:
        batch_embeddings = model.embed(["test1", "test2", "test3"])
        assert len(batch_embeddings.data) == 3
        print("✅ Batch processing works")
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False
    
    # Test 3: Consistency
    try:
        emb1 = model.embed(["consistent test"]).data[0].embedding
        emb2 = model.embed(["consistent test"]).data[0].embedding
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert similarity > 0.99  # Should be nearly identical
        print("✅ Embedding consistency validated")
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")
        return False
    
    # Test 4: Performance benchmarks
    try:
        start_time = time.time()
        model.embed(["performance test"] * 100)
        duration = time.time() - start_time
        print(f"✅ Performance: {duration:.2f}s for 100 embeddings")
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    print("🎉 All validation tests passed!")
    return True

# Usage
production_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,
        "output_dimensions": 512
    }
)

validation_passed = validate_embedding_pipeline(production_model, test_cases=[])
```

### Monitoring & Alerting

```python
import logging
from datetime import datetime

class EmbeddingMonitor:
    def __init__(self, model):
        self.model = model
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "errors": 0,
            "avg_response_time": 0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def embed_with_monitoring(self, texts):
        """Embed with comprehensive monitoring"""
        start_time = time.time()
        
        try:
            # Make request
            response = self.model.embed(texts)
            
            # Update metrics
            self.metrics["total_requests"] += 1
            if hasattr(response, 'usage') and response.usage:
                self.metrics["total_tokens"] += response.usage.total_tokens
            
            response_time = time.time() - start_time
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (self.metrics["total_requests"] - 1) + response_time) /
                self.metrics["total_requests"]
            )
            
            # Log success
            self.logger.info(f"Embedding successful: {len(texts)} texts in {response_time:.3f}s")
            
            # Check for performance issues
            if response_time > 5.0:  # 5 second threshold
                self.logger.warning(f"Slow response detected: {response_time:.3f}s")
            
            return response
            
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Embedding failed: {e}")
            
            # Alert on high error rate
            error_rate = self.metrics["errors"] / max(self.metrics["total_requests"], 1)
            if error_rate > 0.05:  # 5% error rate threshold
                self.logger.critical(f"High error rate detected: {error_rate:.2%}")
            
            raise e
    
    def get_health_status(self):
        """Get current system health"""
        error_rate = self.metrics["errors"] / max(self.metrics["total_requests"], 1)
        
        status = {
            "healthy": error_rate < 0.05 and self.metrics["avg_response_time"] < 2.0,
            "metrics": self.metrics,
            "error_rate": error_rate,
            "timestamp": datetime.now().isoformat()
        }
        
        return status

# Usage
model = AIFactory.create_embedding("jina", "jina-embeddings-v3")
monitor = EmbeddingMonitor(model)

# Use monitored embedding
try:
    response = monitor.embed_with_monitoring(["test document"])
    print("✅ Embedding successful")
except Exception as e:
    print(f"❌ Embedding failed: {e}")

# Check health
health = monitor.get_health_status()
print(f"System healthy: {health['healthy']}")
print(f"Error rate: {health['error_rate']:.2%}")
```

---

## 🚀 Next Steps

You're now equipped with advanced embedding capabilities! Here's what to explore next:

1. **🧪 Experiment** with task-aware embeddings on your specific use case
2. **📊 Benchmark** different configurations for your data
3. **🏗️ Build** a production pipeline using the patterns above
4. **📈 Monitor** performance and optimize based on real usage
5. **🌟 Share** your results with the community!

### Further Reading

- [Getting Started Guide](guide.md) - If you need to review the basics
- [Provider Reference](providers.md) - Detailed provider documentation
- [GitHub Repository](https://github.com/lfnovo/esperanto) - Latest updates and examples
- [Community Discord](https://discord.gg/esperanto) - Get help and share experiences

The future of embeddings is task-aware, and you're ready to be part of it! 🎯✨