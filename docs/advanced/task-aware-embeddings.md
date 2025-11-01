# Task-Aware Embeddings

## Overview

Task-aware embeddings represent a breakthrough in semantic processing. Instead of one-size-fits-all vectors, you can optimize embeddings for specific tasks, achieving 15-30% performance improvements in search relevance, classification accuracy, and similarity matching.

This advanced feature is available across all embedding providers in Esperanto, with varying levels of native support and emulation.

## The Problem with Generic Embeddings

Traditional embedding approaches use the same model configuration for all use cases:

```python
# Traditional approach - same embedding for everything
generic_model = AIFactory.create_embedding("openai", "text-embedding-3-small")

# Query: "fast cars"
query_embedding = generic_model.embed(["fast cars"])

# Document: "high-performance vehicles"
doc_embedding = generic_model.embed(["high-performance vehicles"])

# Search result might miss the connection! 😞
```

The issue is that queries and documents have different semantic patterns. Queries are typically short and intent-focused, while documents are longer and information-rich. Using the same embedding strategy for both results in suboptimal matching.

## The Task-Aware Solution

Task-aware embeddings optimize the model for specific purposes:

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

## Universal Task Types

All providers support these task types through a unified interface:

### Retrieval Tasks

Optimize for search and information retrieval:

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

### Classification Tasks

Optimize for categorizing and labeling text:

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

### Similarity & Clustering

Optimize for finding similar content:

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

### Code Retrieval

Optimize for programming-related content:

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

### Question Answering

Optimize for Q&A patterns:

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

### Fact Verification

Optimize for fact-checking and verification:

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

## Provider Implementation Differences

Different providers implement task-aware embeddings in different ways:

| Provider | Implementation | Performance | Features |
|----------|----------------|-------------|----------|
| **Jina** | Native API | Best | Full task optimization + late chunking |
| **Google** | Native API | Excellent | 8 task types, direct translation |
| **OpenAI** | Smart Prefixes | Good | Intelligent prompt engineering |
| **Transformers** | Local Emulation | Good | Advanced local processing |
| **Others** | Basic Prefixes | Fair | Simple text prefixes |

### Native Implementation

Providers like Jina and Google have native API support for task types. The task type is passed directly to their API:

```python
# Jina native task support
model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)
# Task type sent directly to Jina API
```

### Prefix-Based Implementation

Providers without native support use intelligent prefixes:

```python
# OpenAI uses smart prefixes
model = AIFactory.create_embedding(
    "openai", "text-embedding-3-small",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)
# Text is prefixed with: "Represent this query for retrieving relevant documents: "
```

## Real-World RAG Pipeline

Here's a complete example of using task-aware embeddings in a RAG system:

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

## Use Cases

### Multi-Task Search Engine

Different content types benefit from different task optimizations:

```python
class MultiTaskSearchEngine:
    def __init__(self):
        # Text documents
        self.text_model = AIFactory.create_embedding(
            "jina", "jina-embeddings-v3",
            config={
                "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
                "output_dimensions": 512
            }
        )

        # Code repositories
        self.code_model = AIFactory.create_embedding(
            "jina", "jina-embeddings-v3",
            config={
                "task_type": EmbeddingTaskType.CODE_RETRIEVAL,
                "output_dimensions": 512
            }
        )

        # User queries
        self.query_model = AIFactory.create_embedding(
            "jina", "jina-embeddings-v3",
            config={
                "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
                "output_dimensions": 512
            }
        )

    def index_content(self, items):
        for item in items:
            if item["type"] == "text":
                embedding = self.text_model.embed([item["content"]]).data[0].embedding
            elif item["type"] == "code":
                embedding = self.code_model.embed([item["content"]]).data[0].embedding

            item["embedding"] = embedding

        return items
```

### Email Classification

Optimize for categorization tasks:

```python
# Create classification-optimized model
classifier = AIFactory.create_embedding(
    "google", "text-embedding-004",
    config={"task_type": EmbeddingTaskType.CLASSIFICATION}
)

# Pre-compute category embeddings
categories = {
    "support": "Customer support and technical issues",
    "sales": "Sales inquiries and product information",
    "billing": "Billing, invoices, and payment questions"
}

category_embeddings = {
    name: classifier.embed([desc]).data[0].embedding
    for name, desc in categories.items()
}

# Classify incoming emails
def classify_email(email_text):
    email_emb = classifier.embed([email_text]).data[0].embedding

    # Find closest category
    best_category = max(
        category_embeddings.items(),
        key=lambda x: cosine_similarity(email_emb, x[1])
    )

    return best_category[0]
```

## Best Practices

### Match Task Types for Queries and Documents

When building search systems, use matching task types:

```python
# ✅ Good - Matched task types
query_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

doc_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)

# ❌ Bad - Mismatched task types
query_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.CLASSIFICATION}
)

doc_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)
```

### Choose the Right Provider

Select providers based on task type support:

- **Jina**: Best for retrieval tasks with advanced features
- **Google**: Best for diverse task types (8 supported)
- **OpenAI**: Good for general-purpose with prefix optimization
- **Transformers**: Best for local/privacy-sensitive applications

### Combine with Other Features

Task-aware embeddings work well with other advanced features:

```python
# Combine task awareness with dimension control and late chunking
model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,           # For long documents
        "output_dimensions": 512          # For faster search
    }
)
```

## Performance Benchmarking

Measure the impact of task-aware embeddings:

```python
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

    # Calculate similarities
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

## See Also

- [Embedding Capabilities Guide](/docs/capabilities/embeddings.md) - Overview of embedding features
- [Jina Provider](/docs/providers/jina.md) - Native task-aware support
- [Google Provider](/docs/providers/google.md) - Native task-aware support
- [Transformers Features](/docs/advanced/transformers-features.md) - Local task-aware implementation
- [Model Discovery](/docs/advanced/model-discovery.md) - Finding compatible models
