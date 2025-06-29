# Provider Reference

Complete technical reference for all embedding providers in Esperanto. Choose the right provider for your specific needs.

## 🚀 Quick Provider Selection

| **Need** | **Recommended Provider** | **Why** |
|----------|-------------------------|---------|
| **Getting started** | OpenAI | Reliable, well-documented, great quality |
| **Best performance** | Jina | Task optimization, advanced features, cutting-edge |
| **Complete privacy** | Transformers | Local processing, no data leaves your machine |
| **Enterprise compliance** | Azure | Enterprise security, integrated billing |
| **Cost optimization** | Ollama | Free after setup, good for high volume |
| **Multilingual content** | Google/Mistral | Excellent multilingual support |
| **Code search** | Jina/Voyage | Optimized for code understanding |

## 📊 Feature Comparison Matrix

| Provider | Task Types | Late Chunking | Output Dims | Privacy | Cost | Setup |
|----------|------------|---------------|-------------|---------|------|-------|
| **OpenAI** | ✅ Emulated | ✅ Emulated | ✅ Native | ☁️ Cloud | 💰💰 | ⚡ Instant |
| **Jina** | ✅ Native | ✅ Native | ✅ Native | ☁️ Cloud | 💰💰 | ⚡ Instant |
| **Google** | ✅ Native | ✅ Emulated | ❌ No | ☁️ Cloud | 💰💰 | ⚡ Instant |
| **Transformers** | ✅ Emulated | ✅ Enhanced | ❌ Model | 🏠 Local | 💰 Hardware | 🔧 Setup |
| **Azure** | ✅ Emulated | ✅ Emulated | ✅ Native | ☁️ Cloud | 💰💰 | 🔧 Config |
| **Ollama** | ✅ Emulated | ✅ Emulated | ❌ Model | 🏠 Local | 💰 Hardware | 🔧 Install |
| **Voyage** | ✅ Emulated | ❌ No | ❌ No | ☁️ Cloud | 💰💰 | ⚡ Instant |
| **Mistral** | ✅ Emulated | ❌ No | ❌ No | ☁️ Cloud | 💰💰 | ⚡ Instant |

## 🔥 OpenAI Provider

**Best for:** Getting started, reliable production workloads, proven quality

### Quick Start

```python
from esperanto.factory import AIFactory

# Basic usage
model = AIFactory.create_embedding("openai", "text-embedding-3-small")
embeddings = model.embed(["Hello world"])

# Advanced configuration
model = AIFactory.create_embedding(
    "openai", 
    "text-embedding-3-large",
    config={
        "task_type": "retrieval.query",  # Task optimization via prefixes
        "output_dimensions": 1024        # Reduce from default 3072
    }
)
```

### Environment Setup

```bash
export OPENAI_API_KEY="your-api-key"
```

### Available Models

| Model | Dimensions | Cost (per 1M tokens) | Best For |
|-------|------------|---------------------|----------|
| **text-embedding-3-small** | 1536 | $0.02 | General use, cost-effective |
| **text-embedding-3-large** | 3072 | $0.13 | Highest quality, complex tasks |
| **text-embedding-ada-002** | 1536 | $0.10 | Legacy, compatibility |

### Features

- ✅ **Custom Dimensions**: Reduce embedding size for performance
- ✅ **Task Optimization**: Via intelligent text prefixes
- ✅ **Rate Limits**: Generous limits for production use
- ✅ **Reliability**: 99.9% uptime SLA
- ❌ **Native Task Types**: Uses emulation instead
- ❌ **Late Chunking**: Basic emulation only

### When to Choose OpenAI

**✅ Perfect for:**
- First-time embedding users
- Proven production reliability needed
- Integration with other OpenAI services
- Budget allows for premium quality

**❌ Consider alternatives if:**
- Need cutting-edge task optimization
- Processing highly sensitive data
- Very high volume (cost optimization)
- Need specialized features like late chunking

---

## ⚡ Jina Provider

**Best for:** Production systems requiring maximum performance and advanced features

### Quick Start

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Basic usage
model = AIFactory.create_embedding("jina", "jina-embeddings-v3")

# Advanced features (recommended)
model = AIFactory.create_embedding(
    "jina", 
    "jina-embeddings-v3",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        "late_chunking": True,           # Better long-text handling
        "output_dimensions": 512,        # Optimize for speed
        "truncate_at_max_length": True   # Graceful text handling
    }
)
```

### Environment Setup

```bash
export JINA_API_KEY="your-jina-api-key"
```

### Available Models

| Model | Dimensions | Context | Best For |
|-------|------------|---------|----------|
| **jina-embeddings-v3** | 1024 | 8192 | Production, multilingual |
| **jina-embeddings-v4** | 1024 | 8192 | Latest, multimodal support |
| **jina-clip-v2** | 512 | N/A | Text + image embeddings |

### Advanced Features

#### Task Type Optimization (Native API)

```python
# Search pipeline optimization
query_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
)

document_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3", 
    config={"task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT}
)
```

**Task Type Mappings:**
- `RETRIEVAL_QUERY` → `"retrieval.query"`
- `RETRIEVAL_DOCUMENT` → `"retrieval.passage"`
- `CLASSIFICATION` → `"classification"`
- `CLUSTERING` → `"separation"`
- `SIMILARITY` → `"text-matching"`
- `CODE_RETRIEVAL` → `"code.query"`

#### Late Chunking (Native API)

```python
# Perfect for long documents
model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={
        "late_chunking": True,
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)

# Handles documents > 8K tokens intelligently
long_doc = "Very long document content..." * 1000
embeddings = model.embed([long_doc])  # No information loss
```

### When to Choose Jina

**✅ Perfect for:**
- Production RAG systems
- Advanced search engines
- Performance-critical applications
- Long document processing
- Multilingual content (100+ languages)

**❌ Consider alternatives if:**
- Simple prototype/learning project
- Budget is primary concern
- Don't need advanced features

---

## 🌐 Google Provider

**Best for:** Enterprise integration, multilingual content, native task optimization

### Quick Start

```python
from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Basic usage
model = AIFactory.create_embedding("google", "text-embedding-004")

# With task optimization
model = AIFactory.create_embedding(
    "google", 
    "text-embedding-004",
    config={
        "task_type": EmbeddingTaskType.FACT_VERIFICATION  # Google-specific
    }
)
```

### Environment Setup

```bash
export GOOGLE_API_KEY="your-google-api-key"
# or
export GEMINI_API_KEY="your-google-api-key"
```

### Available Models

| Model | Dimensions | Best For |
|-------|------------|----------|
| **text-embedding-004** | 768 | Latest, highest quality |
| **embedding-001** | 768 | General purpose, stable |

### Native Task Types

Google has **native API support** for task optimization:

```python
# These map directly to Gemini API parameters
EmbeddingTaskType.RETRIEVAL_QUERY → "RETRIEVAL_QUERY"
EmbeddingTaskType.RETRIEVAL_DOCUMENT → "RETRIEVAL_DOCUMENT"  
EmbeddingTaskType.SIMILARITY → "SEMANTIC_SIMILARITY"
EmbeddingTaskType.CLASSIFICATION → "CLASSIFICATION"
EmbeddingTaskType.CLUSTERING → "CLUSTERING"
EmbeddingTaskType.CODE_RETRIEVAL → "CODE_RETRIEVAL_QUERY"
EmbeddingTaskType.QUESTION_ANSWERING → "QUESTION_ANSWERING"
EmbeddingTaskType.FACT_VERIFICATION → "FACT_VERIFICATION"
```

### Specialized Use Cases

```python
# Fact-checking pipeline
fact_model = AIFactory.create_embedding(
    "google", "text-embedding-004",
    config={"task_type": EmbeddingTaskType.FACT_VERIFICATION}
)

# Q&A optimization
qa_model = AIFactory.create_embedding(
    "google", "text-embedding-004", 
    config={"task_type": EmbeddingTaskType.QUESTION_ANSWERING}
)
```

### When to Choose Google

**✅ Perfect for:**
- Google Cloud ecosystem integration
- Fact-checking and verification systems
- Q&A applications
- Need native task optimization
- Multilingual applications

**❌ Consider alternatives if:**
- Need output dimension control
- Require late chunking
- Want the absolute best performance

---

## 🏠 Transformers Provider (Local)

**Best for:** Privacy-first applications, cost optimization, full control

### Setup

```bash
# Install transformers extra
pip install "esperanto[transformers]"
```

### Quick Start

```python
from esperanto.factory import AIFactory

# Basic usage (downloads model automatically)
model = AIFactory.create_embedding(
    "transformers", 
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Advanced configuration
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/all-mpnet-base-v2",
    config={
        "device": "auto",                    # GPU if available
        "pooling_strategy": "mean",          # Pooling method
        "quantize": "8bit",                  # Memory optimization
        "tokenizer_config": {
            "max_length": 512,
            "padding": True,
            "truncation": True
        }
    }
)
```

### Model Categories

#### 🎯 Sentence Transformers (Recommended)

```python
# Best general choice
model = AIFactory.create_embedding(
    "transformers", 
    "sentence-transformers/all-MiniLM-L6-v2"  # 22MB, fast, good quality
)

# High quality
model = AIFactory.create_embedding(
    "transformers", 
    "sentence-transformers/all-mpnet-base-v2"  # 420MB, best quality
)

# Multilingual
model = AIFactory.create_embedding(
    "transformers",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

**Use for:** Search, recommendations, RAG, general semantic tasks

#### 🧠 BERT Models

```python
# For classification
model = AIFactory.create_embedding(
    "transformers", 
    "bert-base-uncased",
    config={"pooling_strategy": "cls"}  # Use [CLS] token
)
```

**Use for:** Text classification, sentiment analysis, NER

#### 📊 Specialized Models

```python
# Code understanding
model = AIFactory.create_embedding(
    "transformers",
    "microsoft/codebert-base"
)

# Scientific text
model = AIFactory.create_embedding(
    "transformers", 
    "allenai/scibert_scivocab_uncased"
)
```

### Advanced Local Features

#### Enhanced Task Optimization

```python
# Local emulation with sophisticated preprocessing
model = AIFactory.create_embedding(
    "transformers",
    "all-MiniLM-L6-v2",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "late_chunking": True  # Intelligent local chunking algorithm
    }
)
```

#### GPU Optimization

```python
# CUDA
model = AIFactory.create_embedding(
    "transformers", "all-mpnet-base-v2",
    config={
        "device": "cuda",
        "quantize": "8bit"  # Use less GPU memory
    }
)

# Apple Silicon
model = AIFactory.create_embedding(
    "transformers", "all-mpnet-base-v2", 
    config={"device": "mps"}
)
```

#### Pooling Strategies

```python
# Different ways to extract embeddings
mean_model = AIFactory.create_embedding(
    "transformers", "bert-base-uncased",
    config={"pooling_strategy": "mean"}  # Average all tokens (default)
)

cls_model = AIFactory.create_embedding(
    "transformers", "bert-base-uncased",
    config={"pooling_strategy": "cls"}   # Use [CLS] token (classification)
)

max_model = AIFactory.create_embedding(
    "transformers", "bert-base-uncased", 
    config={"pooling_strategy": "max"}   # Max pooling (key features)
)
```

### Model Size Guide

| Size | Example Model | Memory | Speed | Quality | Use Case |
|------|---------------|---------|-------|---------|----------|
| **Tiny** | all-MiniLM-L6-v2 (22MB) | 💾 | ⚡⚡⚡ | ⭐⭐ | Development, real-time |
| **Small** | all-MiniLM-L12-v2 (80MB) | 💾💾 | ⚡⚡ | ⭐⭐⭐ | Production, mobile |
| **Base** | all-mpnet-base-v2 (420MB) | 💾💾💾 | ⚡ | ⭐⭐⭐⭐ | High quality |
| **Large** | bge-large-en-v1.5 (1.2GB) | 💾💾💾💾 | 🐌 | ⭐⭐⭐⭐⭐ | Best quality, batch |

### When to Choose Transformers

**✅ Perfect for:**
- Processing sensitive/confidential data
- High-volume, predictable workloads
- Air-gapped environments
- Cost optimization at scale
- Complete control over model versions

**❌ Consider alternatives if:**
- Want zero setup
- Need latest model capabilities
- Have unpredictable usage patterns
- Limited local compute resources

---

## ☁️ Azure Provider

**Best for:** Enterprise environments, integrated Microsoft ecosystem

### Quick Start

```python
from esperanto.factory import AIFactory

# Using environment variables
model = AIFactory.create_embedding("azure", "your-deployment-name")

# Explicit configuration  
model = AIFactory.create_embedding(
    "azure",
    "your-deployment-name",
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com",
    api_version="2024-12-01-preview"
)
```

### Environment Setup

```bash
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com" 
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

### Available Models

Same as OpenAI but through Azure deployment:
- text-embedding-3-small
- text-embedding-3-large  
- text-embedding-ada-002

### Enterprise Features

- ✅ **Private Networks**: VNet integration
- ✅ **Compliance**: SOC 2, HIPAA, etc.
- ✅ **Integrated Billing**: Azure subscription
- ✅ **Regional Control**: Data residency
- ✅ **Enterprise Support**: SLA guarantees

### When to Choose Azure

**✅ Perfect for:**
- Microsoft ecosystem integration
- Enterprise compliance requirements
- Regional data residency needs
- Existing Azure infrastructure

**❌ Consider alternatives if:**
- Simple projects/prototypes
- Want latest OpenAI features immediately
- No Azure infrastructure

---

## 📦 Ollama Provider (Local)

**Best for:** Local development, free usage, easy local setup

### Setup

```bash
# Install Ollama first
curl -fsSL https://ollama.ai/install.sh | sh

# Pull an embedding model
ollama pull nomic-embed-text
```

### Quick Start

```python
from esperanto.factory import AIFactory

# Basic usage
model = AIFactory.create_embedding("ollama", "nomic-embed-text")
embeddings = model.embed(["Hello world"])
```

### Available Models

```bash
# Popular embedding models
ollama pull nomic-embed-text     # 274MB, good quality
ollama pull mxbai-embed-large    # 669MB, high quality
ollama pull snowflake-arctic-embed  # 669MB, enterprise-grade
```

### When to Choose Ollama

**✅ Perfect for:**
- Learning and experimentation
- Local development environment
- Zero ongoing costs
- Simple local setup

**❌ Consider alternatives if:**
- Need production reliability
- Want advanced features
- Require enterprise support

---

## 🚀 Voyage Provider

**Best for:** Retrieval-optimized tasks, research applications

### Quick Start

```python
from esperanto.factory import AIFactory

model = AIFactory.create_embedding("voyage", "voyage-3")
embeddings = model.embed(["Hello world"])
```

### Environment Setup

```bash
export VOYAGE_API_KEY="your-voyage-key"
```

### Available Models

- **voyage-3**: Latest, general purpose
- **voyage-code-2**: Optimized for code
- **voyage-law-2**: Legal domain specialization

### When to Choose Voyage

**✅ Perfect for:**
- Research and academic use
- Specialized domain applications
- Code search systems

---

## ⭐ Mistral Provider

**Best for:** European data residency, multilingual content

### Quick Start

```python
from esperanto.factory import AIFactory

model = AIFactory.create_embedding("mistral", "mistral-embed")
embeddings = model.embed(["Hello world"])
```

### Environment Setup

```bash
export MISTRAL_API_KEY="your-mistral-key"
```

### When to Choose Mistral

**✅ Perfect for:**
- European data residency requirements
- Multilingual applications
- French language content

---

## 🔄 Switching Between Providers

One of Esperanto's key strengths is seamless provider switching:

```python
# Development with cloud
dev_model = AIFactory.create_embedding("openai", "text-embedding-3-small")
dev_embeddings = dev_model.embed(texts)

# Production with local (same interface!)
prod_model = AIFactory.create_embedding("transformers", "all-MiniLM-L6-v2")
prod_embeddings = prod_model.embed(texts)

# Advanced production with Jina
advanced_model = AIFactory.create_embedding(
    "jina", "jina-embeddings-v3",
    config={"task_type": "retrieval.document", "late_chunking": True}
)
advanced_embeddings = advanced_model.embed(texts)

# All embeddings have the same response structure!
```

## 🎯 Decision Framework

### Start Here: Quick Decision Tree

1. **Learning embeddings?** → OpenAI
2. **Processing sensitive data?** → Transformers
3. **Need maximum performance?** → Jina
4. **Enterprise environment?** → Azure/Google
5. **High volume/cost optimization?** → Transformers/Ollama
6. **Specialized domain?** → Voyage/Mistral

### Advanced Considerations

**Performance Requirements:**
- **Real-time (< 100ms)**: Local models (Transformers, Ollama)
- **Batch processing**: Any provider, optimize for cost
- **Highest quality**: Jina with task optimization

**Data Sensitivity:**
- **Public data**: Any cloud provider
- **Internal data**: Consider data residency laws
- **Highly sensitive**: Local only (Transformers, Ollama)

**Scale & Cost:**
- **< 1M embeddings/month**: Any provider works
- **1M-100M/month**: Compare cloud pricing vs local setup
- **> 100M/month**: Local almost always cheaper

**Technical Expertise:**
- **Beginner**: OpenAI → Jina → Others
- **Intermediate**: Start with best fit for use case
- **Expert**: Mix providers for optimal cost/performance

---

Ready to choose your provider? Start with the [Getting Started Guide](guide.md) to understand your requirements, then come back here for the perfect provider match! 🚀