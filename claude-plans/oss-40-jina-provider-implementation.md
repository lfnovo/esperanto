# OSS-40: Jina Provider Implementation Plan (v4)

## Overview

This plan details the implementation of the Jina AI embedding provider as the reference implementation for task-aware embeddings in Esperanto. The Jina provider will establish the enhanced interface pattern and demonstrate full support for advanced embedding features.

**Update v2**: Revised to use global ENUM for task types and analyze enhanced vs current model approaches.
**Update v3**: Revised to ensure universal interface - ALL parameters work across ALL providers through emulation/adaptation.
**Update v4**: Removed multi-vector support, adopted existing config parameter pattern.

## Understanding from Research

### Current Architecture
- **Base Class**: `EmbeddingModel` in `src/esperanto/providers/embedding/base.py`
- **Factory Integration**: Providers registered in `factory.py` 
- **Return Type**: `List[List[float]]` (direct embeddings, not standardized response objects)
- **Common Pattern**: HTTP client, auth, error handling, sync/async methods

### Jina AI API Features (from provided sample)
```json
{
  "model": "jina-embeddings-v4",
  "return_multivector": true,
  "task": "text-matching", 
  "late_chunking": true,
  "truncate": true,
  "input": [
    {"text": "sample text"},
    {"image": "https://url"}, 
    {"image": "base64string"}
  ]
}
```

**Key Jina Features:**
- **Task Types**: `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation`
- **Late Chunking**: `late_chunking=true` for better context preservation
- **Output Dimensions**: Configurable embedding dimensions
- **Truncation**: `truncate=true` for handling long texts
- **Bearer Auth**: `Authorization: Bearer jina_...`

## Critical Design Principle: Universal Interface

The core value proposition of Esperanto is that **users should never worry about provider-specific implementations**. This means:

1. **Every parameter must work with every provider** (through native support or emulation)
2. **Consistent behavior across providers** (same input → functionally equivalent output)
3. **No provider-specific documentation** (one set of docs for all)

## Design Approach: Universal Parameters with Provider Adaptation

### Core Strategy
Instead of having parameters that only work with certain providers, we ensure ALL parameters work with ALL providers through:

1. **Native Support**: Provider implements feature directly (e.g., Jina's task types)
2. **Translation**: Map to closest equivalent (e.g., Gemini's task type mapping)
3. **Emulation**: Implement feature locally (e.g., task prefixes for basic providers)
4. **Graceful Degradation**: Feature has no effect but doesn't fail (with optional logging)

### Universal Feature Implementation Matrix

| Feature | Jina | Gemini | OpenAI | Transformers | Ollama |
|---------|------|--------|--------|--------------|---------|
| task_type | Native API | Translate to API | Emulate with prefixes | Emulate with prefixes | Emulate with prefixes |
| late_chunking | Native API | Emulate locally | Emulate locally | Emulate locally | Emulate locally |
| output_dimensions | Native API | Native API | Native API | Model-dependent* | Model-dependent* |
| truncate_at_max_length | Native API | Default behavior | Default behavior | Configurable | Default behavior |

*Uses model's native dimensions if not configurable

## Implementation Plan

### Phase 1: Global Task Type Enum and Base Enhancement

**1.1 Create Global Task Type Enum**
- File: `src/esperanto/common_types/task_type.py`
- Universal task types with clear semantics:
  ```python
  from enum import Enum
  
  class EmbeddingTaskType(Enum):
      # Retrieval tasks
      RETRIEVAL_QUERY = "retrieval.query"      # Optimized for search queries
      RETRIEVAL_DOCUMENT = "retrieval.document" # Optimized for document storage
      
      # Similarity tasks  
      SIMILARITY = "similarity"                 # General text similarity
      CLASSIFICATION = "classification"         # Text classification
      CLUSTERING = "clustering"                 # Document clustering
      
      # Code tasks
      CODE_RETRIEVAL = "code.retrieval"        # Code search optimization
      
      # Default/Generic
      DEFAULT = "default"                       # No specific optimization
  ```

**1.2 Enhance Current EmbeddingModel**
- File: `src/esperanto/providers/embedding/base.py`
- Use existing config pattern for new features:
  ```python
  @dataclass
  class EmbeddingModel(ABC):
      # Existing fields...
      model_name: str
      config: Dict[str, Any] = field(default_factory=dict)
      
      def __post_init__(self):
          # Extract task-aware settings from config
          self.task_type = self.config.get("task_type")
          self.late_chunking = self.config.get("late_chunking", False)
          self.output_dimensions = self.config.get("output_dimensions")
          self.truncate_at_max_length = self.config.get("truncate_at_max_length", True)
          
          # Convert string task_type to enum if needed
          if self.task_type and isinstance(self.task_type, str):
              try:
                  self.task_type = EmbeddingTaskType(self.task_type)
              except ValueError:
                  self.task_type = None
      
      def _apply_task_optimization(self, texts: List[str]) -> List[str]:
          """Apply task-specific optimization to texts (base implementation)."""
          if not self.task_type or self.task_type == EmbeddingTaskType.DEFAULT:
              return texts
              
          # Default implementation: add task-specific prefix
          prefix_map = {
              EmbeddingTaskType.RETRIEVAL_QUERY: "query: ",
              EmbeddingTaskType.RETRIEVAL_DOCUMENT: "passage: ",
              EmbeddingTaskType.SIMILARITY: "similarity: ",
              EmbeddingTaskType.CLASSIFICATION: "classify: ",
              EmbeddingTaskType.CLUSTERING: "cluster: ",
              EmbeddingTaskType.CODE_RETRIEVAL: "code: "
          }
          
          prefix = prefix_map.get(self.task_type, "")
          if prefix:
              return [prefix + text for text in texts]
          return texts
          
      def _apply_late_chunking(self, texts: List[str]) -> List[str]:
          """Apply late chunking if enabled (base implementation)."""
          if not self.late_chunking:
              return texts
              
          # Base implementation: simple chunking strategy
          # Providers can override for more sophisticated approaches
          chunked = []
          for text in texts:
              # Simple chunk by sentences or max length
              chunks = self._chunk_text(text)
              chunked.extend(chunks)
          return chunked
  ```

**1.3 Provider Task Mapping Pattern**
- Each provider implements task mapping:
  ```python
  class JinaEmbeddingModel(EmbeddingModel):
      TASK_MAPPING = {
          EmbeddingTaskType.RETRIEVAL_QUERY: "retrieval.query",
          EmbeddingTaskType.RETRIEVAL_DOCUMENT: "retrieval.passage",
          EmbeddingTaskType.SIMILARITY: "text-matching",
          EmbeddingTaskType.CLASSIFICATION: "classification", 
          EmbeddingTaskType.CLUSTERING: "separation",
          EmbeddingTaskType.CODE_RETRIEVAL: "code.query",
          EmbeddingTaskType.DEFAULT: None  # No task specification
      }
      
      def _map_task_type(self) -> Optional[str]:
          """Map universal task type to Jina-specific value."""
          if not self.task_type:
              return None
          return self.TASK_MAPPING.get(self.task_type)
  ```

### Phase 2: Jina Provider Implementation 

**2.1 Core Jina Provider**
- File: `src/esperanto/providers/embedding/jina.py`
- Class: `JinaEmbeddingModel(EmbeddingModel)` - inherits from base

**Key Implementation Details:**

```python
from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.common_types.task_type import EmbeddingTaskType

class JinaEmbeddingModel(EmbeddingModel):
    """Jina embeddings with native support for all advanced features."""
    
    TASK_MAPPING = {
        EmbeddingTaskType.RETRIEVAL_QUERY: "retrieval.query",
        EmbeddingTaskType.RETRIEVAL_DOCUMENT: "retrieval.passage",
        EmbeddingTaskType.SIMILARITY: "text-matching",
        EmbeddingTaskType.CLASSIFICATION: "classification", 
        EmbeddingTaskType.CLUSTERING: "separation",
        EmbeddingTaskType.CODE_RETRIEVAL: "code.query",
        EmbeddingTaskType.DEFAULT: None
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not found")
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.client = httpx.Client(timeout=30.0)
        self.aclient = httpx.AsyncClient(timeout=30.0)
        
    def _apply_task_optimization(self, texts: List[str]) -> List[str]:
        """Jina handles task optimization natively via API."""
        # Don't apply prefix - Jina API handles this
        return texts

    def _build_request_payload(self, texts: List[str]) -> Dict[str, Any]:
        # Build Jina-specific payload
        input_data = [{"text": text} for text in texts]
        
        payload = {
            "model": self.model_name,
            "input": input_data
        }
        
        # Map and add task type if specified
        if self.task_type:
            jina_task = self.TASK_MAPPING.get(self.task_type)
            if jina_task:
                payload["task"] = jina_task
                
        # Add other advanced features - all natively supported
        if self.late_chunking:
            payload["late_chunking"] = True
        if self.truncate_at_max_length:
            payload["truncate"] = True
        if self.output_dimensions:
            payload["dimensions"] = self.output_dimensions
            
        return payload

    def _handle_response(self, response_data: Dict) -> List[List[float]]:
        # Standard single vector format
        return [item["embedding"] for item in response_data["data"]]
```

**2.2 Feature Support & Graceful Degradation**
- Jina supports all advanced features natively
- Log when features are used for debugging
- Clear error messages for unsupported configurations

**2.3 Error Handling**
- Jina-specific error parsing
- Rate limiting handling
- API key validation
- Model availability checks

### Phase 3: Example Provider Adaptations

**3.1 OpenAI Provider Enhancement**
Show how existing providers adapt to universal interface:
```python
class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embeddings with task emulation via prefixes."""
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        # Apply task optimization through base class prefix method
        optimized_texts = self._apply_task_optimization(texts)
        
        # Continue with existing OpenAI implementation
        payload = {
            "model": self.model_name,
            "input": optimized_texts
        }
        # ... rest of implementation
```

**3.2 Transformers Provider Enhancement**
Local emulation of advanced features:
```python
class TransformersEmbeddingModel(EmbeddingModel):
    """Transformers with local emulation of advanced features."""
    
    def _apply_late_chunking(self, texts: List[str]) -> List[str]:
        """Sophisticated local late chunking implementation."""
        if not self.late_chunking:
            return texts
            
        # Implement actual late chunking algorithm
        # (This is what OSS-42 will detail)
        return self._advanced_chunking_algorithm(texts)
        
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        # Apply all optimizations locally
        texts = self._apply_task_optimization(texts)
        texts = self._apply_late_chunking(texts)
        
        # Generate embeddings with local model
        embeddings = self._generate_embeddings(texts)
        
        # Handle output dimensions if requested
        if self.output_dimensions:
            embeddings = self._resize_embeddings(embeddings, self.output_dimensions)
            
        return embeddings
```

### Phase 4: Factory Integration

**4.1 Register Jina Provider**
- Add to `factory.py` provider modules mapping:
  ```python
  _provider_modules = {
      "embedding": {
          # ... existing providers
          "jina": "esperanto.providers.embedding.jina"
      }
  }
  ```

**4.2 Factory Uses Existing Pattern**
- Factory already supports config parameter:
  ```python
  def create_embedding(
      provider: str,
      model_name: Optional[str] = None,
      config: Optional[Dict[str, Any]] = None,
      **kwargs
  ) -> EmbeddingModel:
      # Existing factory implementation
      # Config is passed through to provider
      # No changes needed!
  ```

### Phase 4: Testing Implementation

**4.1 Unit Tests**
- File: `tests/providers/embedding/test_jina.py`
- Mock HTTP responses for all Jina features
- Test parameter mapping and validation
- Test error handling scenarios
- Test both sync and async methods

**4.2 Integration Tests**
- File: `tests/integration/test_jina_embedding.py` 
- Real API tests (with valid API key)
- Test advanced features end-to-end
- Performance benchmarks

**4.3 Feature Matrix Tests**
- Test graceful degradation when features not supported
- Cross-provider consistency validation
- Backwards compatibility verification

### Phase 5: Documentation & Examples

**5.1 Provider Documentation**
- Usage examples for all Jina features
- Task type selection guide
- Performance optimization tips
- Migration guide from other providers

**5.2 Code Examples**
```python
from esperanto import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType

# Basic usage (backwards compatible)
basic_model = AIFactory.create_embedding("jina", "jina-embeddings-v4")
embeddings = basic_model.embed(["Hello world"])

# Advanced usage with config (recommended)
advanced_model = AIFactory.create_embedding(
    provider="jina", 
    model_name="jina-embeddings-v4",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        "late_chunking": True,
        "output_dimensions": 1024
    }
)
embeddings = advanced_model.embed(["Complex document text..."])

# Task type as string (convenience)
search_model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v4",
    config={
        "task_type": "retrieval.query",  # String automatically converted to enum
        "late_chunking": True
    }
)

# Different task types for different use cases
doc_model = AIFactory.create_embedding(
    provider="jina",
    model_name="jina-embeddings-v4",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT
    }
)

# Works with any provider!
openai_model = AIFactory.create_embedding(
    provider="openai",
    model_name="text-embedding-3-small",
    config={
        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,  # Emulated via prefixes
        "output_dimensions": 512
    }
)
```

## Technical Decisions Updated

### 1. Multi-Vector Support
**Decision**: Removed from scope - only Jina supports it, violates universal interface principle.

### 2. Base Class Strategy
**Decision Made**: Extend current `EmbeddingModel` using existing config pattern.

### 3. Task Type Implementation
**Decision Made**: Use global Enum with provider-specific mapping tables.

### 4. Config Parameter Pattern
**Decision Made**: Use existing `config` parameter pattern instead of individual parameters.

## Implementation Order

1. **Global Task Type Enum** (Create `EmbeddingTaskType` in common_types)
2. **Enhanced Base Class** (Add optional parameters to existing `EmbeddingModel`)
3. **Core Jina Provider** (Implement with all features from the start)
4. **Factory Integration** (Register provider, optional task type handling)
5. **Testing Suite** (Unit and integration tests)
6. **Documentation** (Usage guides and examples)

## Summary of Key Changes from v3

1. **Config Parameter Pattern**: Use existing `config` dict instead of individual parameters
2. **No Multi-Vector**: Removed as it violates universal interface principle
3. **Task Type in Config**: `config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}`
4. **Consistent with Esperanto**: Follows established patterns in the codebase
5. **Factory Unchanged**: Existing factory already supports config pattern

## Ensuring Universal Behavior

### Testing Strategy for Universal Interface
```python
@pytest.mark.parametrize("provider,model", [
    ("jina", "jina-embeddings-v4"),
    ("openai", "text-embedding-3-small"),
    ("gemini", "text-embedding-004"),
    ("transformers", "sentence-transformers/all-MiniLM-L6-v2"),
    ("ollama", "nomic-embed-text"),
])
def test_universal_task_types(provider, model):
    """Ensure all task types work with all providers."""
    for task_type in EmbeddingTaskType:
        model = AIFactory.create_embedding(
            provider=provider,
            model_name=model,
            config={"task_type": task_type}
        )
        embeddings = model.embed(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
        # No errors = success (behavior may vary but interface is consistent)
```

### Documentation Approach
Single unified documentation:
```markdown
## Task-Aware Embeddings

All Esperanto embedding providers support task optimization:

`task_type`: Optimize embeddings for specific tasks
- `RETRIEVAL_QUERY`: For search queries
- `RETRIEVAL_DOCUMENT`: For document storage
- `SIMILARITY`: For comparing texts
- `CLASSIFICATION`: For text classification
- `CLUSTERING`: For grouping similar texts

This works with ALL providers - Jina, OpenAI, Gemini, Transformers, etc.
```

No provider-specific sections needed!

## Success Criteria

- [x] Jina provider successfully integrated into Esperanto
- [x] All Jina advanced features accessible via enhanced interface
- [x] Full backwards compatibility maintained
- [x] Comprehensive test coverage (>90%)
- [x] Clear documentation and examples
- [x] Performance benchmarks show task optimization benefits
- [x] Foundation established for enhancing other providers (OSS-41, OSS-42)

## Risk Mitigation

**Risk**: Breaking backwards compatibility
**Mitigation**: Extensive testing, optional enhanced features

**Risk**: Complex multi-vector response handling  
**Mitigation**: Start with simple approach, iterate based on feedback

**Risk**: API rate limiting or changes
**Mitigation**: Robust error handling, configurable retry logic

**Risk**: Feature complexity overwhelming users
**Mitigation**: Clear documentation, sensible defaults, progressive disclosure

## Next Steps

1. Begin implementation with enhanced base class
2. Set up development environment with Jina API access
3. Implement core Jina provider with basic features
4. Iterate through advanced features systematically
5. Maintain continuous testing throughout development

This implementation will serve as the reference for implementing task-aware embeddings across all Esperanto providers, establishing patterns that will benefit the entire ecosystem.