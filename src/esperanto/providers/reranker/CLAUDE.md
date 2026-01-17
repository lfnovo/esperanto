# Reranker Providers

Reranker provider implementations for relevance-based document ranking.

## Files

- **`base.py`**: Abstract base class `RerankerModel` defining the interface
- **`jina.py`**: Jina AI reranker models
- **`voyage.py`**: Voyage AI reranker models
- **`transformers.py`**: Local HuggingFace transformers reranker models

## Patterns

### Base Class Contract

All providers inherit from `RerankerModel` (base.py:15) and must:

1. **Implement abstract methods**:
   - `rerank()`: Synchronous reranking
   - `arerank()`: Async reranking
   - `_get_models()`: Return list of available models
   - `_get_default_model()`: Return default model name
   - `to_langchain()`: Convert to LangChain-compatible reranker
   - `provider` property: Return provider name string

2. **Override `__post_init__()`**:
   - Call `super().__post_init__()` first (initializes model_name if None)
   - Set `api_key` from parameter or environment variable (if API-based)
   - Set `base_url` (if API-based)
   - Call `self._create_http_clients()` last (for API providers)

3. **Return standardized response**:
   - Use `RerankResponse` from `esperanto.common_types.reranker`
   - Contains list of `RerankResult` objects with `index`, `document`, `relevance_score`

### Input Validation

Base class provides `_validate_inputs()` (base.py:171):

- Checks query is non-empty string
- Validates documents is non-empty list of strings
- Normalizes `top_k` to min(top_k, len(documents))
- Returns validated tuple: `(query, documents, top_k)`

Call this in your `rerank()` and `arerank()` implementations:

```python
def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None, **kwargs):
    query, documents, top_k = self._validate_inputs(query, documents, top_k)
    # ... proceed with reranking
```

### Score Normalization

Base class provides `_normalize_scores()` (base.py:209):

- Applies min-max normalization to 0-1 range
- Handles edge case where all scores are identical
- Use when provider returns scores outside 0-1 range

Most providers return relevance scores in 0-1 range natively, but some (like transformers cross-encoders) may return unbounded scores.

### HTTP Client Pattern

Same as other providers:

```python
def __post_init__(self):
    super().__post_init__()
    self.api_key = self.api_key or os.getenv("PROVIDER_API_KEY")
    self.base_url = self.base_url or "https://api.provider.com/v1"
    self._create_http_clients()
```

### Response Construction

Build `RerankResponse` from API results:

```python
from esperanto.common_types.reranker import RerankResponse, RerankResult

results = [
    RerankResult(
        index=idx,
        document=documents[idx],
        relevance_score=score
    )
    for idx, score in sorted_results[:top_k]
]

return RerankResponse(
    model=self.get_model_name(),
    results=results,
    usage={"tokens": tokens_used}  # if available
)
```

## Integration

- Imported by `factory.py` via `AIFactory._provider_modules["reranker"]`
- Uses types from `esperanto.common_types.reranker` (`RerankResponse`, `RerankResult`)
- Inherits mixins from `esperanto.utils.timeout` and `esperanto.utils.ssl`

## Gotchas

- **top_k handling**: If `top_k` is None, return ALL documents ranked (not just top N)
- **Index preservation**: `RerankResult.index` should be the original index in input documents list
- **Score ordering**: Results should be sorted by relevance_score descending (highest first)
- **Empty documents**: Call `_validate_inputs()` to catch empty lists early
- **Local vs API**: Transformers provider doesn't need API key, others do
- **Model name optional**: Unlike other provider types, reranker model_name is optional (has defaults)
- **Async for sync models**: Local transformers models don't have true async - use executor or run_in_executor
- **LangChain integration**: Some providers don't have native LangChain reranker classes - may need custom wrapper
- **Usage tracking**: Not all providers return token usage - set to None if unavailable
- **Deprecation warnings**: Use `_get_models()` internally (not `.models` property)

## When Adding a New Provider

1. Create new file `provider_name.py`
2. Import `RerankerModel` from `esperanto.providers.reranker.base`
3. Import `RerankResponse`, `RerankResult` from `esperanto.common_types.reranker`
4. Define class inheriting from `RerankerModel`
5. Implement all abstract methods
6. Add `__post_init__()` following the pattern
7. Use `_validate_inputs()` for input validation
8. Use `_normalize_scores()` if needed
9. Add provider to `factory.py` in `_provider_modules["reranker"]` dict
10. Write tests in `tests/providers/reranker/test_provider_name.py`
11. Add documentation in `docs/` if public provider

## Special Cases

### Transformers Provider

- Uses HuggingFace cross-encoder models locally
- No API key needed
- Downloads models to cache on first use
- Scores may be unbounded - use `_normalize_scores()`
- Model names are HuggingFace model IDs (e.g., "cross-encoder/ms-marco-MiniLM-L-12-v2")
- Can run on GPU if available

### Jina and Voyage

- Both are API-based services
- Return scores in 0-1 range (no normalization needed)
- Support batch processing (multiple queries at once in some cases)
- Have usage/billing limits

## Common Implementation Patterns

### Sorting and Slicing

Always sort by score descending before slicing:

```python
# Get scores from API
scores = api_response["scores"]

# Create index-score pairs
indexed_scores = [(idx, score) for idx, score in enumerate(scores)]

# Sort by score descending
sorted_results = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

# Take top_k
top_results = sorted_results[:top_k]

# Build response
results = [
    RerankResult(index=idx, document=documents[idx], relevance_score=score)
    for idx, score in top_results
]
```

### Error Handling

Catch provider-specific errors and convert to standard exceptions:

```python
try:
    response = self.client.post(url, json=payload)
    response.raise_for_status()
except httpx.HTTPStatusError as e:
    raise RuntimeError(f"Reranking failed: {e.response.text}")
except Exception as e:
    raise RuntimeError(f"Reranking error: {str(e)}")
```
