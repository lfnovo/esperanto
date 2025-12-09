# Performance Metrics Implementation Plan

**Date:** 2025-12-08
**Status:** Complete (All Phases)
**Goal:** Capture TTFT and tokens/second metrics tied to tier and model

## Overview

Implement performance timing metrics (TTFT, tokens/second) in brio-esperanto, the unified abstraction layer for LLM calls. Metrics will be:
- Returned in API responses for frontend display
- Logged locally (JSONL) with rolling window
- Designed for future upstream analytics

## Data Sources

### llama.cpp Built-in Timings
```json
{
  "timings": {
    "prompt_per_second": 884.93,
    "predicted_per_token_ms": 43.09,
    "predicted_per_second": 23.21
  }
}
```

### Measured at Request Time
- **TTFT (streaming):** Time from request sent to first chunk received
- **Total time:** Full request duration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  brio-ext (tier selection, metrics logging)                 │
│    ├── Select tier/model                                    │
│    ├── Call esperanto provider                              │
│    ├── Log metrics (JSONL)                                  │
│    └── Return response with timings to frontend             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  esperanto (unified LLM abstraction)                        │
│    ├── LlamaCppLanguageModel                                │
│    │     ├── Extract timings from llama.cpp response        │
│    │     └── Measure TTFT for streaming                     │
│    └── ChatCompletion response includes Timings             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  llama.cpp server                                           │
│    └── Returns usage + timings in response                  │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Extend Response Types
**File:** `src/esperanto/common_types/response.py`

Add `Timings` model to capture performance metrics:

```python
class Timings(BaseModel):
    """Performance timing metrics from LLM inference."""
    ttft_ms: Optional[float] = None              # Time to first token (ms)
    tokens_per_second: Optional[float] = None    # Generation speed (predicted_per_second)
    prompt_tokens_per_second: Optional[float] = None  # Prompt processing speed
    total_time_ms: Optional[float] = None        # Total request time

class ChatCompletion(BaseModel):
    # ... existing fields ...
    timings: Optional[Timings] = None  # NEW
```

**Effort:** ~20 lines

---

### Phase 2: Extract Timings in LlamaCpp Provider
**File:** `src/brio_ext/providers/llamacpp_provider.py`

#### Non-streaming
Parse `timings` from llama.cpp response JSON:

```python
timings_data = response_json.get("timings", {})
timings = Timings(
    tokens_per_second=timings_data.get("predicted_per_second"),
    prompt_tokens_per_second=timings_data.get("prompt_per_second"),
    total_time_ms=timings_data.get("total_time_ms"),
)
```

#### Streaming
Measure TTFT during chunk iteration:

```python
start_time = time.perf_counter()
first_chunk = True
for chunk in stream:
    if first_chunk:
        ttft_ms = (time.perf_counter() - start_time) * 1000
        first_chunk = False
    yield chunk
# Final chunk may contain usage/timings from server
```

**Effort:** ~50 lines

---

### Phase 3: Simple Metrics Logger
**File:** `src/brio_ext/metrics/logger.py`

JSONL-based logging with minimal overhead:

```python
class MetricsLogger:
    def __init__(self, log_path: Path = None):
        self.log_path = log_path or Path("~/.briodocs/metrics.jsonl").expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, tier_id: str, tier_label: str, model: str,
            timings: dict, usage: dict, context_size: int = None):
        record = {
            "ts": datetime.utcnow().isoformat(),
            "tier_id": tier_id,
            "tier_label": tier_label,
            "model": model,
            "context_size": context_size,
            **timings,
            **usage
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_recent(self, n: int = 100) -> list:
        """Read last n records for frontend display."""
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-n:]]
```

**Log format:**
```jsonl
{"ts": "2025-12-08T10:30:00Z", "tier_id": "fast", "tier_label": "Fast", "model": "llama-3.2-3b-q4", "context_size": 4096, "ttft_ms": 145, "tokens_per_second": 23.5, "prompt_tokens": 88, "completion_tokens": 42}
```

**Effort:** ~40 lines

---

### Phase 4: Wire Up in brio-ext
At the tier/model call site:

```python
response = model.chat_complete(messages)

metrics_logger.log(
    tier_id=selected_tier.id,
    tier_label=selected_tier.label,
    model=response.model,
    context_size=selected_tier.context_size,
    timings=response.timings.model_dump() if response.timings else {},
    usage=response.usage.model_dump() if response.usage else {}
)

return response  # Frontend displays timings from response
```

**Effort:** ~10 lines

---

## Storage Strategy

### Local (Now)
- **Format:** JSONL (one JSON object per line)
- **Location:** `~/.briodocs/metrics.jsonl`
- **Rolling window:** Log rotation (logrotate or manual cleanup)
- **Why:** Zero dependencies, easy to debug, trivial to parse

### Upstream (Future)
- Batch read JSONL file
- POST to analytics endpoint
- Clear sent records or use cursor/offset tracking

---

## Tier Tracking

Snapshot tier context at request time to handle future config changes:

```python
{
    "tier_id": "fast",           # Stable identifier
    "tier_label": "Fast",        # Human-readable (may change)
    "model": "llama-3.2-3b-q4",  # Actual model used
    "context_size": 4096,        # Performance-relevant params
}
```

This ensures historical data remains interpretable even when tier definitions change.

---

## Testing

```bash
# Run tests after each phase
uv run pytest -v tests/test_response_types.py
uv run pytest -v tests/test_llamacpp_timings.py
uv run pytest -v tests/test_metrics_logger.py
```

---

## Summary

| Phase | What | Where | Lines | Status |
|-------|------|-------|-------|--------|
| 1 | Add `Timings` type | esperanto/common_types | ~20 | Complete |
| 2 | Extract/measure timings | brio_ext/llamacpp_provider | ~70 | Complete |
| 3 | Simple JSONL logger | brio_ext/metrics | ~200 | Complete |
| 4 | Wire up at call site | brio_ext/factory | ~30 | Complete |

**Total:** ~320 lines, no new dependencies

## How to Enable Metrics Logging

### Option 1: Environment Variable (Startup)

Set the environment variable before running:

```bash
export BRIO_METRICS_ENABLED=1
```

### Option 2: Runtime Control (Settings Page)

Metrics can be toggled at runtime without restarting the app:

```python
from brio_ext import enable_metrics, disable_metrics, is_metrics_enabled

# Enable logging (uses default path: ~/.briodocs/metrics.jsonl)
enable_metrics()

# Or with custom path
enable_metrics(log_path="/custom/path/metrics.jsonl")

# Disable logging
disable_metrics()

# Check current state for UI toggle
current_state = is_metrics_enabled()
```

### API Reference

| Function | Description |
|----------|-------------|
| `enable_metrics(log_path=None)` | Enable metrics logging. Optionally pass a custom path string. |
| `disable_metrics()` | Disable metrics logging (subsequent LLM calls skip logging) |
| `is_metrics_enabled()` | Returns `True` if metrics are currently being collected |

### Default Log Location

`~/.briodocs/metrics.jsonl` (JSONL format, one record per line)

**Note:** Streaming responses do not log automatically. For streaming, access `response.ttft_ms` after consuming the stream and log manually if needed.

---

## Logged Metrics

Each log entry contains the following fields:

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `ts` | string | ISO 8601 timestamp | Generated |
| `tier_id` | string | Tier identifier (e.g., "llamacpp") | Config or provider |
| `model` | string | Model name used | Request |
| `total_time_ms` | float | Wall-clock request duration (ms) | Measured in factory |
| `tokens_per_second` | float | Generation speed (tokens/sec) | Calculated or llama.cpp |
| `prompt_tokens` | int | Input token count | llama.cpp response |
| `completion_tokens` | int | Output token count | llama.cpp response |
| `total_tokens` | int | Total tokens used | llama.cpp response |
| `tier_label` | string | Human-readable tier name | Config (optional) |
| `context_size` | int | Context window size | Config (optional) |
| `ttft_ms` | float | Time to first token (streaming only) | Measured (optional) |

### Example Log Entry

```json
{"ts": "2025-12-09T02:15:30.123Z", "tier_id": "llamacpp", "model": "mistral-7b-instruct", "tokens_per_second": 4.71, "total_time_ms": 76425.32, "prompt_tokens": 223, "completion_tokens": 360, "total_tokens": 583}
```

### Tokens Per Second Calculation

`tokens_per_second` is calculated automatically if not provided by llama.cpp:

```
tokens_per_second = completion_tokens / (total_time_ms / 1000)
```

### TTFT vs Total Time

| Metric | Description | When Available |
|--------|-------------|----------------|
| `ttft_ms` | Time to first token | **Streaming only** - first chunk arrival |
| `total_time_ms` | Complete request duration | **All requests** - wall-clock time |

For non-streaming requests, all tokens arrive at once, so only `total_time_ms` is meaningful.

---

## Files Changed

- `src/esperanto/common_types/response.py` - Added `Timings` class
- `src/esperanto/common_types/__init__.py` - Exported `Timings`
- `src/brio_ext/providers/llamacpp_provider.py` - Added timing extraction and streaming wrappers
- `src/brio_ext/metrics/logger.py` - New MetricsLogger class
- `src/brio_ext/metrics/__init__.py` - New module init
- `src/brio_ext/__init__.py` - Exported new classes
- `src/brio_ext/factory.py` - Added automatic metrics logging for non-streaming responses
- `tests/test_timings.py` - 17 tests for new functionality
