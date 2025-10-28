# BrioDocs Performance Tiers

This document describes the three performance tiers used by BrioDocs and how they're configured in the testing environment.

## Standard Model Parameters (All Tiers)

All models use these parameters regardless of tier:

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "max_tokens": 512,
  "stream": false
}
```

## Tier 1: High Performance (16GB+ RAM)

**Use Case:** Development, research, maximum quality
**Candidate Count:** 5 responses (with reranking)
**Context Window:** 8,192 tokens

### Models
- Qwen 2.5 7B Instruct (model 1)
- Llama 3.1 8B Instruct (model 3)

### Server Configuration
```bash
./scripts/start_server.sh 1  # Qwen 2.5 7B
./scripts/start_server.sh 3  # Llama 3.1 8B

# Configuration:
--n_ctx 8192
--n_gpu_layers -1    # Use ALL GPU layers
--use_mlock True     # Lock model in RAM (no swapping)
--n_threads 8
--chat_format chatml  # or llama-3 for Llama models
```

---

## Tier 2: Balanced (8GB+ RAM)

**Use Case:** Production, most users
**Candidate Count:** 3 responses (with reranking)
**Context Window:** 4,096 tokens

### Models
- Qwen 2.5 3B Instruct (model 2)
- Llama 3.2 3B Instruct (model 4)
- Mistral 7B Instruct v0.3 (model 5)

### Server Configuration
```bash
./scripts/start_server.sh 2  # Qwen 2.5 3B
./scripts/start_server.sh 4  # Llama 3.2 3B
./scripts/start_server.sh 5  # Mistral 7B

# Configuration:
--n_ctx 4096
--n_gpu_layers -1    # Use ALL GPU layers
--use_mlock True
--n_threads 8
--chat_format chatml  # or llama-3/mistral-instruct depending on model
```

---

## Tier 3: Fast (4GB+ RAM, CPU-only)

**Use Case:** Low-resource environments, quick testing
**Candidate Count:** 1 response (NO reranking)
**Context Window:** 2,048 tokens

### Models
- Phi-4 Mini Instruct (model 6)

### Server Configuration
```bash
./scripts/start_server.sh 6  # Phi-4 Mini

# Configuration:
--n_ctx 2048
--n_gpu_layers 0     # CPU ONLY (no GPU)
--use_mlock True
--n_threads 8
--chat_format chatml
```

---

## Tier 2 Reasoning: Complex Analysis (8GB+ RAM, GPU)

**Use Case:** Complex analytical tasks - risk extraction, contradiction detection, material facts analysis
**Candidate Count:** 1 response (reasoning models don't benefit from reranking)
**Context Window:** 4,096 tokens

### Models
- Phi-4 Reasoning (model 7)

### Server Configuration
```bash
./scripts/start_server.sh 7  # Phi-4 Reasoning

# Configuration:
--n_ctx 4096
--n_gpu_layers -1    # Use GPU for performance
--use_mlock True
--n_threads 8
--chat_format chatml
```

**Note:** Reasoning models expose chain-of-thought and work best with structured output prompts. Use the `reasoning_analyst.txt` system prompt for complex analytical queries.

---

## Important Notes

### Reranking
BrioDocs application handles reranking logic:
- **Tier 1:** Generates 5 candidate responses, selects best
- **Tier 2:** Generates 3 candidate responses, selects best
- **Tier 3:** Generates 1 response only (no reranking)

Esperanto/brio_ext doesn't need to implement reranking—just handle individual requests correctly.

### Testing Configuration
All test configurations are defined in `fixtures/briodocs_config.yaml` and automatically loaded by `scripts/test_with_llm.py`.

### Server Management
- **Start:** `./scripts/start_server.sh <model_number>`
- **Stop:** `./scripts/stop_server.sh`
- **Test:** `python scripts/test_with_llm.py <scenario> <model_number>`

### Context Windows
The old BrioDocs approach used custom context limits (6144, 4096), but the new approach uses native model capacity via `n_ctx`. This is already reflected in the server startup scripts.

### Temperature and Accuracy
The standard temperature of 0.7 provides good variety but may cause creative/incorrect answers for math problems (as seen with Phi-4 pirate test). For accuracy-critical tasks, BrioDocs may want to lower temperature to 0.1-0.3, but that's an application-level decision.
