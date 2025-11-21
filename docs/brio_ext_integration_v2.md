# Section 9: llama.cpp Test Matrix & Scenarios - UPDATED

**Note:** This is an updated version of section 9 from brio_ext_integration.md reflecting the new tier-based architecture.

## 9.1 Server Tiers (New Architecture)

Use the tier-based launcher for simplified configuration:

```bash
# Tier 2 (Balanced, recommended) with Qwen 2.5 7B (Model 1)
./scripts/start_server_v2.sh --tier 2 --model 1

# Tier 1 (High Performance) with Phi-4 Reasoning (Model 7)
./scripts/start_server_v2.sh --tier 1 --model 7

# Tier 3 (Fast/Testing) with Qwen 2.5 3B (Model 2)
./scripts/start_server_v2.sh --tier 3 --model 2
```

**Tier Configuration** (from `fixtures/briodocs_config.yaml`):

| Tier | Context | GPU | Threads | Use Case | Reasoning Test |
|------|---------|-----|---------|----------|----------------|
| Tier 1 | 8K | -1 (all) | 8 | Development, research | Fast (~20s) |
| Tier 2 | 4K | -1 (all) | 8 | Production, recommended | Fast (~30s) |
| Tier 3 | 2K | 0 (CPU-only) | 8 | Quick testing | Slow (>2min) |

**Available Models:**

| Model # | Name | Size | Format | Notes |
|---------|------|------|--------|-------|
| 1 | Qwen 2.5 7B Instruct | 4.4GB | ChatML | Default, all-rounder |
| 2 | Qwen 2.5 3B Instruct | 2.0GB | ChatML | Faster, lighter |
| 3 | Llama 3.1 8B Instruct | 4.9GB | Llama | Requires proper stop tokens |
| 4 | Llama 3.2 3B Instruct | 2.0GB | Llama | Lighter Llama variant |
| 5 | Mistral 7B Instruct v0.3 | 4.4GB | Mistral | Alternative 7B model |
| 6 | Phi-4 Mini Instruct | 2.7GB | ChatML | Microsoft's efficient model |
| 7 | Phi-4 Reasoning | ~3GB | ChatML | Chain-of-thought reasoning |

**Benefits of tier-based approach:**
- Tier defines **HOW** to run (context, GPU settings)
- Model defines **WHAT** to run (which GGUF file)
- Configuration centralized in `fixtures/briodocs_config.yaml`
- Terminal title shows active tier and model
- Easy switching: `./scripts/stop_server.sh` then restart with different tier/model

### Custom Model Names with `chat_format`

When working with custom model names (e.g., from BrioDocs model database), you can specify the chat format explicitly:

```python
from brio_ext.factory import BrioAIFactory

model = BrioAIFactory.create_language(
    provider="llamacpp",
    model_name="phi-4-mini-reasoning",  # Custom model name
    config={
        "base_url": "http://127.0.0.1:8765",
        "chat_format": "chatml",  # Explicitly specify ChatML format
        "temperature": 0.5,
    },
)
```

**Supported `chat_format` values:**
- `"chatml"` – for Qwen, Phi-4 (models 1, 2, 6, 7 above)
- `"llama"` / `"llama3"` – for Llama models (models 3, 4)
- `"mistral"` – for Mistral (model 5)
- `"gemma"` – for Gemma models

**Use case:** This is essential when integrating with systems that store model configurations in databases (like BrioDocs `model_defaults.json`). You can pass the `chat_format` field directly from your model config, enabling custom model names that don't follow standard patterns.

## 9.2 Test Scenarios

Run tests using the test runner:

```bash
# Test individual scenarios
python scripts/test_with_llm.py pirate 1      # Simple system message
python scripts/test_with_llm.py inventor 1    # Medium context (KEY Qwen test)
python scripts/test_with_llm.py multiturn 1   # Conversation history
python scripts/test_with_llm.py reasoning 1   # Complex analysis (requires Tier 2+)

# Run all tests
python scripts/test_with_llm.py all 1
```

**Test Characteristics:**

| Scenario | Prompt Size | Response Size | Tier 3 (CPU) | Tier 2 (GPU) |
|----------|-------------|---------------|--------------|--------------|
| pirate | ~200 tokens | ~50 tokens | Fast (<10s) | Very Fast (<5s) |
| inventor | ~750 tokens | ~30 tokens | Medium (~30s) | Fast (<10s) |
| multiturn | ~800 tokens | ~100 tokens | Medium (~40s) | Fast (<15s) |
| reasoning | ~1.3K tokens | ~600 tokens | **Slow (>2min)** | Fast (~30s) |

## 9.3 Expected Results

All tests should pass with proper `<out>...</out>` fencing:

```
✅ TEST PASSED
✓ Response properly fenced in <out>...</out>
✓ Found: 'Richard H. Xu'  (inventor test)
✓ Found: 'risk'  (reasoning test)
✓ Stop reason: stop
```

**Key Validations:**
1. **Fencing**: Response wrapped in `<out>...</out>`
2. **Content**: Expected phrases present
3. **Stop Reason**: Clean termination (not cut off)
4. **Format Markers**: No `<|im_start|>`, `[INST]`, etc. in output

## 9.4 Configuration

### fixtures/briodocs_config.yaml

Centralized configuration for tiers and model parameters:

```yaml
model_parameters:
  temperature: 0.5      # Reduced from 0.7 for better accuracy
  top_p: 0.9
  top_k: 40
  frequency_penalty: 0.0
  presence_penalty: 0.0
  max_tokens: 512
  stream: false

tiers:
  tier1:
    name: "High Performance"
    description: "16GB+ RAM, 8K context, GPU acceleration"
    server_config:
      n_ctx: 8192
      n_gpu_layers: -1
      use_mlock: true
      n_threads: 8

  tier2:
    name: "Balanced"
    description: "8GB+ RAM, 4K context, GPU acceleration"
    server_config:
      n_ctx: 4096
      n_gpu_layers: -1
      use_mlock: true
      n_threads: 8

  tier3:
    name: "Fast"
    description: "4GB+ RAM, 2K context, CPU-only"
    server_config:
      n_ctx: 2048
      n_gpu_layers: 0
      use_mlock: true
      n_threads: 8
```

### fixtures/test_cases.yaml

Test scenario definitions with validation rules:

```yaml
pirate:
  description: "Simple system message test - Pirate personality"
  system: prompts/system/pirate.txt
  user: prompts/user/simple_math.txt
  validation:
    should_contain_any:
      - ["4", "four"]
    should_fence: true

inventor:
  description: "Medium context inventor lookup"
  system: prompts/system/patent_analyst.txt
  user: prompts/user/who_are_inventors.txt
  content: prompts/content/patent_mobile_device.txt
  insights:
    - prompts/insights/patent_dense_summary.txt
  validation:
    should_contain: ["Richard H. Xu", "Xiaolei Qin", "Phillip C. Krasko", "Douglas A. Cheline"]
    should_fence: true

reasoning:
  description: "Complex analysis task - Tests patent risk analysis"
  system: prompts/system/reasoning_analyst.txt
  user: prompts/user/extract_risks.txt
  content: prompts/content/patent_mobile_device.txt
  max_tokens: 1024  # Override for heavy analysis
  validation:
    should_contain_any:
      - ["FINAL ANSWER", "limitation", "risk"]
    should_fence: true
    notes: "Heavy test - requires Tier 2+ for good performance"
```

## 9.5 Troubleshooting

### Wrong Command Syntax

**WRONG:**
```bash
python scripts/test_with_llm.py --tier 1 --model 1  # ❌ NO FLAGS
```

**CORRECT:**
```bash
# Server startup uses flags:
./scripts/start_server_v2.sh --tier 2 --model 1

# Test script uses positional arguments:
python scripts/test_with_llm.py pirate 1
```

### Timeouts on Reasoning Test

**Cause:** Running Tier 3 (CPU-only) with heavy tests

**Solution:**
```bash
./scripts/stop_server.sh
./scripts/start_server_v2.sh --tier 2 --model 1
python scripts/test_with_llm.py reasoning 1
```

### Adapter Not Selected

**Symptom:** Debug output shows wrong adapter or no adapter

**Cause:** Model name not recognized by renderer

**Solution:** Check model name matches expected patterns in `src/brio_ext/renderer.py`

### Missing Format Markers in Output

**Symptom:** Response contains `<|im_start|>` or `[INST]` markers

**Cause:** Adapter's `clean_response()` not working

**Solution:** Check adapter implementation in `src/brio_ext/adapters/`

## 9.6 Status Matrix

| Model | Format | Status | Notes |
|-------|--------|--------|-------|
| Qwen 2.5 7B | ChatML | ✅ | All tests pass |
| Qwen 2.5 3B | ChatML | ✅ | All tests pass |
| Llama 3.1 8B | Llama | ✅ | Stop tokens fixed |
| Llama 3.2 3B | Llama | ✅ | Stop tokens fixed |
| Mistral 7B v0.3 | Mistral | ✅ | All tests pass |
| Phi-4 Mini | ChatML | ✅ | All tests pass |
| Phi-4 Reasoning | ChatML | ✅ | Works well for reasoning |

## 9.7 Performance Recommendations

**Production Setup:**
- **Tier 2** (GPU + 4K context) for all workloads
- **Qwen 2.5 7B** (Model 1) as default model
- **temperature: 0.5** for better accuracy vs creativity balance

**Development/Research:**
- **Tier 1** (GPU + 8K context) for maximum quality
- Test with multiple models to ensure compatibility
- Use debug mode (`BRIO_DEBUG=1`) to inspect pipeline

**Quick Testing:**
- **Tier 3** (CPU-only) for pirate/inventor tests only
- Avoid reasoning test on Tier 3 (will timeout)

## 9.8 Related Documentation

- **[scripts/README.md](../scripts/README.md)** - Complete testing guide
- **[llama.cpp Test Specification](./llama_cpp_test_specification.md)** - Detailed test spec
- **[start_server_v2.sh](../scripts/start_server_v2.sh)** - Tier-based launcher
- **[test_with_llm.py](../scripts/test_with_llm.py)** - Test runner with full pipeline visibility
