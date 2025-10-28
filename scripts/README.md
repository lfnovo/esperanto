# Manual Testing Scripts for brio_ext

Test scripts for validating brio_ext integration with llama.cpp models and BrioDocs configuration.

## Quick Start

### 1. Download a Model (One-Time Setup)

Start with Qwen 2.5 7B Instruct (~4.4GB):

```bash
./scripts/download_models.sh qwen-2.5-7b-instruct
```

This downloads to `models/qwen2.5-7b-instruct-q4_k_m.gguf`

### 2. Start the Server (Terminal 1)

Use the new tier-based server launcher:

```bash
conda activate briodocs

# Start with Tier 2 (GPU + 4K context) and Model 1 (Qwen 2.5 7B)
./scripts/start_server_v2.sh --tier 2 --model 1
```

**Tier Selection Guide:**
- **Tier 1** (High Performance): 8K context, GPU acceleration, best for development
- **Tier 2** (Balanced): 4K context, GPU acceleration, recommended for production
- **Tier 3** (Fast): 2K context, CPU-only, for quick testing (not recommended for reasoning test)

Wait for "Server ready" or "Uvicorn running" message. The terminal title will show your tier and model configuration.

### 3. Run Tests (Terminal 2)

```bash
conda activate briodocs

# Test by scenario and model number (positional arguments)
python scripts/test_with_llm.py pirate 1
```

**IMPORTANT**: The test script uses positional arguments, not flags:
- First argument: scenario name (`pirate`, `inventor`, `multiturn`, `reasoning`, `all`)
- Second argument: model number (`1`-`7`) or `openai`

## Available Test Scenarios

### pirate
- **Purpose**: Simple system message test
- **What it tests**: Basic adapter selection and system message handling
- **Expected**: Model responds in pirate speak (e.g., "Arrr, that be 4!")
- **If it fails**: System messages are being ignored or adapter not selected
- **Best tier**: Any (Tier 3 is fine for this lightweight test)

### inventor
- **Purpose**: Medium context with real BrioDocs-style payload
- **What it tests**: The Qwen system message bug fix
- **Expected**: Model extracts inventor names from system context
- **If it fails**: System messages not working (this was the original bug)
- **Best tier**: Tier 2+ for good performance

### multiturn
- **Purpose**: Multi-turn conversation with context
- **What it tests**: Conversation history handling
- **Expected**: Model uses previous turns to answer follow-up question
- **If it fails**: Context not being maintained across turns
- **Best tier**: Tier 2+ recommended

### reasoning
- **Purpose**: Complex patent risk analysis with detailed output
- **What it tests**: Structured risk analysis with multiple technical/legal areas
- **Expected**: Model identifies 5-7 technical/legal risks in patent content
- **Details**:
  - Large prompt (1.3K tokens) + long response (600+ tokens)
  - Uses 1024 max_tokens (vs 512 for other tests)
  - Improved prompt points model to specific risk categories
- **If it fails**: Check if model has enough context/tokens or if prompt needs refinement
- **Best tier**: **Tier 2+ required** (Tier 3 CPU-only will timeout or be very slow >2 min)

## Usage Examples

### Test Individual Scenarios

```bash
# Pirate test with Model 1 (Qwen 2.5 7B)
python scripts/test_with_llm.py pirate 1

# Inventor test with Model 3 (Llama 3.1 8B)
python scripts/test_with_llm.py inventor 3

# Multi-turn test with Model 6 (Phi-4 Mini)
python scripts/test_with_llm.py multiturn 6

# Reasoning test with Model 7 (Phi-4 Reasoning) - requires Tier 2+
python scripts/test_with_llm.py reasoning 7

# Reasoning test with Model 1 (Qwen) - also works well
python scripts/test_with_llm.py reasoning 1
```

### Test Against OpenAI Baseline

```bash
# Make sure OPENAI_API_KEY is set
export OPENAI_API_KEY=sk-...

# Test scenarios with GPT-4o-mini (baseline)
python scripts/test_with_llm.py pirate openai
python scripts/test_with_llm.py inventor openai
python scripts/test_with_llm.py reasoning openai
```

### Run All Scenarios

```bash
# Run all tests with Model 1
python scripts/test_with_llm.py all 1

# Press Enter to continue between scenarios
```

## Available Models

| Model # | Name | Size | Format | Notes |
|---------|------|------|--------|-------|
| 1 | Qwen 2.5 7B Instruct | 4.4GB | ChatML | Default, good all-rounder |
| 2 | Qwen 2.5 3B Instruct | 2.0GB | ChatML | Faster, lighter |
| 3 | Llama 3.1 8B Instruct | 4.9GB | Llama | Requires proper stop tokens |
| 4 | Llama 3.2 3B Instruct | 2.0GB | Llama | Lighter Llama variant |
| 5 | Mistral 7B Instruct v0.3 | 4.4GB | Mistral | Alternative 7B model |
| 6 | Phi-4 Mini Instruct | 2.7GB | ChatML | Microsoft's efficient model |
| 7 | Phi-4 Reasoning | ~3GB | ChatML | Chain-of-thought reasoning |

**Download models:**

```bash
# Single model
./scripts/download_models.sh qwen-2.5-7b-instruct

# All models (~20GB)
./scripts/download_models.sh all
```

**Switch models:**

```bash
# Terminal 1: Stop current server
./scripts/stop_server.sh

# Start with different tier and model
./scripts/start_server_v2.sh --tier 1 --model 7

# Terminal 2: Test with the new model
python scripts/test_with_llm.py reasoning 7
```

## Server Configuration (start_server_v2.sh)

The new tier-based launcher separates configuration concerns:

**Syntax:**
```bash
./scripts/start_server_v2.sh --tier <1-3> --model <1-7>
```

**Tiers (HOW to run):**
- Tier 1: 8K context, GPU (-1 layers), 8 threads - High performance
- Tier 2: 4K context, GPU (-1 layers), 8 threads - Balanced
- Tier 3: 2K context, CPU (0 layers), 8 threads - Fast/testing

**Models (WHAT to run):**
- Model selection is independent of tier
- Server detects model format and applies correct chat template
- Configuration centralized in `fixtures/briodocs_config.yaml`

**Terminal title displays:**
```
llama.cpp: Tier 2 - Qwen 2.5 7B Instruct
```

## Test Output Explained

The test runner shows **full pipeline visibility**:

### 1. Test Configuration
```
SCENARIO: reasoning
MODEL: Qwen 2.5 7B Instruct
PROVIDER: llamacpp
DESCRIPTION: Complex analysis task - Tests patent risk analysis with detailed output
```

### 2. Step 1: TEST → ESPERANTO/BRIO_EXT
Shows the messages being sent to brio_ext (system + user)

### 3. Step 2: ESPERANTO/BRIO_EXT → LLM SERVER
Debug output shows:
- Adapter selected (e.g., QwenAdapter, LlamaAdapter)
- Rendered prompt sent to model
- Stop tokens applied

### 4. Step 3: BRIO_EXT → TEST (Fenced Response)
The complete response with `<out>...</out>` fencing

### 5. Step 4: PARSE & VALIDATE
- Fence validation (should be present)
- Extracted content (what's inside `<out>...</out>`)
- Content validation (checks for expected phrases)

### 6. Response Metadata
```
Model: Qwen 2.5 7B Instruct
Provider: llamacpp
Finish reason: stop
Prompt tokens: 1,348
Completion tokens: 599
Total tokens: 1,947
```

### 7. Validation Results
```
✓ Response properly fenced in <out>...</out>

Checking for expected content (1 groups, any match per group):
  ✓ Found 'risk' (from: 'FINAL ANSWER' OR 'limitation' OR 'risk')

✓ Stop reason: stop
```

### 8. Overall Result
```
✅ TEST PASSED
```

## Configuration Files

### fixtures/briodocs_config.yaml
Centralized BrioDocs-standard configuration:

```yaml
model_parameters:
  temperature: 0.5      # Changed from 0.7 after testing
  top_p: 0.9
  top_k: 40
  max_tokens: 512
  # ... other parameters

tiers:
  tier1:
    name: "High Performance"
    server_config:
      n_ctx: 8192
      n_gpu_layers: -1
      n_threads: 8
  # ... tier2, tier3
```

### fixtures/test_cases.yaml
Test case definitions with validation rules:

```yaml
reasoning:
  description: "Complex analysis task - Tests patent risk analysis"
  system: prompts/system/reasoning_analyst.txt
  user: prompts/user/extract_risks.txt
  content: prompts/content/patent_mobile_device.txt
  max_tokens: 1024  # Override for heavy tests
  validation:
    should_contain_any:
      - ["FINAL ANSWER", "limitation", "risk"]
    should_fence: true
```

## Troubleshooting

### Wrong Command Syntax

**WRONG:**
```bash
python scripts/test_with_llm.py --tier 1 --model 1  # ❌ NO FLAGS
```

**CORRECT:**
```bash
# Tier/model selection is for SERVER startup only:
./scripts/start_server_v2.sh --tier 2 --model 1

# Tests use positional arguments:
python scripts/test_with_llm.py pirate 1
```

### Server Won't Start - Port in Use

```bash
./scripts/stop_server.sh

# Or manually:
kill $(lsof -t -i:8765)
```

### Test Timeouts (Reasoning Test)

**Symptom**: Test times out or takes >2 minutes

**Cause**: Running Tier 3 (CPU-only) with heavy tests

**Solution**: Use Tier 2 or higher:
```bash
./scripts/stop_server.sh
./scripts/start_server_v2.sh --tier 2 --model 1
python scripts/test_with_llm.py reasoning 1
```

### Model File Not Found

```bash
# Check if it exists
ls -lh models/

# Re-download if needed
./scripts/download_models.sh qwen-2.5-7b-instruct
```

### Import Errors

```bash
# Make sure you're in the conda environment
conda activate briodocs

# Verify brio_ext can be imported
python -c "from brio_ext.factory import BrioAIFactory; print('OK')"
```

### OpenAI Tests Fail

```bash
# Check API key is set
echo $OPENAI_API_KEY

# Set it if needed
export OPENAI_API_KEY=sk-...
```

## What This Tests

This environment validates the complete brio_ext integration:

1. **Adapter Selection**: Does brio_ext pick the right adapter for each model?
2. **Prompt Rendering**: Are chat templates (ChatML, Llama, Mistral) correct?
3. **Response Cleaning**: Do adapters clean format markers before fencing?
4. **Stop Token Handling**: Are stop tokens properly configured and merged?
5. **Output Fencing**: Do responses come back in `<out>...</out>` tags?
6. **System Message Handling**: The KEY question - do models respect system context?
7. **Provider Integration**: Does llamacpp_provider work correctly with server?
8. **Baseline Comparison**: Does OpenAI passthrough work as expected?

## Performance Characteristics

| Test | Prompt Size | Response Size | Tier 3 (CPU) | Tier 2 (GPU) |
|------|-------------|---------------|--------------|--------------|
| pirate | ~200 tokens | ~50 tokens | Fast (<10s) | Very Fast (<5s) |
| inventor | ~750 tokens | ~30 tokens | Medium (~30s) | Fast (<10s) |
| multiturn | ~800 tokens | ~100 tokens | Medium (~40s) | Fast (<15s) |
| reasoning | ~1.3K tokens | ~600 tokens | **Slow (>2min or timeout)** | Fast (~30s) |

**Recommendation**: Use Tier 2 (GPU + 4K context) for all testing except quick pirate checks.

## Next Steps

Once you validate the test environment:

1. ✅ **Core Tests Pass**: pirate, inventor, multiturn
2. ✅ **Reasoning Test Works**: With Tier 2+, identifies multiple risks
3. 🔄 **Test Other Models**: Llama, Mistral, Phi variants
4. 🔄 **Integrate into BrioDocs**: Use brio_ext in production application
5. 🔄 **Add More Scenarios**: As needed for specific BrioDocs features

## Debug Mode

Debug mode is **always enabled** in test scripts (`BRIO_DEBUG=1`). This shows:
- Renderer decisions and adapter selection
- Prompt generation with exact template format
- Stop token merging logic
- Provider API calls with parameters

This visibility is the key to understanding if brio_ext is working correctly!

## Related Documentation

- **[llama.cpp Test Specification](../docs/llama_cpp_test_specification.md)** - Original test spec with detailed scenarios
- **[brio_ext Integration Guide](../docs/brio_ext_integration.md)** - Integration guide for BrioDocs apps
- **[BrioDocs Config](../fixtures/briodocs_config.yaml)** - Tier and model parameter configuration
- **[Test Cases](../fixtures/test_cases.yaml)** - Test scenario definitions
