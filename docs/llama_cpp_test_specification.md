# llama.cpp Server Test Specification for brio-esperanto

**Version:** 2.0
**Date:** October 27, 2025
**Purpose:** Comprehensive testing specification for brio-esperanto library integration with BrioDocs llama.cpp server

**Status:** ✅ Core issues resolved. Qwen system message bug fixed via adapter-based architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Server Configuration](#server-configuration)
4. [Test Scenarios](#test-scenarios)
5. [Message Structure](#message-structure)
6. [Context Size Test Cases](#context-size-test-cases)
7. [Parameters & Sampling](#parameters--sampling)
8. [Models to Test](#models-to-test)
9. [Expected Behavior](#expected-behavior)
10. [Resolution Status](#resolution-status)

---

## Overview

BrioDocs uses a local llama.cpp server to run quantized language models. The server provides an OpenAI-compatible API that brio-esperanto wraps through the `brio_ext` package. This document specifies how to test that brio-esperanto correctly handles:

- System messages with large context (insights, document content)
- Multi-turn chat history
- Different performance tiers
- Various model configurations (Qwen, Llama, Mistral, Phi-4)
- Adapter-based prompt rendering for different chat formats
- Response cleaning and `<out>...</out>` fencing

**Historical Issue:** Qwen models ignored system messages, leading to "I don't know" responses even when context was provided.

**Resolution:** Fixed via adapter-based architecture that properly renders chat templates for each model format (ChatML, Llama, Mistral).

---

## Quick Start

### 1. Start llama.cpp Server

Use the new tier-based launcher:

```bash
# Tier 2 (GPU + 4K context) with Qwen 2.5 7B (Model 1)
./scripts/start_server_v2.sh --tier 2 --model 1
```

**Tier Selection:**
- **Tier 1**: 8K context, GPU, best for development
- **Tier 2**: 4K context, GPU, recommended for production
- **Tier 3**: 2K context, CPU-only, for quick testing

### 2. Run Test Scenarios

```bash
# Test individual scenario
python scripts/test_with_llm.py pirate 1

# Run all tests
python scripts/test_with_llm.py all 1
```

**Available Scenarios:**
- `pirate` - Simple system message test
- `inventor` - Medium context (KEY test for Qwen bug)
- `multiturn` - Conversation history handling
- `reasoning` - Complex patent risk analysis (requires Tier 2+)

### 3. Expected Results

All tests should **PASS** with proper `<out>...</out>` fencing:

```
✅ TEST PASSED
✓ Response properly fenced in <out>...</out>
✓ Found: 'Richard H. Xu'
✓ Stop reason: stop
```

---

## Server Configuration

### New Tier-Based Architecture (v2.0)

The `start_server_v2.sh` script separates **HOW** to run (tier) from **WHAT** to run (model):

```bash
./scripts/start_server_v2.sh --tier <1-3> --model <1-7>
```

**Benefits:**
- Centralized configuration in `fixtures/briodocs_config.yaml`
- Model selection independent of performance tier
- Clear separation of concerns
- Terminal title shows active tier and model

### Performance Tier 1: High Performance (16GB+ RAM, GPU)

**Server Startup:**
```bash
python -m llama_cpp.server \
    --model /path/to/qwen-2.5-7b-instruct.gguf \
    --host 127.0.0.1 \
    --port 8765 \
    --n_ctx 8192              # 8K context window
    --n_gpu_layers -1         # Use ALL GPU layers
    --use_mlock True          # Lock model in RAM (no swapping)
    --n_threads 8             # Use all CPU cores (adjust for your system)
    --chat_format chatml      # Qwen uses chatml format
```

**Application Settings:**
- **Candidate count:** 5 responses (reranking enabled)
- **Context limit:** 8,192 tokens
- **Use case:** Development, research, maximum quality

---

### Performance Tier 2: Balanced (8GB+ RAM)

**Server Startup:**
```bash
python -m llama_cpp.server \
    --model /path/to/mistral-7b-instruct.gguf \
    --host 127.0.0.1 \
    --port 8765 \
    --n_ctx 4096              # 4K context window
    --n_gpu_layers -1         # Use ALL GPU layers
    --use_mlock True          # Lock model in RAM
    --n_threads 8             # Use all CPU cores
    --chat_format mistral-instruct
```

**Application Settings:**
- **Candidate count:** 3 responses
- **Context limit:** 6,144 tokens (deprecated - now use native model capacity)
- **Use case:** Production, most users

---

### Performance Tier 3: Fast (4GB+ RAM, CPU-only)

**Server Startup:**
```bash
python -m llama_cpp.server \
    --model /path/to/phi-4-mini.gguf \
    --host 127.0.0.1 \
    --port 8765 \
    --n_ctx 2048              # 2K context window (minimal)
    --n_gpu_layers 0          # CPU ONLY (no GPU)
    --use_mlock True          # Lock model in RAM
    --n_threads 8             # Use all CPU cores
    --chat_format chatml
```

**Application Settings:**
- **Candidate count:** 1 response (NO reranking)
- **Context limit:** 4,096 tokens (deprecated)
- **Use case:** Low-resource environments, quick testing

---

## Message Structure

### Single-Turn Chat (First Question)

**Request:**
```json
POST http://localhost:8765/v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {
      "role": "system",
      "content": "# SYSTEM ROLE\nYou are a specialized research assistant analyzing patent documents.\n\n# SOURCE CONTEXT\n\n## SOURCE CONTENT\n**Source ID:** source:abc123\n**Title:** Mobile Device with Enhanced Touch Interface (US20200336491A1)\n**Content:** [Truncated to 20,000 chars]\n\nBACKGROUND OF THE INVENTION\n[1] This invention relates to mobile communication devices...\n[Content continues for ~20K chars]\n\n## SOURCE INSIGHTS\n**Insight ID:** insight:xyz789\n**Type:** Dense Summary SPR\n**Content:** This patent describes a dynamic communication profile system for mobile devices. The key innovation allows devices to switch between local and global cellular networks seamlessly. The inventors are: Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, Douglas A. Cheline. The invention solves the problem of expensive roaming charges by enabling automatic local network detection and profile switching.\n[Insight continues for ~5K chars, truncated from 195K]\n\n## CONTEXT METADATA\n- Source count: 1\n- Insight count: 1\n- Total tokens: 5,734\n- Total characters: 22,935"
    },
    {
      "role": "user",
      "content": "Who are the inventors of this patent?"
    }
  ],
  "model": "qwen-7b-instruct",
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Expected Response:**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The inventors of this patent are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline."
      }
    }
  ]
}
```

---

### Multi-Turn Chat (With History)

**Request:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "[Same 22,935 char system message with context]"
    },
    {
      "role": "user",
      "content": "Who are the inventors?"
    },
    {
      "role": "assistant",
      "content": "The inventors of this patent are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline."
    },
    {
      "role": "user",
      "content": "What problem does this invention solve?"
    }
  ],
  "model": "qwen-7b-instruct",
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Expected Response:**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "This invention solves the problem of expensive roaming charges when traveling internationally. It allows mobile devices to automatically detect and switch to local cellular networks instead of relying on global (home network) connections, which are often costly."
      }
    }
  ]
}
```

---

## Context Size Test Cases

### Test Case 1: Small Context (~2,500 chars, ~625 tokens)

**Scenario:** Document without insights (e.g., engagement letter)

**System Message Structure:**
```
# SYSTEM ROLE
You are a specialized research assistant...

# SOURCE CONTEXT

## SOURCE CONTENT
**Source ID:** source:abc123
**Title:** 2023.12.12 Engagement Letter to Douglas Cheline
**Content:** [Document content - 2,500 chars]

## CONTEXT METADATA
- Source count: 1
- Insight count: 0
- Total tokens: 625
```

**Test Query:** "What is this document about?"

**Expected:** Model uses title and content to describe engagement letter  
**Actual Bug (Qwen):** "I don't have information about this document"  
**Actual Works (GPT-4o-mini):** Correct description of engagement letter

---

### Test Case 2: Medium Context (~23,000 chars, ~5,750 tokens)

**Scenario:** Patent document with 1 truncated insight

**System Message Structure:**
```
# SYSTEM ROLE
You are a specialized research assistant...

# SOURCE CONTEXT

## SOURCE CONTENT
**Source ID:** source:80cdi72vie9y3gllak56
**Title:** Mobile Device with Enhanced Touch Interface (US20200336491A1)
**Content:** [Truncated to 20,000 chars - full document is 148K chars]

## SOURCE INSIGHTS
**Insight ID:** insight:a7i32qbc1mamzpeurnbt
**Type:** Dense Summary SPR
**Content:** [20,000 chars of insight - truncated from 195K chars]
[Content truncated - insight is very large]

## CONTEXT METADATA
- Source count: 1
- Insight count: 1
- Total tokens: 5,734
- Total characters: 22,935
```

**Test Query:** "Who are the inventors?"

**Expected:** Model uses insight to answer correctly  
**Actual Bug (Qwen):** "I don't know which patent you're referring to"  
**Actual Works (GPT-4o-mini):** "The inventors are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline [source:80cdi72vie9y3gllak56][insight:a7i32qbc1mamzpeurnbt]"

---

### Test Case 3: Large Context (~200,000 chars, ~50,000 tokens - BEFORE truncation)

**Scenario:** Large patent with 2 full synthesized insights (tests our ContextBuilder truncation logic)

**System Message Structure (before our truncation):**
```
# SYSTEM ROLE
...

## SOURCE INSIGHTS
**Insight ID:** insight:abc
**Type:** Dense Summary SPR
**Content:** [195,827 chars - VERY LARGE synthesized insight from chunking]

**Insight ID:** insight:def
**Type:** Patent SPR
**Content:** [98,234 chars - Another large insight]
```

**Note:** This tests that our application properly truncates insights BEFORE sending to brio-esperanto. You should NOT receive requests this large. If you do, we have a bug in our ContextBuilder.

---

## Parameters & Sampling

### Standard Parameters (all tiers)

```json
{
  "model": "qwen-7b-instruct",       // Model identifier
  "temperature": 0.7,                 // Sampling temperature
  "top_p": 0.9,                       // Nucleus sampling
  "top_k": 40,                        // Top-K sampling
  "frequency_penalty": 0.0,           // Penalize frequent tokens
  "presence_penalty": 0.0,            // Penalize present tokens
  "max_tokens": 512,                  // Max response length
  "stream": false                     // Streaming (optional)
}
```

### Tier-Specific Behavior

**Tier 1 (High Performance):**
- Generates 5 candidate responses
- Application selects best using reranking heuristic
- You'll receive 5 sequential/parallel requests with same messages

**Tier 2 (Balanced):**
- Generates 3 candidate responses
- Application selects best

**Tier 3 (Fast):**
- Generates 1 response only
- No reranking

**Note:** brio-esperanto doesn't need to implement reranking logic—that's handled by our application. You just need to handle the individual requests correctly.

---

## Models to Test

### Local Models (llama.cpp)

#### Qwen 2.5 7B Instruct
```bash
# Download
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf

# Server startup
python -m llama_cpp.server \
    --model qwen2.5-7b-instruct-q4_k_m.gguf \
    --host 127.0.0.1 \
    --port 8765 \
    --n_ctx 8192 \
    --n_gpu_layers -1 \
    --use_mlock True \
    --n_threads 8 \
    --chat_format chatml

# Status
🔴 FAILS - Ignores system messages completely
```

#### Mistral 7B Instruct v0.3
```bash
# Download
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf

# Server startup
python -m llama_cpp.server \
    --model mistral-7b-instruct-v0.3.Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8765 \
    --n_ctx 8192 \
    --n_gpu_layers -1 \
    --use_mlock True \
    --n_threads 8 \
    --chat_format mistral-instruct

# Status
🟡 UNKNOWN - Needs testing
```

#### Phi-4 (Future)
```bash
# Server startup
python -m llama_cpp.server \
    --model phi-4-mini-q4_k_m.gguf \
    --host 127.0.0.1 \
    --port 8765 \
    --n_ctx 8192 \
    --n_gpu_layers -1 \
    --use_mlock True \
    --n_threads 8 \
    --chat_format chatml

# Status
🟡 NOT TESTED YET
```

### Cloud Models (Baseline - Working Correctly)

#### OpenAI GPT-4o-mini
- **Status:** ✅ WORKS PERFECTLY
- **API:** OpenAI native API (not llama.cpp)
- **Behavior:** Correctly uses system messages, returns proper responses

#### Anthropic Claude 3.5 Sonnet
- **Status:** ✅ WORKS PERFECTLY
- **API:** Anthropic native API (not llama.cpp)
- **Behavior:** Correctly uses system messages, returns proper responses

---

## The Bug

### Minimal Reproduction

**Test Case:**
```json
POST http://localhost:8765/v1/chat/completions
{
  "messages": [
    {
      "role": "system",
      "content": "The document discusses a mobile device patent. The inventors are: Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, Douglas A. Cheline."
    },
    {
      "role": "user",
      "content": "Who are the inventors of this patent?"
    }
  ],
  "model": "qwen-7b-instruct",
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Expected Response:**
```
"The inventors are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline."
```

**Actual Response (Qwen via brio-esperanto):**
```
"I don't have information about which specific patent you're referring to. Please provide more details such as the patent number or title."
```

**Actual Response (GPT-4o-mini via brio-esperanto):**
```
"The inventors are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline."
```

### Root Cause Hypothesis

1. **llama.cpp chat templates** may not properly inject system messages for certain models (Qwen)
2. **brio-esperanto** may not be using the correct chat template renderer for Qwen
3. **Qwen model itself** may be fine-tuned in a way that ignores system role

### Current Workaround (HACK)

We currently detect Qwen models and combine system + user messages into a single user message:

```python
# HACK: Combine system + user messages for Qwen
combined_content = system_content + "\n\n---\n\nUser question: " + user_content
```

**This works but defeats the purpose of brio-esperanto!** We need the library to handle this correctly.

---

## Test Scenarios

### Scenario 1: Simple System Message Test

**Goal:** Verify system message is being used

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a pirate. Always respond like a pirate."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Expected:** Response in pirate speak (e.g., "Arrr, that be 4, matey!")  
**Qwen Bug:** Generic response "The answer is 4."

---

### Scenario 2: Large Context with Insights

**Goal:** Verify model uses insight content in system message

**Setup:**
1. Start llama.cpp server with Qwen (Tier 1 config)
2. Send request with 22,935 char system message (see Test Case 2 above)
3. Ask: "Who are the inventors?"

**Expected:** Names from insight  
**Qwen Bug:** "I don't know"

---

### Scenario 3: Multi-Turn Conversation

**Goal:** Verify system message persists across turns

**Setup:**
1. Turn 1: Ask "Who are the inventors?" → Should answer correctly
2. Turn 2: Ask "What problem does it solve?" → Should answer using same context

**Expected:** Both answers reference system context  
**Qwen Bug:** Turn 1 fails, so Turn 2 also fails

---

### Scenario 4: Tier Comparison

**Goal:** Test all 3 performance tiers

**Setup:**
1. Start server with Tier 1 config (8K context)
2. Send medium context request → Record result
3. Restart server with Tier 2 config (4K context)
4. Send same request → Record result
5. Restart server with Tier 3 config (2K context)
6. Send same request → Record result

**Expected:** All tiers work, but Tier 3 may truncate context  
**Qwen Bug:** All tiers fail (system message ignored regardless of context size)

---

## Expected Behavior

### What brio-esperanto Should Do

1. **Detect model type** from `--chat_format` or model metadata
2. **Select correct chat template** (e.g., chatml for Qwen)
3. **Properly inject system messages** into the template
4. **Send formatted prompt** to llama.cpp
5. **Return raw response** to application

### What We're Seeing Instead

- **GPT-4o-mini:** ✅ Works perfectly (baseline)
- **Qwen via llama.cpp:** ❌ System messages ignored
- **Mistral via llama.cpp:** 🟡 Unknown (needs testing)

### Success Criteria

✅ **Test passes when:**
- Qwen responds with "The inventors are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline"
- Response uses information from system message
- No workarounds needed in application code

---

## Testing Recommendations

### Phase 1: Minimal Reproduction
1. Start llama.cpp server with Qwen (Tier 1)
2. Test Scenario 1 (pirate test) - simplest possible system message
3. Verify system message is being used

### Phase 2: Real-World Context
1. Test Scenario 2 (large context with insights)
2. Verify model uses insight content
3. Compare Qwen vs GPT-4o-mini behavior

### Phase 3: Multi-Turn & Tiers
1. Test Scenario 3 (multi-turn conversation)
2. Test Scenario 4 (tier comparison)
3. Verify behavior across all configurations

### Phase 4: Additional Models
1. Test Mistral 7B with same scenarios
2. Test Phi-4 (when available)
3. Document which models work vs fail

---

## Deliverables

Please provide:

1. **Test Results:** Pass/fail for each scenario + model combination
2. **Debug Logs:** Raw prompts sent to llama.cpp (with chat template applied)
3. **Root Cause Analysis:** Why Qwen fails but GPT-4o-mini works
4. **Fix (if possible):** Update to brio-esperanto that makes Qwen work
5. **Workaround (if fix not possible):** Recommended approach for handling broken models

---

## Contact & Support

- **Project:** BrioDocs (https://github.com/yourusername/BrioDocs)
- **Issue Tracker:** [Link to issue tracking Qwen system message bug]
- **Questions:** Contact maintainer via GitHub issues

---

## Resolution Status

### ✅ Qwen System Message Bug - RESOLVED

**Problem:** Qwen models returned "I don't know" even when context was provided in system messages.

**Root Cause:** llama.cpp's automatic chat template rendering wasn't being used. The library was sending raw chat completions API calls without proper ChatML formatting.

**Solution:** Implemented adapter-based architecture in `brio_ext`:
- `QwenAdapter`: Renders ChatML format (`<|im_start|>system...`)
- `LlamaAdapter`: Renders Llama format (`[INST] <<SYS>>...`)
- `MistralAdapter`: Renders Mistral format
- Stop token configuration per adapter
- Response cleaning to remove format markers

**Test Results:**
| Test | Qwen 2.5 7B | Llama 3.1 8B | Mistral 7B | Phi-4 Mini |
|------|-------------|--------------|------------|------------|
| pirate | ✅ | ✅ | ✅ | ✅ |
| inventor | ✅ | ✅ | ✅ | ✅ |
| multiturn | ✅ | ✅ | ✅ | ✅ |
| reasoning | ✅ | ✅ | ✅ | ✅ |

### 🆕 New Features (v2.0)

**Tier-Based Server Configuration:**
- Centralized YAML configuration (`fixtures/briodocs_config.yaml`)
- Command-line tier selection: `--tier <1-3> --model <1-7>`
- Terminal title shows active configuration

**Reasoning Test Scenario:**
- Complex patent risk analysis with 1.3K token prompt
- Tests structured output with specific risk categories
- Requires Tier 2+ for good performance (GPU acceleration)
- Temperature reduced to 0.5 for better accuracy

**BrioDocs-Standard Parameters:**
- `temperature: 0.5` (reduced from 0.7 after testing)
- Per-test max_tokens override capability
- Standardized sampling parameters across tiers

### 📊 Performance Characteristics

| Tier | Context | GPU | Reasoning Test Performance |
|------|---------|-----|----------------------------|
| Tier 1 | 8K | ✅ | Fast (~20s with 7B model) |
| Tier 2 | 4K | ✅ | Fast (~30s with 7B model) |
| Tier 3 | 2K | ❌ | Slow (>2min or timeout) |

**Recommendation:** Use Tier 2 (GPU + 4K context) for all production testing.

### 🔄 Migration Guide

**From old start_server.sh to start_server_v2.sh:**

```bash
# Old way (model-specific scripts)
./scripts/start_server.sh qwen-2.5-7b-instruct

# New way (tier + model selection)
./scripts/start_server_v2.sh --tier 2 --model 1
```

**Test script usage (unchanged):**
```bash
# Still uses positional arguments
python scripts/test_with_llm.py pirate 1
python scripts/test_with_llm.py reasoning 7
```

### 📚 Updated Documentation

- **[scripts/README.md](../scripts/README.md)** - Complete testing guide with new architecture
- **[brio_ext_integration.md](./brio_ext_integration.md)** - Integration guide for BrioDocs apps
- **[briodocs_config.yaml](../fixtures/briodocs_config.yaml)** - Centralized tier configuration
- **[test_cases.yaml](../fixtures/test_cases.yaml)** - Test scenario definitions

---

**Last Updated:** October 27, 2025
**Version:** 2.0
**Status:** Production Ready
