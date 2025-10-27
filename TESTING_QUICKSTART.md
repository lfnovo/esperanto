# Manual Testing Quick Start

Simple environment to validate brio_ext pipeline with real models.

## 🚀 3-Step Quickstart

### 1. Download Qwen Model

```bash
./scripts/download_models.sh qwen-2.5-7b-instruct
```

Takes ~5 minutes to download 4.4GB file.

### 2. Start Server (Terminal 1)

```bash
conda activate briodocs
./scripts/start_server.sh qwen-2.5-7b-instruct
```

Wait for "Server ready" message.

### 3. Run Test (Terminal 2)

```bash
conda activate briodocs
python scripts/test_with_llm.py pirate qwen2.5-7b-instruct
```

## What You Get

**Full pipeline visibility:**
- ✅ See which adapter is selected (QwenAdapter, LlamaAdapter, etc.)
- ✅ See the rendered prompt sent to llama.cpp
- ✅ See stop tokens being merged
- ✅ See raw response from model
- ✅ See validation results

**Test scenarios:**
- `pirate` - Simple system message test
- `inventor` - The KEY test for Qwen system message bug fix
- `multiturn` - Multi-turn conversation test
- `all` - Run all scenarios

## Key Tests

```bash
# Test if system messages work (the bug we fixed)
python scripts/test_with_llm.py inventor qwen2.5-7b-instruct

# Compare against OpenAI baseline
export OPENAI_API_KEY=sk-...
python scripts/test_with_llm.py inventor gpt-4o-mini

# Test all scenarios
python scripts/test_with_llm.py all qwen2.5-7b-instruct
```

## Expected Results

✅ **If brio_ext is working correctly:**
- Pirate test: Model responds in pirate speak
- Inventor test: Model returns all 4 inventor names
- All responses fenced in `<out>...</out>`

❌ **If something is broken:**
- Debug output shows which step failed
- Can see adapter selection, rendering, stops, provider calls

## Full Documentation

See [scripts/README.md](scripts/README.md) for:
- All available models
- Detailed test explanations
- Troubleshooting guide
- How to add more scenarios

## What This Validates

This tests the entire brio_ext pipeline:
1. Adapter selection (QwenAdapter for Qwen models)
2. Prompt rendering (ChatML format with `<|im_start|>` tags)
3. Stop token merging (`</out>` + `<|im_end|>`)
4. Provider integration (llamacpp_provider)
5. **System message handling** (the KEY bug fix)
6. Output fencing (`<out>...</out>` tags)

This is exactly what BrioDocs will use in production!
