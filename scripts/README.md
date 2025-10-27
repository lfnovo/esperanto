# Manual Testing Scripts for brio_ext

Simple scripts to validate brio_ext pipeline with real models.

## Quick Start

### 1. Download a Model (One-Time Setup)

Start with Qwen 2.5 7B Instruct (~4.4GB):

```bash
./scripts/download_models.sh qwen-2.5-7b-instruct
```

This downloads to `models/qwen2.5-7b-instruct-q4_k_m.gguf`

### 2. Start the Server (Terminal 1)

```bash
conda activate briodocs
./scripts/start_server.sh qwen-2.5-7b-instruct
```

Wait for "Server ready" or "Uvicorn running" message.

### 3. Run Tests (Terminal 2)

```bash
conda activate briodocs
python scripts/test_with_llm.py pirate qwen2.5-7b-instruct
```

## Usage

### Test Individual Scenarios

```bash
# Pirate test (simple system message)
python scripts/test_with_llm.py pirate qwen2.5-7b-instruct

# Inventor test (the KEY test for Qwen system message bug)
python scripts/test_with_llm.py inventor qwen2.5-7b-instruct

# Multi-turn conversation
python scripts/test_with_llm.py multiturn qwen2.5-7b-instruct
```

### Test Against OpenAI Baseline

```bash
# Make sure OPENAI_API_KEY is set
export OPENAI_API_KEY=sk-...

# Test pirate scenario with GPT-4o-mini
python scripts/test_with_llm.py pirate gpt-4o-mini

# Test inventor scenario with GPT-4o-mini
python scripts/test_with_llm.py inventor gpt-4o-mini
```

### Run All Scenarios

```bash
python scripts/test_with_llm.py all qwen2.5-7b-instruct
```

## What You'll See

The test runner shows **full pipeline visibility**:

1. **Input to brio_ext**: The messages being sent
2. **brio_ext Processing**: Debug output showing:
   - Which adapter was selected (e.g., QwenAdapter)
   - Rendering mode (PROMPT vs MESSAGES)
   - Stop tokens being added
   - Prompt length
3. **Calling Model**: What's being sent to the provider
4. **Raw Response**: Exactly what came back
5. **Response Metadata**: Tokens, finish reason, etc.
6. **Parsed Content**: Extracted from `<out>...</out>` tags
7. **Validation**: Pass/fail checks for expected behavior
8. **Overall Result**: Test passed or failed

### Example Output

```
================================================================================
SCENARIO: inventor
MODEL: qwen2.5-7b-instruct
PROVIDER: llamacpp
BASE_URL: http://127.0.0.1:8765
================================================================================

[1. INPUT TO BRIO_EXT]
────────────────────────────────────────────────────────────────────────────
Messages being sent:
1. SYSTEM:
   You are a specialized research assistant...
   The inventors are: Richard H. Xu, Xiaolei Qin, Phillip C. Krasko...

2. USER:
   Who are the inventors of this patent?

[2. BRIO_EXT PROCESSING]
────────────────────────────────────────────────────────────────────────────
[RENDERER] model_id=qwen2.5-7b-instruct, provider=llamacpp
[RENDERER] adapter=QwenAdapter
[QwenAdapter] Rendering 2 messages
[QwenAdapter] Generated prompt: 2847 chars
[QwenAdapter] Stop tokens: ['</out>', '<|im_end|>']
[RENDERER] mode=PROMPT (completions)
[RENDERER] prompt_length=2847 chars
[RENDERER] stops=['</out>', '<|im_end|>']

[3. CALLING MODEL]
────────────────────────────────────────────────────────────────────────────
[LlamaCppProvider] Calling /v1/completions
[LlamaCppProvider] base_url=http://127.0.0.1:8765
[LlamaCppProvider] prompt_length=2847 chars
[LlamaCppProvider] stop_tokens=['</out>', '<|im_end|>']
[LlamaCppProvider] max_tokens=512

[4. RAW RESPONSE FROM PROVIDER]
────────────────────────────────────────────────────────────────────────────
The inventors are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline.</out>

[5. RESPONSE METADATA]
────────────────────────────────────────────────────────────────────────────
Model: qwen2.5-7b-instruct
Provider: llamacpp
Finish reason: stop
Prompt tokens: 711
Completion tokens: 28
Total tokens: 739

[6. PARSED CONTENT]
────────────────────────────────────────────────────────────────────────────
Extracted from <out>...</out> tags:
The inventors are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline.

[7. VALIDATION]
────────────────────────────────────────────────────────────────────────────
✓ Response properly fenced in <out>...</out>

Checking for expected content (4 phrases):
  ✓ Found: 'Richard H. Xu'
  ✓ Found: 'Xiaolei Qin'
  ✓ Found: 'Phillip C. Krasko'
  ✓ Found: 'Douglas A. Cheline'

✓ Stop reason: stop

[8. OVERALL RESULT]
────────────────────────────────────────────────────────────────────────────
✅ TEST PASSED
```

## Available Models

Download any of these:

```bash
# Start with this one
./scripts/download_models.sh qwen-2.5-7b-instruct    # 4.4GB

# Other models
./scripts/download_models.sh qwen-2.5-3b-instruct    # 2.0GB
./scripts/download_models.sh llama-3.1-8b-instruct   # 4.9GB
./scripts/download_models.sh llama-3.2-3b-instruct   # 2.0GB
./scripts/download_models.sh mistral-7b-instruct     # 4.4GB
./scripts/download_models.sh phi-4-mini              # 2.7GB

# Download everything (~20GB)
./scripts/download_models.sh all
```

To test a different model, switch the server:

```bash
# Terminal 1: Stop current server
./scripts/stop_server.sh

# Start new model
./scripts/start_server.sh mistral-7b-instruct

# Terminal 2: Test with new model
python scripts/test_with_llm.py all mistral-7b-instruct
```

## Test Scenarios Explained

### pirate
- **Purpose**: Simple system message test
- **What it tests**: Basic adapter selection and system message handling
- **Expected**: Model responds in pirate speak (e.g., "Arrr, that be 4!")
- **If it fails**: System messages are being ignored or adapter not selected

### inventor
- **Purpose**: Medium context with real BrioDocs-style payload
- **What it tests**: The Qwen system message bug fix
- **Expected**: Model extracts inventor names from system context
- **If it fails**: System messages not working (this was the original bug)

### multiturn
- **Purpose**: Multi-turn conversation
- **What it tests**: Conversation history handling
- **Expected**: Model uses previous turns to answer follow-up question
- **If it fails**: Context not being maintained across turns

## Troubleshooting

### Server won't start - port in use

```bash
./scripts/stop_server.sh
# Or manually:
kill $(lsof -t -i:8765)
```

### Model file not found

```bash
# Check if it exists
ls -lh models/

# Re-download if needed
./scripts/download_models.sh qwen-2.5-7b-instruct
```

### OpenAI tests fail

```bash
# Make sure API key is set
echo $OPENAI_API_KEY

# Set it if needed
export OPENAI_API_KEY=sk-...
```

### Import errors

```bash
# Make sure you're in the conda environment
conda activate briodocs

# And that brio_ext can be imported
python -c "from brio_ext.factory import BrioAIFactory; print('OK')"
```

## What This Tests

This manual testing environment validates:

1. **Adapter Selection**: Does brio_ext pick the right adapter?
2. **Prompt Rendering**: Is the ChatML template correct?
3. **Stop Token Merging**: Are `</out>` and adapter stops combined?
4. **Provider Integration**: Does llamacpp_provider work correctly?
5. **System Message Handling**: The KEY question - does Qwen respect system messages?
6. **Output Fencing**: Do responses come back in `<out>...</out>` tags?
7. **Baseline Comparison**: Does OpenAI passthrough work?

## Next Steps

Once you validate Qwen 2.5 7B works:
1. Test other models (Mistral, Llama, Phi-4)
2. Add more test scenarios if needed
3. Integrate into BrioDocs application
4. Consider automating these as pytest integration tests

## Debug Mode

Debug mode is **always enabled** in the test script (`BRIO_DEBUG=1`). This shows:
- Renderer decisions
- Adapter selection
- Prompt generation
- Stop token merging
- Provider calls

This is the key to understanding if brio_ext is working correctly!
