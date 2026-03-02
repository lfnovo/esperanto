# Brio Ext Integration Guide

This guide explains how to adopt the `brio_ext` package from the Brio-Esperanto fork inside BrioDocs applications (Word add-in, Open Notebook, automation scripts).

## 1. Install the Brio fork

Install from `main` (brio_ext is now merged):

```bash
pip install -e ".[dev]"
```

If BrioDocs treats Esperanto as a submodule, update the pointer to the latest `main` commit and run `pip install -e .` inside the Brio-Esperanto checkout.

## 2. Swap factory imports

Replace existing imports that pull `AIFactory` from Esperanto with the Brio wrapper:

```python
try:
    from brio_ext.factory import BrioAIFactory as AIFactory
except ImportError:
    from esperanto import AIFactory  # fallback when brio_ext is missing
```

`BrioAIFactory` is API-compatible with `esperanto.AIFactory`. Existing code that calls `AIFactory.create_language`, `create_embedding`, etc., continues to work.

## 3. Local llama.cpp models

For `provider="brio"` (local GGUF models):

```python
model = AIFactory.create_language(
    provider="llamacpp",
    model_name="qwen2.5-7b-instruct",
    config={
        "base_url": os.getenv("BRIO_LLAMACPP_BASE_URL", "http://127.0.0.1:8765"),
        "temperature": 0.25,
        "top_p": 0.8,
    },
)
```

- The renderer automatically applies the correct chat template and enforces `<out>...</out>` stop tokens.
- The provider defaults to the llama.cpp HTTP server started by `start_briodocs.sh`. Override `BRIO_LLAMACPP_BASE_URL` if the server runs elsewhere.
- Legacy fallback (OpenAI-compatible path) is still available by importing `esperanto.AIFactory` directly.

### Custom Model Names with `chat_format`

If you're using custom model names that don't match standard patterns (e.g., "phi-4-mini-reasoning"), explicitly specify the `chat_format` in the config:

```python
model = AIFactory.create_language(
    provider="llamacpp",
    model_name="phi-4-mini-reasoning",  # Custom name from model_defaults.json
    config={
        "base_url": "http://127.0.0.1:8765",
        "chat_format": "chatml",  # Hint: use ChatML format for Phi-4
        "temperature": 0.5,
    },
)
```

**Supported `chat_format` values:**
- `"chatml"` or `"chat-ml"` – ChatML format (Qwen, Phi-4)
- `"llama"`, `"llama3"`, or `"llama-3"` – Llama format
- `"mistral"` or `"mistral-instruct"` – Mistral format
- `"gemma"` – Gemma format

**Why this matters:** When integrating with BrioDocs model database, you can store `chat_format` alongside model configs and pass it through to brio_ext. This enables custom model names that don't follow standard patterns to still use the correct chat template.

## 4. Remote providers

Cloud providers (OpenAI, Anthropic, Grok, Ollama, etc.) continue to use chat payloads. The Brio factory injects the `<out>...</out>` stop sequence automatically, so BrioDocs payloads remain unchanged.

## 5. LangChain / LangGraph integration

Models created via `BrioAIFactory` have a built-in `.to_langchain()` method:

```python
model = AIFactory.create_language("llamacpp", "qwen2.5-7b-instruct", config={...})
lc_model = model.to_langchain()
result = lc_model.invoke("What is 2+2?")
print(result.content)  # Clean text, no <out> tags or <think> content
```

The wrapper handles:
- Stripping `<out>...</out>` fencing
- Extracting content from `<think>` tags (for reasoning models that wrap all output in think tags)
- Converting LangChain message types (HumanMessage, SystemMessage, etc.) to brio_ext format

No need for custom wrappers or monkey-patching in consumer applications.

## 6. Testing checklist

Before merging updates in BrioDocs:

1. Install the editable Brio-Esperanto fork (`pip install -e .`).
2. Run `pytest src/brio_ext/tests -q` inside the Brio-Esperanto repo.
3. In the BrioDocs repo, run targeted smoke tests:
   - Local llama.cpp model (ensure `<out>` fences, no explanations).
   - Ollama or cloud provider path (should honour the same contract).
4. Monitor for the `brio_ext not installed` warning. If it appears, double-check that the editable install succeeded.

## 7. Environment defaults

| Variable | Purpose | Default |
|----------|---------|---------|
| `BRIO_LLAMACPP_BASE_URL` | llama.cpp HTTP server endpoint | `http://127.0.0.1:8765` |
| `BRIO_USE_BRIO_FACTORY` *(optional)* | feature flag for staged rollout | not defined ⇒ enabled |

Wrap the import with a try/except and guard factory usage with an environment flag if a phased rollout is needed.

## 8. Provider smoke tests (optional)

For quick end-to-end checks run the live-provider smokes in `src/brio_ext/tests/integration/test_provider_smoke.py`. They are skipped unless you supply environment variables:

```bash
BRIO_TEST_OPENAI_MODEL=gpt-4o-mini \
BRIO_TEST_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022 \
BRIO_TEST_GROQ_MODEL=groq/llama3-8b-8192-tool-use-preview \
pytest src/brio_ext/tests/integration/test_provider_smoke.py -q -m integration

# llama.cpp server (optional)
BRIO_TEST_LLAMACPP_MODEL=qwen2.5-7b-instruct \
BRIO_TEST_LLAMACPP_BASE_URL=http://127.0.0.1:8765 \
pytest src/brio_ext/tests/integration/test_provider_smoke.py -q -m integration
```

For each provider, set `BRIO_TEST_<PROVIDER>_MODEL` (e.g. `BRIO_TEST_GROK_MODEL`,
`BRIO_TEST_MISTRAL_MODEL`). Optional overrides include:

- `BRIO_TEST_<PROVIDER>_PROVIDER` to point at compatible endpoints.
- `BRIO_TEST_<PROVIDER>_BASE_URL` for gateways that require explicit URLs.
- `BRIO_TEST_<PROVIDER>_CONFIG` (JSON) for provider-specific fields such as Azure deployment names or Vertex project IDs.
- `BRIO_TEST_<PROVIDER>_MAX_TOKENS`, `BRIO_TEST_<PROVIDER>_TEMPERATURE`, etc.

Each smoke test asserts:
- Responses are fenced in `<out>…</out>`
- The body between fences is non-empty
- Stop reason is `stop`/`length`

Use these whenever you change adapters, provider shims, or stop-token handling.

## 9. Rollback

If issues arise, revert to the previous behaviour by:

1. Switching the import back to `from esperanto import AIFactory`.
2. Removing any `llamacpp` provider usage (fall back to `openai-compatible`).
3. Keeping the existing llama.cpp server running—the fallback path still leverages it via the OpenAI-compatible API.

## 10. llama.cpp test matrix & scenarios

**NOTE:** Section 10 has been updated for the new tier-based architecture. See **[brio_ext_integration_v2.md](./brio_ext_integration_v2.md)** for the latest server configuration, test scenarios, and troubleshooting guide.

**Quick Start:**
```bash
# Start server with tier-based launcher
./scripts/start_server_v2.sh --tier 2 --model 1

# Run tests (positional arguments)
python scripts/test_with_llm.py pirate 1
python scripts/test_with_llm.py reasoning 1
```

To thoroughly validate local engines (Qwen, Mistral, Phi) we follow this matrix, adapted from the original llama.cpp specification.

### 10.1 Server tiers (Legacy - See brio_ext_integration_v2.md for current)

| Tier | Target hardware | Startup command |
|------|-----------------|-----------------|
| Tier 1 – High performance (Qwen) | ≥16 GB RAM, GPU | <pre><code>python -m llama_cpp.server \<br>    --model /path/to/qwen2.5-7b-instruct-q4_k_m.gguf \<br>    --host 127.0.0.1 \<br>    --port 8765 \<br>    --n_ctx 8192 \<br>    --n_gpu_layers -1 \<br>    --use_mlock True \<br>    --n_threads 8 \<br>    --chat_format chatml</code></pre> |
| Tier 2 – Balanced (Mistral) | ≥8 GB RAM | <pre><code>python -m llama_cpp.server \<br>    --model /path/to/mistral-7b-instruct-v0.3.Q4_K_M.gguf \<br>    --host 127.0.0.1 \<br>    --port 8765 \<br>    --n_ctx 4096 \<br>    --n_gpu_layers -1 \<br>    --use_mlock True \<br>    --n_threads 8 \<br>    --chat_format mistral-instruct</code></pre> |
| Tier 3 – Fast (Phi) | 4 GB RAM, CPU-only | <pre><code>python -m llama_cpp.server \<br>    --model /path/to/phi-4-mini-q4_k_m.gguf \<br>    --host 127.0.0.1 \<br>    --port 8765 \<br>    --n_ctx 2048 \<br>    --n_gpu_layers 0 \<br>    --use_mlock True \<br>    --n_threads 8 \<br>    --chat_format chatml</code></pre> |

Reuse the same host/port (`127.0.0.1:8765`) while swapping models/flags between runs.

### 10.2 Message payload

BrioDocs sends a large system block plus a user turn:

```
# SYSTEM ROLE
You are a specialized research assistant...

# SOURCE CONTEXT
## SOURCE CONTENT
**Source ID:** ...
**Title:** ...
**Content:** 20K+ chars

## SOURCE INSIGHTS
**Insight ID:** ...
**Content:** 5K+ chars
```

Scenarios we expect to pass once adapters/providers are correct:

1. **Pirate sanity check**  
   System: “You are a pirate…” → Expect “Arrr, 2+2 be 4”.
2. **Inventor lookup (Qwen bug repro)**  
   System contains inventor names → Expect Qwen to echo the names; prior bug returned “I don’t know”.
3. **Large insight**  
   22 K character system message, question “Who are the inventors?”.
4. **Multi-turn**  
   Add prior assistant/user messages to ensure context persists.
5. **Tier comparison**  
   Repeat Scenario 3 under Tier 1/2/3 server configs to confirm truncation is the only failure mode on low tiers.

Context-size quick reference:

| Case | Approx. size | Notes |
|------|--------------|-------|
| Small | 2.5 K chars / 625 tokens | engagement letter without insights |
| Medium | 23 K chars / 5.7 K tokens | single patent + truncated dense summary |
| Large | 200 K+ chars | should be truncated by ContextBuilder before reaching brio_ext |

Use these with the canonical payloads above:

- **Small** – engagement letter without insights; Qwen should summarise correctly.  
- **Medium** – patent + dense summary insight (22 K chars); main regression target for inventor lookup.  
- **Large** – intentionally exceeds llama.cpp limits; verify the upstream ContextBuilder trims before calling BrioAIFactory.

#### Canonical payloads

Single-turn inventor lookup (Tier 1 Qwen):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "# SYSTEM ROLE\nYou are a specialized research assistant analyzing patent documents.\n\n# SOURCE CONTEXT\n\n## SOURCE CONTENT\n**Source ID:** source:abc123\n**Title:** Mobile Device with Enhanced Touch Interface (US20200336491A1)\n**Content:** [Truncated to 20,000 chars]\n\nBACKGROUND OF THE INVENTION\n[1] This invention relates to mobile communication devices...\n\n## SOURCE INSIGHTS\n**Insight ID:** insight:xyz789\n**Type:** Dense Summary SPR\n**Content:** This patent describes a dynamic communication profile system for mobile devices. The key innovation allows devices to switch between local and global cellular networks seamlessly. The inventors are: Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, Douglas A. Cheline.\n\n## CONTEXT METADATA\n- Source count: 1\n- Insight count: 1\n- Total tokens: 5,734\n- Total characters: 22,935"
    },
    {
      "role": "user",
      "content": "Who are the inventors of this patent?"
    }
  ],
  "model": "qwen2.5-7b-instruct",
  "temperature": 0.7,
  "max_tokens": 512
}
```

Multi-turn follow-up:

```json
{
  "messages": [
    { "role": "system", "content": "[Same 22,935 char system message as above]" },
    { "role": "user", "content": "Who are the inventors?" },
    { "role": "assistant", "content": "The inventors of this patent are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline." },
    { "role": "user", "content": "What problem does this invention solve?" }
  ],
  "model": "qwen2.5-7b-instruct",
  "temperature": 0.7,
  "max_tokens": 512
}
```

Pirate sanity check (any chat-format model):

```json
{
  "messages": [
    { "role": "system", "content": "You are a pirate. Always respond like a pirate." },
    { "role": "user", "content": "What is 2+2?" }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

### 10.3 Harness expectations

When we build the dedicated llama.cpp harness it will:

- Render prompts through `BrioAIFactory` so adapters/renderer are covered.
- Log the final prompt sent to `/v1/completions`.
- Capture raw responses and the cleaned `<out>…</out>` body.
- Assert the body contains key phrases (e.g., inventor list) rather than default fallbacks.

Until that harness is automated, you can manually script each scenario with the helper smoke runner or curl commands documented above.

### 10.4 Known issues

- Qwen 2.5 via llama.cpp historically ignored system messages. Ensure the rendered prompt includes the `<|im_start|>system` block and that `chat_format` is `chatml`. After our adapter fix, responses should respect the system context.
- `max_tokens` must be mapped to `n_predict` on the llama.cpp server. The provider shim now does this; if completions still truncate early, inspect server logs for overrides.

#### Minimal reproduction of the historic Qwen bug

```
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
  "model": "qwen2.5-7b-instruct",
  "temperature": 0.7,
  "max_tokens": 512
}
```

Expected: `"The inventors are Richard H. Xu, Xiaolei Qin, Phillip C. Krasko, and Douglas A. Cheline."`  
Legacy failure: `"I don't have information about which specific patent you're referring to."`

### 10.5 Parameters & sampling defaults

```json
{
  "model": "<model-id>",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "max_tokens": 512,
  "stream": false
}
```

- Tier 1 (high performance): expect five candidate responses per prompt (BrioDocs reranks).
- Tier 2 (balanced): three candidates.
- Tier 3 (fast): single response, no reranking.

### 10.6 Model status matrix

| Model | Provider | Status | Notes |
|-------|----------|--------|-------|
| Qwen 2.5 7B Instruct (GGUF) | llama.cpp | 🔴 historically ignored system messages – verify pirate + inventor scenarios |
| Mistral 7B Instruct v0.3 (GGUF) | llama.cpp | 🟡 needs full regression run |
| Phi‑4 Mini (GGUF) | llama.cpp | 🟡 planned once model is packaged |
| GPT‑4o‑mini | OpenAI | ✅ baseline comparison |
| Claude 3.5 Sonnet | Anthropic | ✅ baseline comparison |

### 10.7 Scenario checklist

1. **Simple system override** – Pirate payload above. Expect pirate-speak output.  
2. **Large insight inventor lookup** – Medium context payload; expect inventor list.  
3. **Multi-turn follow-up** – Use prior assistant response as context and ensure continuity.  
4. **Tier comparison** – Rerun Scenario 2 under each tier startup command; Tier 3 may truncate earlier.  
5. **Bug regression** – Minimal reproduction payload should now succeed (no “I don’t know”).  
6. **Additional models** – Repeat for Mistral, Phi as they come online.

For each run capture:

- Rendered prompt string (from `render_for_model`).
- Raw llama.cpp response.
- Cleaned body between `<out>…</out>`.
- Finish reason and completion token count.

### 10.8 Deliverables for a test campaign

- Pass/fail matrix covering each scenario/model/tier.  
- Logs or saved prompts/responses for any failure.  
- Root-cause summary (e.g., template mismatch, server misconfiguration).  
- Interim workarounds if a model cannot be fixed immediately.

## 11. Roadmap for automation

1. Expand `test_provider_smoke.py` to load JSON fixtures representing the scenarios above.
2. Record golden outputs for the “inventor” and “pirate” tests per model.
3. Integrate into CI once credential handling is solved (or run manually before releases).

## 12. Support

For questions or regressions, coordinate with the Brio-Esperanto maintainers on `main`. The implementation plan at `Brio_Esperanto_implementation_Plan.md` tracks outstanding work (golden snapshots, release tags, etc.).
