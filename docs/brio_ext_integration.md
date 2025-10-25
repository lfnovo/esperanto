# Brio Ext Integration Guide

This guide explains how to adopt the `brio_ext` package from the Brio-Esperanto fork inside BrioDocs applications (Word add-in, Open Notebook, automation scripts).

## 1. Install the Brio fork

During development, install the fork directly from the `brio/providers` branch:

```bash
pip install -e ".[dev]"
# or, to pin to a specific commit:
# pip install "esperanto @ git+https://github.com/dcheline/esperanto.git@<commit>"
```

If BrioDocs treats Esperanto as a submodule, update the pointer to the commit that contains `brio_ext` and run `pip install -e .` inside the Brio-Esperanto checkout.

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

## 4. Remote providers

Cloud providers (OpenAI, Anthropic, Grok, Ollama, etc.) continue to use chat payloads. The Brio factory injects the `<out>...</out>` stop sequence automatically, so BrioDocs payloads remain unchanged.

## 5. Testing checklist

Before merging updates in BrioDocs:

1. Install the editable Brio-Esperanto fork (`pip install -e .`).
2. Run `pytest src/brio_ext/tests -q` inside the Brio-Esperanto repo.
3. In the BrioDocs repo, run targeted smoke tests:
   - Local llama.cpp model (ensure `<out>` fences, no explanations).
   - Ollama or cloud provider path (should honour the same contract).
4. Monitor for the `brio_ext not installed` warning. If it appears, double-check that the editable install succeeded.

## 6. Environment defaults

| Variable | Purpose | Default |
|----------|---------|---------|
| `BRIO_LLAMACPP_BASE_URL` | llama.cpp HTTP server endpoint | `http://127.0.0.1:8765` |
| `BRIO_USE_BRIO_FACTORY` *(optional)* | feature flag for staged rollout | not defined ⇒ enabled |

Wrap the import with a try/except and guard factory usage with an environment flag if a phased rollout is needed.

## 7. Rollback

If issues arise, revert to the previous behaviour by:

1. Switching the import back to `from esperanto import AIFactory`.
2. Removing any `llamacpp` provider usage (fall back to `openai-compatible`).
3. Keeping the existing llama.cpp server running—the fallback path still leverages it via the OpenAI-compatible API.

## 8. Support

For questions or regressions, coordinate with the Brio-Esperanto maintainers on the `brio/providers` branch. The implementation plan at `Brio_Esperanto_implementation_Plan.md` tracks outstanding work (golden snapshots, release tags, etc.).
