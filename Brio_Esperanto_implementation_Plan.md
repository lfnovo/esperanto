# Implementation Plan – BrioDocs × Esperanto (Library Integration)

⸻

## 0. Guiding Principles
- Stay within the upstream architecture: no API server, no upstream patches. BrioDocs consumes a pure Python library.
- Preserve Esperanto’s provider compatibility while guaranteeing BrioDocs prompt templating (`<out>...</out>`) across all model families.
- Default to reversible, well-documented changes on the long-lived `brio/providers` branch; gate releases with `bridocs-vX.Y` tags.

⸻

## 1. Repository & Branch Workflow
- Repo: `github.com/dcheline/Brio-Esperanto` (fork of `lfnovo/esperanto`).
- Remotes
  - `origin`: fork
  - `upstream`: `https://github.com/lfnovo/esperanto.git`
- Branches
  - Long-lived: `brio/providers`
  - Feature: `feat/<adapter-family>`, `feat/factory-integrations`, `test/goldens`
  - Release: `release/bridocs-vX.Y` → tag `bridocs-vX.Y`
- Sync cadence
  ```bash
  git checkout brio/providers
  git fetch upstream
  git rebase upstream/main
  git push --force-with-lease origin brio/providers
  ```

⸻

## 2. Module Layout (`brio_ext/`)
```
brio_ext/
  __init__.py
  adapters/
    __init__.py
    qwen_adapter.py
    llama_adapter.py
    mistral_adapter.py
    gemma_adapter.py
    phi_adapter.py
  registry.py
  renderer.py
  providers/
    llamacpp_provider.py      # completions with rendered prompt
    hf_local_provider.py      # transformers/TGI completions with rendered prompt
  factory.py                  # BrioAIFactory + register_with_factory()
  telemetry.py                # helpers for stop-hit, truncation stats (optional)
  tests/
    test_adapters_unit.py
    test_renderer_unit.py
    test_factory_integration.py
    test_golden_snapshots.py
  README.md
```

⸻

## 3. Core Logic
**Adapters (`brio_ext/adapters/`)**
- Implement `ChatAdapter` base class (methods `can_handle(model_id)` and `render(messages)`).
- Families: Qwen, Llama, Mistral, Gemma, Phi. Each `render` returns either `{"prompt": str, "stop": [...]}` or `{"messages": [...], "stop": [...]}` and always appends `<out>` preamble when producing prompts.
- Ensure multi-system messages collapse cleanly; keep stop lists family-specific (e.g., Qwen adds `<|im_end|>`).

**Registry (`brio_ext/registry.py`)**
- Instantiate adapters once: `ADAPTERS = [QwenAdapter(), ...]`.
- Provide `get_adapter(model_id)` returning a matching instance or `None`.

**Renderer (`brio_ext/renderer.py`)**
- Export `render_for_model(model_id, messages, provider)`.
- Routing rules:
  - Providers that template reliably (`openai`, `anthropic`, `grok`, `ollama`) bypass adapters unless a matching adapter exists.
  - Local engines (`llamacpp`, `hf_local`) or adapter-matched models → return rendered prompt.
  - Fallback: pass-through messages with enforced `["</out>"]` stop.
- Helper `_merge_stops(existing, override)` ensures `</out>` is always included and duplicates removed.

⸻

## 4. Factory Integration
**BrioAIFactory (`brio_ext/factory.py`)**
- Subclass Esperanto’s `AIFactory`.
- Override `chat(model_id, provider, messages, **gen)`:
  1. Call `render_for_model(...)`.
  2. Merge stops: adapter stop list + caller-provided stop values (if any).
  3. If renderer returned `messages`, delegate to `super().chat(...)`.
  4. If renderer returned `prompt`, delegate to `super().completions(...)`.
- Override `completions(...)` to enforce `</out>` even when called directly.

**Instance Hook**
- Provide `register_with_factory(factory)` to monkey-patch a vanilla `AIFactory` (useful for incremental rollout or tests).

**Provider Shims (`brio_ext/providers/`)**
- `llamacpp_provider.py`: wraps Esperanto llama-cpp client ensuring `/v1/completions` usage with rendered prompt.
- `hf_local_provider.py`: handles local Hugging Face / TGI calls with prompt + stop list.
- Only override behaviour where Esperanto’s default provider logic cannot accept rendered prompts cleanly.

⸻

## 5. BrioDocs Integration
- Replace factory import: `from brio_ext.factory import BrioAIFactory`.
- Instantiate with same configuration currently passed to `AIFactory`.
- No change to BrioDocs payload structure:
  ```python
  response = factory.chat(
      model_id=model_id,
      provider=provider,
      messages=messages,
      temperature=0.25,
      top_p=0.8,
      repetition_penalty=1.0,
      max_tokens=dynamic_cap,
      stop=["</out>"],
  )
  ```
- Optional feature flag `BRIO_USE_BRIO_FACTORY` to toggle during rollout.

⸻

## 6. Testing Strategy
**Unit Tests**
- `test_adapters_unit.py`: each adapter returns correct structure, stop list includes `</out>`, handles multiple system/user messages.
- `test_renderer_unit.py`: verify routing decisions by provider/model combos (messages vs prompt, stop merging).

**Integration Tests**
- `test_factory_integration.py`: instantiate `BrioAIFactory` with mocked provider clients:
  - `provider="llamacpp", model="qwen2.5-7b-instruct"`
  - `provider="ollama", model="mistral-7b-instruct"`
  - `provider="openai", model="phi-4-mini-instruct"`
- Assert requests passed to provider include `<out>` preface and merged stops; assert stop hit.

**Golden Snapshots**
- Store `<out>...` body for 5 legal fixtures × 3 tools × 3 families (Qwen/Llama/Mistral).
- Diff on regressions: defined terms preserved (regex for quoted Caps), numbering intact, no prefatory text.

**CI**
- GitHub Actions workflow on `brio/providers` and `release/*`: `pytest brio_ext/tests -q`.
- Allow override to run local smoke tests manually with real engines (document commands in README).

⸻

## 7. Telemetry & Performance
- Track per-request metadata (within BrioDocs app): `tool`, `model_id`, `provider`, prompt/output tokens, `stop_hit`, `truncated`, `duration_ms`, `exception`.
- Target local latency `<1s` median for ~512 token input on supported laptops.
- Implement debug toggle to expose final rendered prompt (first 2 KB) for developer diagnostics; default off.

⸻

## 8. Migration Checklist
1. Scaffold `brio_ext/` structure and add initial adapters/renderer/factory implementations.
2. Commit on `feat/brio-factory` → PR into `brio/providers`.
3. Update BrioDocs application to import `BrioAIFactory`; guard with feature flag.
4. Run unit tests + integration tests; perform local smoke test per provider family.
5. Merge to `brio/providers`, tag `bridocs-vX.Y`, update BrioDocs dependency (git submodule or pinned install).
6. Roll out behind feature flag; monitor telemetry for stop-hit ≥ 99%, truncations < 1%.

⸻

## 9. Risks & Mitigations
- **Double templating**: Avoided by routing completions whenever adapters render prompts.
- **Stop leakage**: `_merge_stops` guarantees `</out>` and adapter-specific stops persist.
- **llama-cpp chat API quirks**: Always use `/v1/completions` with pre-rendered prompt.
- **Upstream divergence**: Keep upstream untouched; rebase weekly; isolate Brio files under `brio_ext/`.
- **Testing brittleness**: Golden tests scoped to `<out>` body; regenerate fixtures via helper script when prompt contract changes.

⸻

## 10. Next Steps
- Implement `ChatAdapter` base + Qwen/Llama adapters first to unblock llama-cpp path.
- Build out `BrioAIFactory` with stop merging.
- Add unit tests and confirm integration under mock providers.
- Coordinate with BrioDocs team for import swap and feature-flag rollout.

