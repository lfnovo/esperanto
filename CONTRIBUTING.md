# Contributing to Esperanto

Thank you for considering contributing to Esperanto! Before you start, please read our [Architecture & Design Principles](ARCHITECTURE.md) — it explains the core decisions behind the project and will help you write contributions that align with the project's direction.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## Before You Start

### Read the Architecture Guide

The most common reason PRs need significant rework is a mismatch with our design principles. The [ARCHITECTURE.md](ARCHITECTURE.md) covers:

- **Provider parity**: New features must work across all (or most) providers
- **Provider tiers**: When a new provider class is justified vs. using OpenAI-Compatible
- **Testing philosophy**: What we expect in terms of test coverage

### Open an Issue First

For non-trivial changes (new features, new providers, architectural changes), **open an issue before writing code**. This lets us discuss the design and avoid wasted effort. Bug fixes and documentation improvements can go straight to a PR.

## How to Contribute

### Reporting Bugs

Before creating bug reports, check the issue list. When creating a bug report, include:

* A clear and descriptive title
* Steps to reproduce the problem
* Expected vs. actual behavior
* Error messages and stack traces
* Provider and model being used

### Suggesting Features

Open an issue with:

* A description of the feature
* Which providers it would apply to (ideally all of them)
* Example usage code showing the desired API

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`uv run pytest -v`)
5. Run the linter (`uv run ruff check .`)
6. Commit your changes using [conventional commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.)
7. Push and open a Pull Request

#### PR Guidelines

* **Read [ARCHITECTURE.md](ARCHITECTURE.md)** before implementing
* Follow existing code style and patterns
* Write tests for new features — test across all affected providers
* Update documentation for any user-facing changes
* Keep commits focused and atomic
* One feature/fix per PR

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/lfnovo/esperanto.git
cd esperanto
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv sync --group dev
```

If you need the `transformers` extra (for local model support):
```bash
uv sync --group dev --extra transformers
```

3. Run tests:
```bash
uv run pytest -v
```

4. Run linting:
```bash
uv run ruff check .
```

Fix auto-fixable issues:
```bash
uv run ruff check . --fix
```

The project's ruff configuration is in `pyproject.toml` and enforces:
- Line length of 88 characters
- Standard Python style rules (E, F)
- Import sorting (I)

## Release Tests

The `tests/integration/` directory contains tests that call real provider APIs. These are marked with the `release` pytest marker and are excluded from the default `uv run pytest` run to avoid accidental API charges.

To run release tests:

```bash
uv run pytest -m release
```

**Important:**
- These tests cost real money — they make live API calls to provider endpoints.
- They require provider API keys to be set in a `.env` file at the repo root.
- They are deliberately excluded from CI. Running them is a local-only ritual, intended for maintainers to verify everything works end-to-end before publishing a release.
- Do not add release tests to the default test scope, and do not run them in automated pipelines.

### Provider credentials reference

Each release-gated test class is `skipif`-gated on the env vars its provider needs. Set whichever subset you want to validate; tests without configured credentials skip cleanly. Common envs:

| Provider | Required env vars |
|----------|-------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google (Gemini) | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Vertex AI | `VERTEX_PROJECT` or `GOOGLE_CLOUD_PROJECT` (auth auto-discovered: ADC, `GOOGLE_APPLICATION_CREDENTIALS`, or `gcloud auth application-default login`) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY[_LLM/_EMBEDDING/_STT/_TTS]` + `AZURE_OPENAI_ENDPOINT[_*]` + `AZURE_OPENAI_API_VERSION[_*]`; for TTS also `AZURE_OPENAI_DEPLOYMENT_NAME_TTS` |
| Ollama | none — auto-probes `http://localhost:11434`. Override with `OLLAMA_BASE_URL` or `OLLAMA_API_BASE` for remote/non-default. |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Perplexity | `PERPLEXITY_API_KEY` (note: tool-calling tests skip — Perplexity API doesn't support tools) |
| MiniMax | `MINIMAX_API_KEY` |
| DashScope (Qwen) | `DASHSCOPE_API_KEY` |
| Jina | `JINA_API_KEY` |
| Voyage | `VOYAGE_API_KEY` |
| ElevenLabs | `ELEVENLABS_API_KEY` |
| Deepgram | `DEEPGRAM_API_KEY` |
| Transformers (local reranker) | none — gates on `sentence-transformers` package being installed |
| OpenAI-compatible (LiteLLM, vLLM, Together, etc.) | `OPENAI_COMPATIBLE_BASE_URL[_LLM/_EMBEDDING/_STT/_TTS]` (required); `OPENAI_COMPATIBLE_API_KEY[_*]` (optional — local servers like LiteLLM may not need auth; cloud-hosted ones like Together do) |

If a test fails with a "deployment not found" or similar provider-specific error rather than skipping, it usually means partial credentials are set (e.g., API key but no endpoint for Azure). The skipif gates require all the env vars the provider's `__post_init__` actually reads.

### Release-test playbook

A start-to-finish runbook for the release-time validation ritual.

**1. Pre-flight**

```bash
# Be on main, fully synced
git checkout main
git pull --ff-only

# Working tree clean (release tests must not be polluted by WIP changes)
git status

# .env loaded with the credentials you want to exercise
ls .env
```

The release suite is meant to be run from `main` against the code about to ship — do not run it from a feature branch unless you're specifically validating that branch.

**2. Run the full suite**

```bash
uv run pytest -m release
```

Expected runtime: **~2-3 minutes** with credentials for ~15 providers configured. Rough cost: **<$0.50 per full run** (most provider calls are short prompts to small models). Costs scale with how many providers your `.env` enables.

For a quick smoke before the full run:

```bash
# Just chat completion across providers (fastest, cheapest)
uv run pytest -m release tests/integration/test_chat_completion_real.py

# Just one provider, all surfaces
uv run pytest -m release -k TestOpenAI

# Just the streaming path (highest regression-prevention value)
uv run pytest -m release -k "streaming"
```

**3. Interpret results**

Each test reports as one of:

| Result | Meaning | Action |
|--------|---------|--------|
| `PASSED` | Real API call succeeded | None |
| `SKIPPED` | Required env var(s) not set, or known-unsupported feature (e.g. Perplexity tool calling) | None — gating works as designed |
| `XFAIL` | Known-broken test, tracked in a follow-up issue (e.g. `xfail(reason="see #N")`) | None — fix lands when the linked issue is resolved |
| `XPASS` | Known-broken test that unexpectedly passed | Investigate — flip the `xfail` to expected-pass and close the linked issue |
| `FAILED` | Real regression, env mismatch, or provider API change | Triage (next section) |

Healthy release run looks like: many PASS, some SKIPPED (providers without credentials), 0 FAILED.

**4. Triage failures**

When something fails, distinguish:

- **Provider parity bug** (Esperanto's fault): the test caught an inconsistency between what users get from one provider vs another. Fix in the provider source under `src/esperanto/providers/`. Example: Azure streaming yielding empty-`choices` chunks (PR #179).
- **Provider API change** (their fault): the underlying provider deprecated a model, changed an endpoint, or evolved a request format. Fix in the provider source or in test defaults. Example: Google `text-embedding-004` deprecated on v1beta (#177).
- **Test infrastructure bug**: the test gating, fixture, or assertion is wrong. Fix in `tests/integration/test_*_real.py`.
- **Env mismatch**: partial credentials, wrong deployment name, expired token. Fix your `.env` or skip cleanly via skipif tightening.

For non-trivial failures, file a follow-up issue rather than blocking the release. Mark the failing tests `@pytest.mark.xfail(reason="see #N")` so the suite stays green for the next maintainer.

**5. Where in the release process to run this**

Run the release suite **before tagging a release** — it's the last gate that catches cross-provider regressions the mocked unit tests can't. Specifically:

1. Cut a release branch (or work on `main` if shipping straight from there).
2. Update `CHANGELOG.md` with the release version and date.
3. Run `uv run pytest` (default, mocked) — must be green.
4. Run `uv run ruff check .` and `uv run mypy src/esperanto` — must be green.
5. **Run `uv run pytest -m release`** — must be green or have only known-tracked xfails.
6. Bump version, commit, tag, push tag.
7. Build + publish.

If step 5 surfaces a real regression, the release waits.

**6. Audio fixture**

`tests/fixtures/sample.mp3` is a committed 8-second MP3 (sliced from `notebooks/podcast.mp3`) used by the STT release tests. The test asserts the transcription contains `"Supernova"` (case-insensitive substring). The rest of `tests/fixtures/` is gitignored — only `sample.mp3` is committed via a `.gitignore` negation pattern.

If you replace the fixture with a different clip, update `EXPECTED_TRANSCRIPT_FRAGMENT` in `tests/integration/test_stt_real.py`.

**7. Local Ollama coverage (optional)**

The Ollama tests auto-probe `http://localhost:11434/api/tags`. To enable local Ollama coverage:

```bash
# Install Ollama (one-time): https://ollama.com/
ollama serve  # if not already running

# For tool-calling tests specifically, pull qwen3:32b
ollama pull qwen3:32b

# Tests will now auto-detect and run
uv run pytest -m release -k "Ollama"
```

For remote Ollama, set `OLLAMA_BASE_URL=https://your-ollama-host`.

## Adding a New Provider

This is the most common type of contribution. To keep Esperanto maintainable, we have clear criteria for what we accept.

### Acceptance Criteria

| Scenario | What to do |
|----------|------------|
| **OpenAI-compatible, no extra dependency** (just a different base_url and API key) | Add a profile in `src/esperanto/providers/llm/profiles.py`. We accept these for providers with demonstrated adoption (public docs, active pricing page, presence in LLM benchmarks/rankings). Profiles are ~6 lines of config, so the maintenance cost is minimal. |
| **Requires a new SDK dependency** (e.g., a provider-specific Python package) | The SDK must have **>500k monthly downloads on PyPI** consistently over 3+ months. This ensures we're not taking on maintenance burden for niche dependencies. |
| **OpenAI-compatible but needs custom behavior** (custom headers, special error handling, unique parameters) | Open an issue first. May justify a class extending `OpenAICompatibleLanguageModel`, but we'll evaluate case by case. |
| **Fundamentally different API** (non-OpenAI message format, unique auth) | Open an issue first. Needs a first-class provider class. Must meet the SDK download threshold if it adds a dependency. |
| **Doesn't meet the above criteria** | Register it yourself at runtime using `AIFactory.register_openai_compatible_profile()` in your own code. No PR needed — this is a feature, not a limitation. |

### Adding a Profile (OpenAI-compatible providers)

Most new provider requests are OpenAI-compatible endpoints. These are handled by adding a profile — no new Python class needed:

1. Add the profile to `BUILTIN_PROFILES` in `src/esperanto/providers/llm/profiles.py`
2. Add tests in `tests/providers/llm/test_profiles.py`
3. Add docs in `docs/providers/{provider}.md`
4. Update provider matrices in `README.md`, `docs/providers/README.md`, `docs/configuration.md`
5. Run the full test suite: `uv run pytest -v`

### Adding a First-class Provider (unique API)

For providers that need their own class:

1. **Open an issue** describing the provider and why a profile isn't sufficient.
2. **Read the base class** for your provider type (e.g., `LanguageModel`, `EmbeddingModel`).
3. **Study 2-3 existing providers** to understand the patterns.

If approved, follow this checklist:

- [ ] Create provider class in `src/esperanto/providers/{type}/{provider}.py`
- [ ] Implement all abstract methods from the base class
- [ ] Follow the `__post_init__()` pattern: `super().__post_init__()` first, `_create_http_clients()` last
- [ ] Register in `factory.py` under `_provider_modules["{type}"]`
- [ ] Add optional import in `src/esperanto/__init__.py` with try/except
- [ ] Write tests in `tests/providers/{type}/test_{provider}.py`
- [ ] Add docs in `docs/providers/{provider}.md`
- [ ] Run the full test suite: `uv run pytest -v`

## Adding a New Feature

Features that touch the public interface (new parameters, new response fields, new methods) **must work across all relevant providers**. This is our most important design principle.

Before implementing:

1. **Open an issue** with your proposed API design
2. **Show how it works across at least 3 providers** (e.g., OpenAI, Anthropic, Google)
3. **Discuss edge cases** — what happens with providers that don't support this feature?

## Questions?

Feel free to open an issue with your question. We'll do our best to help!
