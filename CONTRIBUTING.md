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
