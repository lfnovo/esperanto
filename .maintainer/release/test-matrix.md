# Release test matrix

Instantiate this matrix against each candidate's risks, not just its feature list.
Record passed, failed, not-run and not-applicable checks with revision and evidence.

## Bucket A — automated

| Check | Canonical reference | Coverage |
|---|---|---|
| Default mocked suite | commands.validator -> make test | pytest excludes release marker |
| Lint | commands.ruff -> CLAUDE.md | Same check as lint.yml |
| Types | commands.mypy -> CLAUDE.md | src/esperanto, as in lint.yml |
| Packaging | artifacts.pypi.gate | Missing canonical clean-room gate; blocks release |

The setup target installs all extras. CI additionally installs mxbai-rerank.
CI tests 3.10/3.11/3.12; package metadata also supports 3.13. Track actual coverage.

## Regression probes to select by diff

Use existing tests/providers, tests/unit, tests/common_types and deprecation tests.
Pay particular attention to shared factory/configuration changes, optional imports,
normalized response types, tool calling, structured outputs, and streaming parity.
Do not claim a provider was exercised merely because a shared base test passed.

## Bucket B — investment candidates

- Implement the package gate described in runbook.md, including wheel/sdist and extras
  ([issue #279](https://github.com/lfnovo/esperanto/issues/279), deferred by the owner).
- Add Python 3.13 CI coverage if the maintainer elects to cover the full declared range.
- Reconcile the CI-specific mxbai-rerank setup with local validation documentation.

## Bucket C — manual before publication

Run only with the maintainer's configured credentials and authorization, using the
release-marker commands already documented in CONTRIBUTING.md.

| Surface | Existing suite |
|---|---|
| Chat and streaming | tests/integration/test_chat_completion_real.py |
| Tools | tests/integration/test_tool_calling_real.py |
| Structured output | tests/integration/test_structured_output_release.py |
| Embeddings | tests/integration/test_embedding_real.py |
| Reranking | tests/integration/test_reranker_real.py |
| Speech to text | tests/integration/test_stt_real.py |
| Text to speech | tests/integration/test_tts_real.py |

Test changed providers plus a baseline for unchanged modalities. Missing credentials
leave that surface unverified. Separate known tracked xfails and environment issues
from regressions; regressions block GO. Review notes as the notes-approved gate.

## Post-publication

Verify exact-version index installation, library behavior, extras, artifact identities
and the release notes when a GitHub Release is created. Local wheel tests do not
replace index verification because the publication workflow rebuilds the package.
