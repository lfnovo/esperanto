# Triage rules

Apply the plugin's maturity-ladder preset with the repository's existing policy
in ARCHITECTURE.md and CONTRIBUTING.md. State measures specification maturity,
not priority. Confirm issue mutations according to the configured approval mode.

## Existing labels

- ready: resolved design, concrete implementation pointers and acceptance criteria.
- needs-triage: intake awaiting maintainer classification; remove when triaged.
- needs-vision: a product direction decision must precede technical design;
  recognized by triage, not assigned by the default preset.
- needs-design: unresolved interface, provider coverage, authentication or scope.
- awaiting-demand: valid idea parked pending concrete demand; preserve it unless
  the maintainer explicitly requests reconsideration.
- bug, enhancement and documentation classify the work independently of maturity.
- question, duplicate, invalid, wontfix, help wanted, good first issue and codex
  also exist; their presence does not make an issue implementation-ready.
- released: work verified in a published release, not merely merged. It is not a
  maturity state; apply only after publication evidence is available.

The owner authorized creating needs-triage, needs-vision and released during
profile adoption on 2026-09-05; all profile labels now exist on GitHub. An issue
without a maturity label can also be reviewed as intake. Recheck live labels before
later mutations. Under the default assignable states, triage sends unresolved
vision questions to needs-design and states the missing product decision.

## Ready criteria and design fit

Require a reproducible problem or concrete use case, files/patterns to follow,
acceptance criteria, a known external contract and no unresolved design choices.
For provider additions, first evaluate the OpenAI-compatible profile path.
For shared features, state supported providers, normalized types, sync/async and
streaming behavior, and explicit handling of unsupported capabilities.

## Close criteria

Verify that work is implemented, duplicated or superseded before recommending
closure; link the evidence or canonical issue. Do not close merely because
implementation cannot be found. Preserve existing awaiting-demand decisions;
for new speculative requests follow the preset's open-door closure criteria.
Security reports follow SECURITY.md and do not belong in public issue detail.

## Areas

Use factory/model-discovery, common types, shared utilities, and LLM, embedding,
reranker, STT and TTS providers to describe scope. No area-specific assignee map
is declared; do not invent owners. Public comments use English.
