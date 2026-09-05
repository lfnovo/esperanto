# Gotchas

- Provider parity is a release concern. Changes to base classes, common types,
  factory registration or utilities require checks across affected providers.
  Read ARCHITECTURE.md and module CLAUDE.md files before evaluating behavior.
- Base initialization must precede provider configuration; create HTTP clients
  after keys, URLs and inherited configuration have been resolved.
- Bare installs must remain usable without the transformers or validation extras.
  Editable checkout imports do not prove that a built package works.
- make test delegates to the default pytest suite. pyproject.toml excludes the
  release marker, but new tests must still mock external calls. CLAUDE.md also
  documents the narrower automated-agent validator; do not confuse its scope
  with all collected tests. Never run the release marker as ordinary validation.
- CI test setup additionally installs mxbai-rerank; make setup does not. CI tests
  Python 3.10, 3.11 and 3.12, while the package permits 3.13 as well. Record which
  environments actually ran; do not claim the full supported range from one run.
- make ruff uses --fix; the release check is the non-mutating command from
  CLAUDE.md. make lint checks a broader scope than CI's mypy src/esperanto.
- make tag creates AND pushes a version tag. publish.yml publishes on v* tag
  pushes and manual dispatch. Neither path is a pre-publication validation step.
- create-tag.yml is a separate tag-writing path. Its success message is not proof
  that Publish ran; verify the downstream workflow and index artifacts explicitly.
- Publish rebuilds the package and runs no tests. Local artifact hashes cannot
  be claimed as the identities of those uploaded by CI.
- pyproject.toml and the Esperanto entry in uv.lock currently both declare 2.26.0.
  Regenerate the lock after a version change and commit them together.
- Integration skips mean missing coverage, not passed provider checks. Separate
  environment failures, stale assertions and real regressions. Existing guidance
  allows tracked xfails; do not hide a new regression under an unreviewed xfail.
- STT real-API tests use the committed tests/fixtures/sample.mp3. Audio tests can
  leave local outputs; inspect status and remove only files created by the run.
- CHANGELOG.md uses merge=union for independent additions. Review the final
  section structure after merges; the driver does not resolve semantic conflicts.
