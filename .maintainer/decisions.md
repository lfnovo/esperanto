# Maintenance decisions

Append-only record of maintainer decisions. For each entry record the date,
candidate or issue, decision, reasoning and evidence. No release decision has
been made by this profile bootstrap.

## 2026-09-05 — Adopt oss-maintainer

- Decision: apply the pypi-library profile; make tag is the current official
  publication path, confirmed by the maintainer.
- Decision: defer the clean-room package gate to issue #279. Keep release
  configuration incomplete until implemented; no gate waiver was requested.
- Decision: enable GitHub Discussions and record the live category IDs; use the
  plugin's pull graduation and answer/landed-work close policies.
- Decision: create the missing needs-triage, needs-vision and released labels.
- Decision: omit the application smoke capability because Esperanto is a library.
- Evidence: explicit maintainer approval in the adoption session; Makefile and
  .github/workflows/publish.yml; verified GitHub settings, categories and labels;
  https://github.com/lfnovo/esperanto/issues/279.
- This authorizes profile adoption and repository setup, not a release publication.
