# Esperanto maintainer profile

Configuration for the oss-maintainer plugin, schema v1. This directory records
repository policy and maintenance context; it is not a list of maintainers.

| File | Purpose |
|---|---|
| `profile.toml` | Commands, release gates, labels and communication settings |
| `PROFILE.md` | Ownership, public tone and private material |
| `gotchas.md` | Fragile areas and release pitfalls |
| `triage.md` | Project-specific maturity and close criteria |
| `release/runbook.md` | Release sequence and distribution boundary |
| `release/test-matrix.md` | Automated, future and manual verification |
| `decisions.md` | Append-only maintenance decisions |
| `profile.local.toml` | Ignored machine-specific preferences |
| `state/` | Ignored run records and evidence |

Commands remain canonical in `AGENTS.md` and `Makefile`; contribution and
architecture policy remain in `CONTRIBUTING.md` and `ARCHITECTURE.md`.
Read the relevant module-level `AGENTS.md` when reviewing its implementation.

Review, triage, Discussions and release are configured. The owner confirmed
`make tag` as the release trigger. `make package-check` is the canonical
clean-room package gate implemented by
[issue #279](https://github.com/lfnovo/esperanto/issues/279).
A valid profile is not evidence that release checks have passed.
Discussions is enabled, with its live category IDs in `profile.toml`.
The application smoke skill does not apply to this library. Real-provider
integration tests belong to release bucket C.

Discussions use the plugin's pull graduation policy: proposals become issues when
someone will implement them, with verified bugs and explicit maintainer requests
as exceptions. Close answered threads and threads whose graduated work has landed.
Public responses use English and the sources listed in discussions.public_anchors.

Agent instructions live in `AGENTS.md`, at the root and in each documented
module. Each corresponding `CLAUDE.md` contains only `@AGENTS.md`, importing the
same instructions without duplication. The root instructions point to this profile.

The plugin's init validator requires Python 3.11+, independently of the library's
Python >=3.10,<3.14 support. Run the installed plugin's
`skills/init/scripts/validate_profile.py --root <checkout>` using a suitable Python.

Local overlays may change only the preference fields allowed by schema v1
(owner language, command-document path, upstream checkouts, announcement
channels and smoke URLs). They cannot override gates or public-language policy.

`.gitattributes` export-ignore affects Git archives. It does not prove exclusion
from Hatch-built wheel/sdist artifacts; inspect their contents during packaging.
