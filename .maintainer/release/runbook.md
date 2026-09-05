# Release runbook

The human release-test playbook remains in CONTRIBUTING.md. Commands are defined
in CLAUDE.md and Makefile and referenced through profile.toml. The oss-maintainer
release skill supplies run records, gate evidence, notes and the GO decision.

## Preparation and bucket A

Select the candidate revision and inspect its diff and working tree. Use the
existing release-test playbook for a main-based release; never discard local work.
Run commands.setup when dependencies need installation, then commands.validator,
commands.ruff and commands.mypy. The validator references make test, whose pytest
configuration excludes release tests. For an ordinary agent implementation task,
use the narrower validator documented in CLAUDE.md.

Record the candidate SHA, interpreter, command, exit status and evidence. CI's
extra mxbai-rerank install and its Python matrix are documented in test.yml.

## Package gate — missing canonical command

TODO: define and validate a canonical build-and-clean-room-install command in
AGENTS.md or Makefile, then replace artifacts.pypi.gate with its reference.
The owner deferred implementation to [issue #279](https://github.com/lfnovo/esperanto/issues/279).
The existing uv build step in publish.yml is build-only; it cannot satisfy this gate.

The gate must build wheel and sdist into a clean output location, check metadata
against the candidate version, inspect packaged assets, install the wheel outside
the checkout with isolated Python imports, import Esperanto and exercise a
credential-free public API call. Confirm optional dependencies stay optional and
that both transformers and validation extras resolve. Verify the sdist can build
an installable wheel. Record SHA-256 identities and actual module origins.
No package-gate pass or release GO is possible until this evidence exists.

## Bucket C — maintainer-run integrations

Use CONTRIBUTING.md's release-test playbook and the concrete suites listed in
release/test-matrix.md. These calls require credentials and can cost money;
ordinary automated validation must not invoke them. Scope to changed providers
plus representative unchanged modalities. Record skipped providers as unverified.
Classify failures; real regressions block. Known tracked xfails need review and
must not conceal regressions introduced by this candidate.

## Cut

Choose the version from the consumer-visible diff. Update pyproject.toml and date
the CHANGELOG.md release section, then use release.lock_command to regenerate
uv.lock. Commit both version files together. Follow CONTRIBUTING.md for the PR.
Repeat gates against the final candidate after version or source changes.
Prepare and approve release notes before any distribution trigger.

## Publish — after GO

The owner confirmed make tag as the canonical distribution trigger. Its Makefile target
reads pyproject.toml, creates v<version> and pushes it to origin; publish.yml
then builds and publishes to PyPI. Dispatching publish.yml also publishes directly.
Treat create-tag.yml as a tag mutation path requiring the same release decision.
Do not invoke any of these while preparing or validating the candidate.

The profile does not grant publication permission. Record the owner's GO for the
specific candidate and version in the release run record before distribution.

## Post-publication verification

Observe Publish for the correct revision and verify the version actually appears
on PyPI. Install the exact version from the index outside the checkout, repeat
public-library and extras checks, and record the distributed artifact identities.
Publish rebuilds the artifacts: distinguish tested local builds from index builds.
If creating GitHub release notes, attach approved notes to the existing tag and
verify the release page; publish.yml does not create a GitHub Release itself.

TODO: document canonical index-install verification commands alongside the package
gate. This verification completes publication; it cannot run before publication.

## Cleanup

Keep evidence in .maintainer/state/. Inspect git status and remove only artifacts
created by this run. Preserve credentials, notebooks and unrelated changes.
Record partial publication; never reuse a published version or move its tag.
