# Release runbook

The human release-test playbook remains in CONTRIBUTING.md. Commands are defined
in AGENTS.md and Makefile and referenced through profile.toml. The oss-maintainer
release skill supplies run records, gate evidence, notes and the GO decision.

## Preparation and bucket A

Select the candidate revision and inspect its diff and working tree. Use the
existing release-test playbook for a main-based release; never discard local work.
Run commands.setup when dependencies need installation, then commands.validator,
commands.ruff and commands.mypy. The validator references make test, whose pytest
configuration excludes release tests. For an ordinary agent implementation task,
use the narrower validator documented in AGENTS.md.

Record the candidate SHA, interpreter, command, exit status and evidence. CI's
extra mxbai-rerank install and its Python matrix are documented in test.yml.

## Package gate

Run `make package-check`. The command builds fresh wheel and sdist artifacts in
a temporary directory, checks their metadata and packaged runtime files, and
records SHA-256 identities. It installs the wheel outside the checkout with
isolated Python, verifies a bare import and credential-free factory discovery,
then builds a second wheel from the sdist and repeats the clean-room smoke. It
also resolves the `transformers` and `validation` extras independently. The
existing uv build step in publish.yml is build-only and cannot satisfy this gate.

Capture the command output as release evidence. It includes the interpreter,
artifact identities and actual module origins. A package-gate pass applies only
to those exact locally built artifacts and candidate revision.

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

Install the exact published version outside the checkout with:

```bash
check_dir="$(mktemp -d)"
cd "$check_dir"
uv run --isolated --no-project --with "esperanto==<version>" \
  python -I -c 'import esperanto; print(esperanto.__file__)'
```

Repeat with `esperanto[transformers]==<version>` and
`esperanto[validation]==<version>`, download the wheel and sdist from PyPI, and
record their SHA-256 identities. This verification completes publication; it
cannot run before publication.

## Cleanup

Keep evidence in .maintainer/state/. Inspect git status and remove only artifacts
created by this run. Preserve credentials, notebooks and unrelated changes.
Record partial publication; never reuse a published version or move its tag.
