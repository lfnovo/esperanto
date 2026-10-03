# Security Policy

Esperanto is a unified interface to many AI providers, and as such it handles
provider API keys and brokers requests to external services. We take the
security of the library and its users seriously.

## Supported Versions

Security fixes are released for the latest published minor version on PyPI.
We recommend always running the most recent release.

| Version | Supported          |
| ------- | ------------------ |
| 2.25.x  | :white_check_mark: |
| < 2.25  | :x:                |

When a new minor version ships, support for the previous line ends. If you are
pinned to an older version, upgrade to receive security fixes.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the [**Security** tab](https://github.com/lfnovo/esperanto/security) of
   this repository.
2. Click **Report a vulnerability**.
3. Fill in the advisory form with as much detail as you can.

This opens a private channel visible only to the maintainers.

Please include, where applicable:

- A description of the vulnerability and its impact.
- Steps to reproduce, or a proof-of-concept.
- The affected version(s) and, if known, the affected provider(s) or code path.
- Any suggested remediation.

## Response Timeline

- **Acknowledgement:** we aim to acknowledge your report within **72 hours**.
- **Assessment:** an initial assessment and severity triage within **7 days**.
- **Resolution:** for confirmed vulnerabilities, we will work on a fix and keep
  you informed of progress. Timelines depend on severity and complexity.

Once a fix is available, we will coordinate a release and, with your consent,
credit you in the release notes and the security advisory.

## Scope

This policy covers the Esperanto library itself. Vulnerabilities in the
upstream provider APIs or third-party SDKs should be reported to their
respective maintainers, though we welcome a heads-up if the issue affects how
Esperanto should be used safely.

Thank you for helping keep Esperanto and its users safe.
