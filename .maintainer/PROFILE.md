# Scope and public communication

## What Esperanto owns

The provider-agnostic Python interface, AIFactory, common response and error types,
provider adapters and compatible profiles, model discovery, shared utilities,
optional dependency boundaries, package metadata, tests and documentation.
Use ARCHITECTURE.md to resolve design questions: provider parity, predictable
hot-swapping, explicit unsupported features and demand-driven abstractions.

## What Esperanto does not own

Provider availability, pricing, quotas, model lifecycle and generated content;
upstream SDKs, local inference services and downstream applications.
Distinguish upstream behavior from defects in Esperanto's adapters with evidence.

## Communication

Speak Portuguese with the maintainer and English in public issues, PRs,
release notes and documentation. Answer directly, state verification limits,
and do not promise delivery dates. Do not add agent attribution by default.

## Never cite publicly

Credentials or their contents, .env files, google-credentials.json, private
notebooks, local specs, .maintainer/profile.local.toml, .maintainer/state/,
.harny/ state, unpublished drafts or private user material. Public explanations
should cite committed source, README.md, docs/, ARCHITECTURE.md or CONTRIBUTING.md.
Follow SECURITY.md for private vulnerability reporting.
