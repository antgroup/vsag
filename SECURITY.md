# Security Policy

The VSAG team takes the security of the library seriously. This document
describes how to report vulnerabilities and which release lines we keep
patched.

## Reporting a Vulnerability

**Please do not file a public GitHub issue for security-sensitive reports.**

The preferred channel is GitHub's private vulnerability reporting:

1. Navigate to the repository's
   [Security tab](https://github.com/antgroup/vsag/security).
2. Click **Report a vulnerability** and fill in the form.

This opens a private advisory visible only to the reporter and the VSAG
maintainers, and lets us coordinate a fix before any public disclosure.

When you report an issue, please include as much of the following as you
can:

- A clear description of the issue and its impact (which API surface,
  which index type, which build configuration).
- A minimal reproduction — source snippet, dataset shape, or a stack
  trace from a sanitizer build (ASAN / TSAN) is especially helpful.
- The VSAG version (`vsag::version()` output, or the `pyvsag` /
  `vsag` npm package version) and the platform you observed it on.
- Any mitigation or workaround you have already identified.

You should receive an acknowledgement within **5 business days**. We aim
to triage the report and propose a remediation plan within
**10 business days**, and to ship a patched release within
**90 days** of the initial report, in line with widely used coordinated
disclosure practice. Embargo dates are negotiated case by case for
issues that warrant it.

## Supported Versions

VSAG follows [Semantic Versioning 2.0](https://semver.org/) (see also
the [Release Notes](docs/docs/en/src/resources/release_notes.md) page in
the documentation site). Security fixes are landed on:

- the current development branch (`main`); and
- any **minor release line** still receiving patch releases on
  [GitHub Releases](https://github.com/antgroup/vsag/releases).

Once a minor line stops receiving patch releases, it stops receiving
security backports as well. If you depend on an unsupported line and
need a backport, please mention this in your report so we can discuss
the scope.

## Public Disclosure

When a vulnerability is fixed, we publish:

- a GitHub Security Advisory on this repository, with a CVE identifier
  when applicable; and
- release notes for the patched versions on GitHub Releases.

We credit reporters in the advisory unless they ask to remain
anonymous.

## Out of Scope

The following do not count as security vulnerabilities for the purposes
of this policy:

- Performance regressions, including denial-of-service patterns that
  require an attacker-controlled index build with adversarial parameters
  but no malformed input — please file these as regular issues so we can
  evaluate them as quality bugs.
- Findings in third-party dependencies bundled under `extern/` — please
  report those upstream first (e.g. CRoaring, fmt, OpenBLAS, DiskANN);
  we will rebase to the fixed version once it ships.
- Issues that require a host with already-compromised privileges or
  arbitrary code execution.
