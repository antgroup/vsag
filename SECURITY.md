# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x:                |

## Reporting a Vulnerability

The VSAG team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing the maintainers at:

**[security@vsag-project.org](mailto:security@vsag-project.org)**

If you prefer to use encrypted communications, you can also open a [Security Advisory](https://github.com/antgroup/vsag/security/advisories/new) on GitHub.

Please include the following information in your report:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

## Preferred Languages

We prefer all communications to be in English or Chinese.

## Response Timeline

* We will acknowledge receipt of your vulnerability report within 3 business days
* We will send a more detailed response within 7 days indicating the next steps in handling your report
* We will keep you informed of the progress towards a fix and announcement

## Disclosure Policy

* When we receive a security bug report, we will assign it to a primary handler
* The handler will coordinate the fix and release process, involving the following steps:
  * Confirm the problem and determine the affected versions
  * Audit code to find any potential similar problems
  * Prepare fixes for all supported releases
  * Release new security fixes as soon as possible

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue to discuss.
