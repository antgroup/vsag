# Contributing to VSAG

First of all, thank you for taking the time to contribute to VSAG! Contributors like you are what keep the project alive and growing. 🎉

> If this is your first open-source contribution, we recommend walking through the
> [First Contributions](https://github.com/firstcontributions/first-contributions) tutorial to get
> familiar with the basic workflow.

The sections below cover what you may want to know before contributing.

## Ways to Contribute

1. **Report bugs.** File a bug issue with enough detail to reproduce the problem. If you consider
   the issue urgent, mention the VSAG team in a comment.
2. **Propose features.** File a feature request issue describing the expected behavior. Discuss the
   design with the VSAG team and the community before implementation. Once the plan is agreed,
   follow the [contribution flow](#contribution-flow).
3. **Implement features or fix bugs.** Pick up an open issue and follow the
   [contribution flow](#contribution-flow). Feel free to ask for clarifications by commenting on
   the issue and @-mentioning the VSAG team.

## Contribution Flow

> We use [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow) to
> collaborate on VSAG.

1. Fork the VSAG repository on GitHub.
2. Clone your fork locally: `git clone git@github.com:<yourname>/vsag.git`.
3. Create a working branch: `git checkout -b my-topic-branch`.
4. Make changes, run local checks, commit, and push with
   `git push --set-upstream origin my-topic-branch`.
5. Open a pull request on GitHub.

If you already have a local clone, update it before starting so that merge conflicts are less
likely:

```bash
git remote add upstream git@github.com:antgroup/vsag.git
git checkout main
git pull upstream main
git checkout -b my-topic-branch
```

## Guidelines

Before opening a pull request, make sure your changes pass local checks and follow the VSAG
coding style.

- New features must ship with tests that demonstrate correct behavior and guard against
  regressions.
- Bug fixes should add a regression test covering the triggering case; a missing test is usually
  what allowed the bug in the first place.
- Preserve API compatibility when editing code under `include/`.
- Do not include internal headers (from `src/`) in public headers (under `include/`).
- When contributing a new feature, remember that the maintenance cost shifts to the VSAG team by
  default — we evaluate contributions by weighing benefit against long-term maintenance.

## Signing Off (DCO)

All contributions to this project must include a
[Developer Certificate of Origin (DCO)](https://developercertificate.org/) sign-off. The sign-off
**must** be included in every commit message in the form
`Signed-off-by: {{Full Name}} <{{email address}}>` (without the `{}`). Contributions without a DCO
sign-off cannot be accepted.

```text
This is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
```

Git provides a `-s` flag that appends the trailer automatically:

```bash
git commit -s -m "This is my commit message"
```

For contributions co-authored by a human developer and an AI coding agent (OpenCode, Claude Code,
Codex, etc.), include a `Signed-off-by` trailer for **each** collaborator, for example:

```text
Signed-off-by: Random J Developer <random@developer.example.org>
Signed-off-by: OpenCode (claude-sonnet-4.5) <noreply@opencode.ai>
```

## Commit Messages and PR Labels

- Follow [Conventional Commits](https://www.conventionalcommits.org/); common prefixes include
  `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`.
- If a commit must skip CI, put `[skip ci]` at the **beginning** of the subject line, e.g.
  `[skip ci] docs: fix typo in README`.
- Every PR **must** carry two labels (enforced by Mergify, required to merge):
  - `kind/*`: `kind/bug`, `kind/feature`, `kind/improvement`, or `kind/documentation`.
  - `version/*`: the target release, e.g. `version/1.0`, `version/0.18`.

## Coding Style

VSAG follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with project-specific
tweaks covering indentation, naming, and line width. The authoritative configuration lives in the
repository:

- clang-format: <https://github.com/antgroup/vsag/blob/main/.clang-format>
- clang-tidy: <https://github.com/antgroup/vsag/blob/main/.clang-tidy>

> `clang-tidy` enforces not only naming conventions but also style checks such as magic-number
> usage.

The Makefile exposes formatting targets; `clang-format` and `clang-tidy` (both version 15) must be
installed.

Format code:

```bash
make fmt
```

Run static analysis (fix the reported issues manually):

```bash
make lint
```

Some clang-tidy findings can be auto-fixed:

```bash
make fix-lint
```

## Local Testing

Run the full test suite and make sure it passes:

```bash
make test
```
