# AGENTS.md

> **Read me first.** This file is the entry point for AI coding agents
> working on VSAG. It is intentionally short. The bullets under
> *Hard Constraints* must hold for every change; the *Documentation Index*
> points at the detailed rules for each topic.

## Project Overview

VSAG is a high-performance vector indexing library for similarity search,
written primarily in C++ with Python bindings provided by `pyvsag` and
Node.js/TypeScript bindings provided by `vsag`.

## Hard Constraints (must hold every change)

- **Tool versions are fixed:** `clang-format` and `clang-tidy` **must be
  version 15 exactly**. Newer versions are not acceptable; CI enforces this.
- **Formatting:** 4-space indentation, 100-character line limit, `.cpp`
  (not `.cc`) suffix, every committed text file ends with a trailing newline.
- **Layout:** public APIs live in `include/vsag/`; implementation lives in
  `src/`; keep code in the `vsag` namespace unless the file clearly requires
  otherwise.
- **Prefer `uint64_t` over `size_t`** in code changes to avoid macOS compile
  issues.
- **Do not modify `extern/`** unless the task explicitly requires third-party
  dependency changes.
- **Tests:** new features need tests; bug fixes need regression coverage; the
  C++ library (`src/` + `include/`) must maintain **≥ 90 % unit-test
  coverage**.
- **Commits:** use Conventional Commits (`feat:`, `fix:`, `docs:`,
  `chore:`, …) and DCO sign-off. **AI agents must NOT add their own
  `Signed-off-by:`** — only humans can certify the DCO. **AI agents must
  also NOT invoke `git commit -s`**, because `-s` appends `Signed-off-by:`
  at the end of the message (with a blank line separating it from earlier
  trailers) and breaks the required ordering. Instead, write both trailers
  directly in the commit message body: derive the human identity from
  `git config user.name` / `user.email`, place `Signed-off-by:` first, and
  follow it immediately (no blank line) with
  `Assisted-by: <AgentName>:<ModelVersion>` (e.g.
  `Assisted-by: OpenCode:claude-opus-4.7`). For `[skip ci]`, place it at
  the **start** of the subject line.
- **Pull requests need two labels:** one `kind/*` (`kind/bug`,
  `kind/feature`, `kind/improvement`, `kind/documentation`) and one
  `version/*` (e.g. `version/1.0`). Mergify enforces both.
- **`kind/bug` / `kind/feature` PRs must link an issue** in the
  description using a GitHub auto-closing keyword (`Fixes: #N`,
  `Closes: #N`, `Resolves: #N`; cross-repo or full issue URL also OK).
  `kind/improvement` and `kind/documentation` PRs are exempt. Enforced
  by the `PR Issue Link Check` Action and a Mergify protection.
- **Issues do NOT carry `Signed-off-by:`** — DCO only applies to commits.

## Quick Command Reference

```bash
make debug        # debug build
make test         # build and run unit + functional tests (no coverage instrumentation)
make fmt          # run clang-format
make lint         # run clang-tidy
make fix-lint     # apply clang-tidy fixes where safe
make cov          # build with coverage instrumentation; run the test binaries and
                  # scripts/coverage/collect_cpp_coverage.sh to produce a report
make release      # optimized build
make pyvsag       # build the Python wheel
```

Full build and environment details: [`docs/agents/build-and-test.md`](docs/agents/build-and-test.md).

## Documentation Index (read the relevant file before non-trivial work)

| Topic | Document |
| --- | --- |
| Build / test / dev environment / tool versions | [`docs/agents/build-and-test.md`](docs/agents/build-and-test.md) |
| C++ style, layout, common patterns, testing | [`docs/agents/coding-standards.md`](docs/agents/coding-standards.md) |
| Branching, commits, DCO, `Assisted-by`, PR labels, doc sync | [`docs/agents/contribution-workflow.md`](docs/agents/contribution-workflow.md) |
| Drafting & submitting GitHub issues (`/create-issue`) | [`docs/agents/filing-issues.md`](docs/agents/filing-issues.md) |
| Repository / docs map | [`docs/agents/docs-map.md`](docs/agents/docs-map.md) |

## How To Research The Project

Before making non-trivial changes, consult the project documentation:

- Official documentation site: <https://vsag.io/docs> (English and Chinese;
  index parameters, best practices, performance references, and the
  `eval_performance` tool guide all live there).
- In-repo sources mirror the site under `docs/docs/{en,zh}/src/`; this is
  the canonical location for user-facing documentation. Top-level `docs/*.md`
  files, when present, are historical design notes rather than preferred user
  docs. Consult [`docs/agents/docs-map.md`](docs/agents/docs-map.md) for the
  full layout.

When updating user-facing behavior, update the website docs under
`docs/docs/{en,zh}/src/`. Tool-level READMEs should stay as local pointers to
those canonical pages. See
[`docs/agents/contribution-workflow.md`](docs/agents/contribution-workflow.md#documentation-expectations).

## References

- [`README.md`](README.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`DEVELOPMENT.md`](DEVELOPMENT.md)
- [`.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md`](.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md)
- [`docs/agents/`](docs/agents/) — detailed agent operating docs
