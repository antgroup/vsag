# AGENTS.md

## Project Overview

VSAG is a high-performance vector indexing library for similarity search, written primarily in C++
with Python bindings provided by `pyvsag`.

## Where To Learn About The Project

Before making non-trivial changes, consult the project documentation:

- The official documentation site: <https://vsag.io/docs> (English and Chinese versions are
  available, including index parameters, best practices, performance references, and the
  `eval_performance` tool guide).
- The in-repo `docs/` directory, which mirrors the website source under `docs/docs/{en,zh}/src/`
  and also contains design notes (e.g. `docs/hgraph.md`, `docs/ivf.md`, `docs/sindi.md`,
  `docs/eval_performance.md`, `docs/dataset_format.md`).

When updating user-facing behavior, keep both the website docs and any related in-repo READMEs
(`tools/eval/README.md`, `tools/eval/README_zh.md`, etc.) in sync.

## What To Optimize For

- Preserve API compatibility unless a breaking change is explicitly intended.
- Keep performance in mind for hot paths, memory layout, and parallel code.
- Follow existing project structure and naming before introducing new patterns.
- Prefer minimal, targeted changes over broad refactors.

## Repository Structure

- `include/`: public headers
- `src/`: core implementation and unit tests
- `tests/`: functional tests
- `examples/cpp/`: C++ examples
- `examples/python/`: Python examples
- `python/`: `pyvsag` packaging and Python-side code
- `python_bindings/`: pybind11 bindings
- `docs/`: design and user-facing documentation
- `tools/`: utility and analysis tools

## Build And Test

Preferred commands:

```bash
make debug
make test
make fmt
make lint
make fix-lint
make cov
make release
make pyvsag
```

## Development Environment

- Recommended: use the published Docker development image.
- Supported local environments include Ubuntu 20.04+ and CentOS 7+.
- Compiler baseline: GCC 9.4.0+ or Clang 13.0.0+.
- Build baseline: CMake 3.18.0+.

## Required Tool Versions

- `clang-format` must be version 15 exactly.
- `clang-tidy` must be version 15 exactly.

Do not assume newer versions are acceptable; the repository enforces these versions for consistent
formatting and diagnostics.

## Coding Standards

Follow the Google C++ Style Guide with project-specific rules:

- 4-space indentation
- use `.cpp` instead of `.cc`
- 100-character line limit
- ensure committed text files end with a trailing newline

Additional expectations:

- Keep public APIs in `include/vsag/`.
- Keep implementation in `src/`.
- Place tests near the code they validate when appropriate.
- Add or update Doxygen comments for public APIs.

## Testing Expectations

- New features should include tests.
- Bug fixes should include regression coverage.
- Contributions are expected to maintain at least 90% code coverage on the C++ library code
  (sources under `src/` and public headers under `include/`), as measured by the coverage job
  invoked via `make test` and the corresponding CI coverage workflow. This threshold is based on
  the unit-test suite; functional tests under `tests/` and Python code are not currently included
  in the coverage metric unless explicitly documented otherwise.

## Common Code Patterns

- Builder-style chained APIs are common.
- `std::shared_ptr<T>` is used widely in public interfaces.
- Prefer existing error-handling/result patterns used in the surrounding code.
- Keep code in the `vsag` namespace unless the file clearly requires otherwise.

## Documentation Expectations

Update relevant docs when behavior changes:

- `README.md` for user-facing features or examples
- `DEVELOPMENT.md` for build or environment changes
- `CONTRIBUTING.md` for workflow or contribution policy changes

## Contribution Notes

- Before modifying code, create and switch to a working branch from an up-to-date `main`.
- Keep changes scoped and reviewable.
- Be careful with performance-sensitive code and cross-platform behavior.
- Prefer `uint64_t` over `size_t` in code changes to avoid potential macOS compile issues.
- Avoid changing files under `extern/` unless the task explicitly requires third-party dependency changes.
- If changing build logic or dependencies, document the rationale clearly.
- Commit messages should follow Conventional Commits, such as `feat:`, `fix:`, `docs:`, or
  `chore:`.
- Commits in this project are expected to use DCO sign-off (`git commit -s`).
- Only humans can legally certify the DCO; AI coding agents **must not** add their own
  `Signed-off-by` trailer. The human submitter is responsible for reviewing AI-generated
  changes, ensuring license compliance, and taking full responsibility for the contribution.
- For changes produced with AI assistance, attribute the agent with an `Assisted-by:` trailer
  in the form `Assisted-by: AgentName:ModelVersion` (see the
  [Linux kernel AI Coding Assistants policy](https://docs.kernel.org/process/coding-assistants.html)).
  Determine `AgentName` and `ModelVersion` at execution time from the active agent name and
  model name (e.g. `Assisted-by: OpenCode:claude-opus-4.7`).
- Trailer order: place the human `Signed-off-by:` first, followed by the `Assisted-by:` line
  for the AI agent.
- When using skip-CI commit messages, follow the repository convention and place `[skip ci]` at the
  beginning of the subject line.

## Pull Request Labels

Every pull request **must** have two labels before it can be merged:

- A `kind/*` label for the type of change: `kind/bug` (bug fix), `kind/feature` (new feature), `kind/improvement` (refactor, chore, or minor improvement), or `kind/documentation` (documentation change).
- A `version/*` label for the target version (e.g. `version/1.0`, `version/0.18`).

Mergify enforces these via check runs. Always add both labels when creating a PR.

## Filing Issues With An Agent

Agents are also expected to help users file high-quality issues. The shared
workflow lives in `.github/agent-prompts/create-issue.md` and the writing
rules live in `.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md`. Both Claude Code,
OpenCode and Codex expose this workflow as a `/create-issue` slash command
(see `.claude/commands/`, `.opencode/command/`, `.codex/prompts/`).

When drafting an issue:

- Map every required field of the chosen YAML template under
  `.github/ISSUE_TEMPLATE/`. Use `// TODO` for genuinely unknown values.
- Cite sources as `path:line` for repo references and full URLs for
  `vsag.io/docs` pages.
- Run `gh issue list --repo antgroup/vsag --search ... --state all --limit 5`
  and append the matches under a `## Related issues` section. Do not block
  creation on duplicates; the maintainers will close as needed.
- Append a footer of the form
  `_Drafted with AI assistant: <AgentName>:<ModelVersion>_` for transparency.
  Do **not** add `Signed-off-by:` to issues — DCO applies only to commits.
- Show a dry-run preview and only call `gh issue create` after the user
  confirms. On failure, fall back to `gh issue create --web`.

A shell wrapper is provided at `tools/issue-helper/new-issue.sh` for users
who prefer not to drive the agent's slash command directly.

## References

- `README.md`
- `CONTRIBUTING.md`
- `DEVELOPMENT.md`
- `.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md`
