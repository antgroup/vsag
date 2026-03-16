# CLAUDE.md

## Project Overview

VSAG is a high-performance vector indexing library for similarity search, written primarily in C++
with Python bindings provided by `pyvsag`.

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
- Avoid changing files under `extern/` unless the task explicitly requires third-party dependency changes.
- If changing build logic or dependencies, document the rationale clearly.
- Commit messages should follow Conventional Commits, such as `feat:`, `fix:`, `docs:`, or
  `chore:`.
- Commits in this project are expected to use DCO sign-off (`git commit -s`).
- For co-developed changes, include a `Signed-off-by:` trailer for each contributor.
- If dual sign-off is requested, include both the requester's identity and the current agent
  identity in the `Signed-off-by:` trailers.
- Determine the agent sign-off at execution time from the active agent name and model name, and use
  the format `AgentName (ModelName)`.
- When using skip-CI commit messages, follow the repository convention and place `[skip ci]` at the
  beginning of the subject line.

## References

- `README.md`
- `CONTRIBUTING.md`
- `DEVELOPMENT.md`
- `.github/copilot-instructions.md`
