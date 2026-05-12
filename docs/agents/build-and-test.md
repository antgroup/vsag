<!-- agent-hints
canonical: docs/agents/build-and-test.md
purpose: Build, test, and development environment rules for VSAG
key-facts:
  - clang-format and clang-tidy must be version 15 exactly
  - Preferred build path is the published Docker image
  - Use the make targets listed below instead of raw cmake
related:
  - ../../DEVELOPMENT.md
  - ../docs/en/src/development/building.md
last-reviewed: 2026-05-12
-->

# Build And Test

## Preferred commands

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

Prefer these targets over invoking `cmake` directly so behavior matches CI.

## Development environment

- Recommended: use the published Docker development image
  (`vsaglib/vsag:ubuntu`).
- Supported local environments include Ubuntu 20.04+ and CentOS 7+.
- Compiler baseline: GCC 9.4.0+ or Clang 13.0.0+.
- Build baseline: CMake 3.18.0+.

See [`DEVELOPMENT.md`](../../DEVELOPMENT.md) for the full setup walkthrough.

## Required tool versions

- `clang-format` must be version **15** exactly.
- `clang-tidy` must be version **15** exactly.

Do not assume newer versions are acceptable; the repository enforces these
versions for consistent formatting and diagnostics. CI will fail otherwise.

## Testing expectations

- New features should include tests.
- Bug fixes should include regression coverage.
- Contributions are expected to maintain at least **90% code coverage** on the
  C++ library code (sources under `src/` and public headers under `include/`).
  `make test` builds and runs the unit + functional suite but does **not**
  enable coverage instrumentation. To produce the coverage report locally use
  the `make cov` flow (which configures the build with `ENABLE_COVERAGE=ON`)
  followed by running the test binaries and
  `scripts/coverage/collect_cpp_coverage.sh`; the equivalent CI coverage
  workflow enforces the 90% threshold. This threshold is based on the
  unit-test suite; functional tests under `tests/` and Python code are not
  currently included in the coverage metric unless explicitly documented
  otherwise.
