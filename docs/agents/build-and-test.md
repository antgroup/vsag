<!-- agent-hints
canonical: docs/agents/build-and-test.md
purpose: Build, test, and development environment rules for VSAG
key-facts:
  - clang-format and clang-tidy must be version 20 exactly
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
make test-module MODULE=datacell
make fmt
make lint
make fix-lint
make cov
make release
make pyvsag
```

Prefer these targets over invoking `cmake` directly so behavior matches CI.

`make test-module MODULE=<name>` is the local acceleration path for one unit-test subsystem. The maintained module names are `gpu`, `simd`, `common`, `algorithm`, `factory`, `attr`, `datacell`, `layout`, `quantization`, `storage`, `io`, `utils`, and `impl`. It accepts the same `CASE` and `SHARD` runtime arguments as `make test`; use the aggregate `make test` path before submitting changes.

## Development environment

- Recommended: use the published Docker development image
  (`vsaglib/vsag:ubuntu`).
- Maintained CI images use the `ci-x86-*` and `ci-aarch64-*` tags and are
  published to both `vsaglib/vsag` on Docker Hub and
  `ghcr.io/antgroup/vsag` on GitHub Container Registry.
- Supported local environments include Ubuntu 20.04+ and CentOS 7+.
- Compiler baseline: GCC 9.4.0+ or Clang 13.0.0+. Development/CI uses GCC 13 and Clang 20.
- Build baseline: CMake 3.18.0+.

See [`DEVELOPMENT.md`](../../DEVELOPMENT.md) for the full setup walkthrough.

## Required tool versions

- `clang-format` must be version **20** exactly.
- `clang-tidy` must be version **20** exactly.

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
  followed by `VSAG_TEST_SEED=424242 bash scripts/testing/test_parallel_bg.sh`
  and `bash scripts/coverage/collect_cpp_coverage.sh`. The C++ report combines the
  non-daily unit and functional suites, but measures only maintained production
  sources under `src/` and `include/`; it excludes the generated
  `src/version.h` and vendored `include/vsag/expected.hpp`. Python sources and
  platform paths not compiled by the coverage job are not part of this metric.

## C++ standard and compiler validation

Full VSAG CPU implementation targets require C++20; installed public headers and the `vsag::vsag` target support C++17 clients. CUDA translation units, Node.js bindings, and standalone Lite stay on C++17. Linux development uses GCC 13 as the primary compiler and Clang 20 as the second compiler. Use separate build directories:

```bash
make test DEBUG_BUILD_DIR=build_gcc13 EXTRA_DEFINED="-DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13"
make test DEBUG_BUILD_DIR=build_clang20 EXTRA_DEFINED="-DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20"
```

The Ubuntu dependency installer requires Ubuntu 24.04+ and installs GCC 13 and LLVM 20 (using apt.llvm.org if needed). Other distributions need a matching toolchain prepared manually. Compiler requirements concern source builds, not the glibc baseline of prebuilt packages. Existing manylinux release baselines and both libstdc++ ABI choices are preserved. Installing a package does not require a C++20 client compiler.

Maintained CI container images use Ubuntu 26.04 with GCC 13 and LLVM 20. Ubuntu 24.04 remains a development and runtime compatibility target. Published packages retain their existing manylinux/glibc baselines; validate their Ubuntu 24.04 runtime compatibility separately from the Ubuntu 26.04 CI build.

TSan CI sets `OPENBLAS_NUM_THREADS=1` because the system OpenBLAS library is not instrumented. This keeps BLAS work on the calling thread while VSAG thread pools and concurrent tests remain enabled; sanitizer failures still fail the job.
