# Building

This page documents how to build VSAG from source.

## Build phase and cache reporting

The collector preserves Ninja history for clean, warm-cache, and unchanged no-op phases.
A no-op is verified only when available Ninja telemetry records zero build edges and the
command succeeds; missing telemetry is explicitly unverified. Every observed category,
including nested production modules such as datacell, tools, examples, and unclassified
work, appears in JSON and Markdown. Multi-output edges count once. Cumulative edge time
sums overlapping parallel durations and is not wall time; nested ExternalProject work
is included in its parent edge rather than individually timed.

Cacheable-request hit rate is `hits / (hits + misses)`. Available uncacheable reasons,
including `could_not_use_precompiled_header`, are reported separately; missing counters
and zero denominators are `null` in JSON and `n/a` in Markdown. Overall cache coverage is
unavailable because compiler invocations bypassing ccache are not measured. Raw counters
remain in JSON. This reporting does not change PCH or cache correctness settings.

Dependency preparation timers cover source preparation and archive restoration before
configure, not all dependency work: further downloads, configuration, compilation and
installation occur in configure/build phases. Peak RSS from `/usr/bin/time -v` is the
maximum resident set size reported for the timed command and waited-for children, not
the sum of simultaneously resident build processes.

## Prerequisites

- **OS**: Ubuntu 20.04+, CentOS 7+, or macOS 14+ on Apple Silicon
- **Compiler**: GCC 9.4.0+, Clang 13.0.0+, or Apple Clang from Xcode Command Line Tools
- **CMake**: 3.18.0+
- **Ninja**: preferred when available; Unix Makefiles are used as the fallback
- **clang-format / clang-tidy**: exactly version 15 (enforced)
- Optional: HDF5 (for `tools/eval/eval_performance`), libaio (for the `async_io` data-cell backend),
  liburing (for `uring_io` on Linux), Intel MKL.

We recommend using the official Docker dev image, which already contains the matching toolchain:

```bash
docker pull vsaglib/vsag:ubuntu
```

## Makefile Targets

Running `make help` prints a concise list; the most common targets are:

```text
debug       Build debug binaries (no sanitizers; tests/tools/examples OFF by default)
release     Build release binaries (tests/tools/examples OFF by default)
dev         Developer build: debug + tests + tools + examples
test        Build with tests enabled and run unit + functional tests
cov         Build with coverage instrumentation enabled
asan        Build with AddressSanitizer
tsan        Build with ThreadSanitizer
fmt         Run clang-format
lint        Run clang-tidy
fix-lint    Apply clang-tidy fix-its in-place (destructive)
pyvsag      Build pyvsag for a specific Python version (PY_VERSION=...)
pyvsag-all  Build pyvsag wheels for all supported Python versions
dist-pre-cxx11-abi  Build redistributable tarball (pre-C++11 ABI)
dist-cxx11-abi      Build redistributable tarball (C++11 ABI)
dist-libcxx         Build redistributable tarball (libc++)
clean       Remove build trees
```

These targets prefer Ninja when a usable `ninja` executable is available. Otherwise they fall back
to Unix Makefiles. An explicit generator always wins; for example, use
`make debug CMAKE_GENERATOR='Unix Makefiles'` to request the fallback directly. Because CMake build
trees are generator-specific, run the matching clean target before changing the generator for an
existing build directory.

For configure/build targets, pass `DEBUG_BUILD_DIR` as a plain path, for example
`make asan DEBUG_BUILD_DIR="custom build"`. The shell removes these command-line quotes;
the recipes quote the path when invoking CMake. Do not embed literal quote characters in the variable.

## Dependency size metrics

The build metrics JSON uses schema version 3. Each dependency's `local_bytes` (the Markdown
report's **Local size**) is an aggregate of file sizes, not allocated disk space. It replaces
`source_bytes` and `build_bytes`: HDF5 and OpenBLAS build in their source trees, and ANTLR4's
binary directory is nested under its source tree, so those categories cannot be separated reliably.

For ExternalProject dependencies, the aggregate counts the entire dependency prefix once,
including source, build, install, and any metadata inside that prefix. Shared `BUILD_INFO_DIR`
files (by default `.vsag-build-info`: temporary files, stamps, logs, and metadata) are excluded
from both `local_bytes` and the preparation table's `external_build_bytes`; neither measures all
ExternalProject storage. For FetchContent dependencies, it
sums the sibling `<name>-src` and `<name>-build` trees. Symlink entries are excluded. Separately
cached ExternalProject archives are reported in the preparation table, not added to `local_bytes`.
System dependencies show zero because host installations are not measured; unclassified or missing
trees also contribute zero. This is a snapshot of the measured local trees, not a source-only size
or a measurement of generated build artifacts alone.

## Step-by-Step

```bash
git clone https://github.com/antgroup/vsag.git
cd vsag
./scripts/deps/install_deps.sh
make release
```

The dependency script selects the matching Linux distribution or macOS installer. macOS support
currently targets the arm64 C++ build; published archives and Python wheels remain Linux-focused.

Resulting binaries from a plain `make release`:

- Library: `build-release/src/libvsag.{a,so}`

Examples and tools are not built by default. To include them, either use `make dev`, or enable
the corresponding Makefile variables (`VSAG_ENABLE_EXAMPLES=ON`, `VSAG_ENABLE_TOOLS=ON`) or the
underlying CMake cache options (`-DENABLE_EXAMPLES=ON`, `-DENABLE_TOOLS=ON`).

## Environment Variables / CMake Options

The Makefile exposes a few `VSAG_ENABLE_*` environment variables that are translated into CMake
cache options (`ENABLE_*`). Defaults below reflect a plain `make release`.

| Makefile env var | CMake option | Default | Effect |
|------------------|--------------|---------|--------|
| `VSAG_ENABLE_INTEL_MKL` | `ENABLE_INTEL_MKL` | `OFF` | Use Intel MKL for BLAS kernels |
| `VSAG_ENABLE_LIBAIO` | `ENABLE_LIBAIO` | `ON` on Linux | Enable the `async_io` data-cell backend via libaio |
| `VSAG_ENABLE_TOOLS` | `ENABLE_TOOLS` | `OFF` | Build utilities under `tools/` |
| `VSAG_ENABLE_EXAMPLES` | `ENABLE_EXAMPLES` | `OFF` | Build sample programs under `examples/cpp/` |
| n/a | `ENABLE_LIBURING` | `OFF` | Enable the Linux `uring_io` backend when liburing is installed |
| n/a | `ENABLE_CUDA` | `OFF` | Build the optional CUDA backend; requires an NVIDIA toolkit |
| n/a | `CMAKE_BUILD_TYPE` | driven by Makefile target | Debug / Release |

When invoking CMake directly instead of using `make`, use the underlying CMake cache option names:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_INTEL_MKL=ON
cmake --build build-release -j
```

To enable `io_uring`, install liburing and add `-DENABLE_LIBURING=ON`. On non-Linux systems or
when liburing is unavailable, VSAG compiles without native `io_uring` support; a configuration
that requests `uring_io` then logs a one-time warning and falls back to `buffer_io`.

## Offline / Air-gapped Builds

VSAG downloads its third-party libraries at configure/build time. In offline or
restricted-network environments, set the per-dependency `VSAG_THIRDPARTY_*`
environment variables to fetch each archive from a local path or an internal
mirror (internal HTTP server, object storage, etc.). See
[Offline / Air-gapped Builds](offline_build.md) for the full list of variables
and worked examples.

## Python Wheel (pyvsag)

```bash
make pyvsag PY_VERSION=3.10
# Or build all supported versions in parallel:
make pyvsag-all
```

Wheels are emitted under `python/dist/`.

## Distribution Tarballs

For ABI-compatible redistribution use one of:

```bash
make dist-pre-cxx11-abi   # _GLIBCXX_USE_CXX11_ABI=0
make dist-cxx11-abi       # _GLIBCXX_USE_CXX11_ABI=1
make dist-libcxx          # libc++ (Clang)
```

The produced tarballs contain headers, static/shared libraries, and version metadata.

## Release Publishing

To publish a new GitHub Release, use the `Build and Publish Release` workflow in the GitHub
Actions tab and run it manually with:

- `branch`: the branch, tag, or commit SHA to release from
- `tag_name`: the new release tag, such as `v1.0.0`
- `prerelease`: whether to mark the release as a prerelease

For a local dry run of the same packaging script, run:

```bash
COMPILE_JOBS=6 bash ./scripts/release/dist.sh
```

You can increase `COMPILE_JOBS` if your machine has enough memory, but the default is conservative
to avoid out-of-memory failures in CI runners.
