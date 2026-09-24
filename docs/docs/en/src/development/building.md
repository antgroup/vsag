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
## PR ASan builds and standalone build benchmarks

> **Legacy no-op measurements are not trustworthy until [#2899](https://github.com/antgroup/vsag/pull/2899).** The standalone workflow deliberately fails its preflight on collectors without the supported schema 3 `true_noop` reporting. Separating PR builds does not itself fix the collector. While the prerequisite is unmerged, scheduled runs fail clearly instead of publishing misleading legacy baselines.

PR CI's **ASan Build X86** runs one `make asan COMPILE_JOBS=3`, with Ninja, compile commands, system OpenBLAS, examples and tools enabled. The Makefile still enables tests, Sanitize and ASan/UBSan; changed-source lint, downstream tests, compatibility checks and build artifacts retain their existing contracts. `ccache --zero-stats` resets counters, **not cached objects**. The build step logs elapsed time, and an always-run step prints verbose/fallback cache diagnostics without hiding a build failure. ccache stays enabled in its default directory on the disposable GitHub-hosted VM; no compiler cache is restored or saved between jobs. PR X86 dependency archives intentionally change from restore-and-save to **restore-only** (`actions/cache/restore@v4`), preventing PR contexts from writing archives back to the shared cache; this policy is not limited to the standalone benchmark.

The separate **Build Performance Benchmark** workflow (`.github/workflows/build_performance.yml`) runs weekly on Mondays at 03:23 UTC on upstream `main`, or manually from Actions. It is not a PR required check and uses separate concurrency, an Ubuntu 22.04 hosted runner, a 180-minute timeout and 14-day report retention. Once the workflow is on the default branch, for example:

```bash
gh workflow run build_performance.yml --ref main -f ref=main
# Or use a full 40-character commit SHA reachable from this repository's main:
gh workflow run build_performance.yml --ref main -f ref=<full-main-history-sha>
```

The workflow must itself be dispatched from `main` (other workflow branches are skipped). It explicitly checks out `antgroup/vsag` upstream `main`, accepts only `main` or a full SHA in that fetched upstream history, and records the actual checked-out SHA (not the dispatch event SHA) and the main baseline in `target.txt`. PR refs, arbitrary branches and unmerged feature commits are intentionally rejected. Therefore a draft migration branch cannot benchmark itself through this entry point; hosted dispatch and end-to-end validation remain pending until the workflow and supported collector are available on main. Scheduled execution is disabled on forks.

Security is deliberately conservative: read-only contents permission, no business secrets, no persisted checkout credentials, quoted environment input with SHA/ancestry validation, and a fixed workflow-owned Ubuntu package list instead of privileged installation scripts from the selected ref. Packages match the PR's OpenBLAS-only host setup; keep these lists synchronized. Dependencies are checked out at pinned revisions by the existing composite action; download archives are restored **without saving**. There is no compiler-cache restore/save. The fresh benchmark job owns `build/` and `.benchmark-ccache`; `--clear-ccache` never touches the normal PR cache. Default `build/` retains compatibility with the existing collector/Makefile API.

“Cold” means a clean build with an empty **compiler cache**, not a cold network or dependency-source download. The collector's existing CLI measures configure → cold build → clean → warm ccache rebuild → no-op without another clean between warm and no-op. Reports include JSON, Markdown, phase logs, peak RSS, Ninja statistics and ccache details; `toolchain.txt` records tool versions and actual workflow environment values, while `collector-command.txt` records the shell-escaped collector invocation executed by the workflow. The collector's phase logs remain authoritative for the underlying build commands/options; dependency preparation time/cache metadata remain in its report. The compiler-cache key is descriptive metadata, not evidence of restored cache. Parallel cumulative edge time is not wall time, and cache hit rates do not cover bypassed/PCH/link work; missing telemetry must not be read as zero.

The preflight checks the known schema 3 capability without executing the collector; it is not a proof of measurement correctness. After collection, the workflow additionally requires success, available Ninja telemetry, `true_noop: true` and zero build edges. #2899 can report a false no-op without failing collection, so this workflow fails such runs and preserves their raw reports for diagnosis. Residual generated/check edges require inspecting the logs and the collector's build-edge classification. Unknown future schemas fail closed until reviewed. This trades temporary red/absent baselines for avoiding confidently mislabeled measurements; do not bypass the gate to restore legacy reports. Recheck #2899's final API and tests after merge, including missing logs, rewritten history, multi-output edges and source/header rebuilds. This migration does not modify the collector, PCH, compiler flags or persistent caching.

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
