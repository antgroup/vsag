# VSAG Developer Guide

Welcome to the developer guide for VSAG! This guide is designed to provide both new and experienced contributors with a comprehensive resource for understanding the project's codebase, development processes, and best practices.

Whether you're an open-source enthusiast looking to make your first contribution or a seasoned developer seeking insights into the project's architecture, the guide aims to streamline your onboarding process and empower you to contribute effectively.

Let's dive in and explore how you can become an integral part of our vibrant open-source community!

## Development Environment

There are two ways to build and develop the VSAG project now.

### Use Docker(recommended)

![Docker Pulls](https://img.shields.io/docker/pulls/vsaglib/vsag)
![Docker Image Size](https://img.shields.io/docker/image-size/vsaglib/vsag)

```bash
docker pull vsaglib/vsag:ubuntu
```

### or Install Development Requirements

- Operating System:
  - Ubuntu 20.04 or later
  - or CentOS 7 or later
- Compiler:
  - GCC version 9.4.0 or later
  - or Clang version 13.0.0 or later
- Build Tools: 
  - CMake version 3.18.0 or later
  - clang-tidy version 15 EXACTLY (not higher, not lower - required for consistent lint diagnostics)
  - clang-format version 15 EXACTLY (not higher, not lower - required for consistent formatting)
- Additional Dependencies:
  - gfortran
  - python 3.6+
  - omp
  - aio
  - curl

```bash
# for Debian/Ubuntu
$ ./scripts/deps/install_deps_ubuntu.sh

# for CentOS/AliOS
$ ./scripts/deps/install_deps_centos.sh
```

## VSAG Build Tool
VSAG project use the Unix Makefiles to compile, package and install the library. Here is the commands below:
```bash
Usage: make <target>

Targets:
help:                    ## Show the help.
##
## ================ development ================
debug:                   ## Build vsag with debug options.
dev:                     ## Build full developer configuration.
test:                    ## Build and run unit tests.
asan:                    ## Build with AddressSanitizer option.
test_asan: asan          ## Run unit tests with AddressSanitizer option.
tsan:                    ## Build with ThreadSanitizer option.
test_tsan: tsan          ## Run unit tests with ThreadSanitizer option.
clean:                   ## Clear build/ directory.
##
## ================ integration ================
fmt:                     ## Format codes.
cov:                     ## Build unit tests with code coverage enabled.
lint:                    ## Check coding styles defined in `.clang-tidy`.
fix-lint:                ## Fix coding style issues in-place via clang-apply-replacements (destructive).
test_parallel:           ## Run all tests parallel (used in CI).
test_asan_parallel: asan ## Run unit tests parallel with AddressSanitizer option.
test_tsan_parallel: tsan ## Run unit tests parallel with ThreadSanitizer option.
##
## ================ distribution ================
release:                 ## Build vsag with release options.
run-dist-tests:          ## Run distribution tests.
dist-pre-cxx11-abi:      ## Build vsag with distribution options (pre C++11 ABI).
dist-cxx11-abi:          ## Build vsag with distribution options (C++11 ABI).
dist-libcxx:             ## Build vsag using libc++.
pyvsag:                  ## Build a specific Python version wheel. Usage: make pyvsag PY_VERSION=3.10
pyvsag-all:              ## Build wheels for all supported versions.
clean-release:           ## Clear build-release/ directory.
install:                 ## Build and install the release version of vsag.
```

Build target behavior:

- `make debug` builds the default minimal configuration. It does not enable tests, examples, tools, Python bindings, or `mockimpl` unless they are explicitly turned on.
- `make dev` builds the full developer configuration with tests, examples, tools, Python bindings, and `mockimpl` enabled.
- `make test`, `make asan`, `make tsan`, and the related parallel test targets automatically enable tests and `mockimpl`.
- `make release` follows the same minimal defaults as `make debug`. Enable optional components explicitly when needed, for example `make release VSAG_ENABLE_TOOLS=ON`.

## CMake Build Options

VSAG provides several CMake options to customize the build:

### BLAS Library Options

- **`ENABLE_INTEL_MKL`** (default: `OFF`)
  - Enable Intel MKL as the BLAS backend (x86_64 platforms only)
  - When disabled, OpenBLAS is used instead
  - MKL resolution uses `MKL_PATH`, `OMP_PATH`, and `MKL_INCLUDE_PATH` as CMake cache overrides when the libraries are not installed in standard locations

### System Third-Party Dependencies

VSAG can reuse third-party libraries already provided by the host system or by a parent CMake project instead of always building bundled copies.

- **`VSAG_USE_SYSTEM_DEPS`** (default: `AUTO`)
  - `AUTO` — use a system / pre-existing copy when one is found, fall back to the bundled build otherwise.
  - `ON` — require a system copy of every supported dependency; fail configuration if any of them is missing.
  - `OFF` — always build bundled copies, ignoring system packages.

- **`VSAG_USE_SYSTEM_<DEP>`** (default: empty — inherit `VSAG_USE_SYSTEM_DEPS`)
  - Per-dependency override. Set to `AUTO`, `ON`, or `OFF` to override the global policy for a single dependency.

**Currently supported dependencies for system reuse:** `OPENBLAS`.

Additional dependencies will be enabled in follow-up changes.

#### OpenBLAS

When `VSAG_USE_SYSTEM_OPENBLAS=ON` (or inherited via `VSAG_USE_SYSTEM_DEPS=ON`), VSAG looks for OpenBLAS in the following order and stops at the first hit:

1. an `OpenBLAS::OpenBLAS` target already defined by a parent project,
2. `find_package(OpenBLAS CONFIG)`,
3. a manual search for `libopenblas` plus `cblas.h` / `lapacke.h` under common system prefixes.

Example:

```bash
# Install OpenBLAS on Ubuntu/Debian
sudo apt-get install libopenblas-dev liblapacke-dev

# Build with system OpenBLAS
cmake -DVSAG_USE_SYSTEM_OPENBLAS=ON -DENABLE_INTEL_MKL=OFF -B build
cmake --build build
```

The legacy switch **`USE_SYSTEM_OPENBLAS`** (default: `OFF`) is kept as a deprecated alias. When `VSAG_USE_SYSTEM_OPENBLAS` is empty, setting `USE_SYSTEM_OPENBLAS=ON` preserves its previous "try system, fall back to bundled" behaviour (equivalent to `VSAG_USE_SYSTEM_OPENBLAS=AUTO` — it does **not** hard-require a system copy). Prefer the new option in new scripts.

### Third-Party Source Overrides

VSAG downloads its third-party libraries at configure/build time. Each
downloaded dependency honors a `VSAG_THIRDPARTY_<LIB>` **environment variable**
that, when set, is tried before the upstream URL and the project's Aliyun OSS
mirror. The value may be a local filesystem path or any URL (internal HTTP
server, OSS bucket, etc.), which makes it the primary mechanism for offline,
air-gapped, or internal-mirror builds.

- **`VSAG_THIRDPARTY_OPENBLAS`** (representative example)
  - Override the OpenBLAS source archive URL/path used by `ExternalProject_Add`
  - Useful for offline builds, local mirrors, or pre-downloaded archives

The full list of variables (`VSAG_THIRDPARTY_ANTLR4`, `VSAG_THIRDPARTY_BOOST`,
`VSAG_THIRDPARTY_FMT`, …), usage examples, and the exact archives to mirror are
documented in the [Offline / Air-gapped Builds](docs/docs/en/src/development/offline_build.md)
guide. The note next to each `URL_HASH` in `extern/<lib>/<lib>.cmake` is the
source of truth for the exact upstream URL and expected checksum.

### Other Build Options

- **`ENABLE_TESTS`** (default: `OFF`)
  - Build unit tests and functional tests

- **`ENABLE_EXAMPLES`** (default: `OFF`)
  - Build C++ example programs under `examples/cpp/`

- **`ENABLE_TOOLS`** (default: `OFF`)
  - Build tools under `tools/`

- **`ENABLE_PYBINDS`** (default: `OFF`)
  - Build the `_pyvsag` Python extension module

- **`ENABLE_MOCKIMPL`** (default: `OFF`)
  - Build the `mockimpl` targets used by interface and compatibility-style testing

For a complete list of build options, see the `option()` directives in `cmake/VSAGOptions.cmake`.

## Project Structure
- `cmake/`: cmake util functions
- `docker/`: the dockerfile to build develop and ci image
- `docs/`: the design documents
- `examples/`: cpp and python example codes
- `extern/`: third-party libraries
- `include/`: export header files
- `mockimpl/`: the mock implementation that can be used in interface test
- `python/`: the pyvsag package and setup tools
- `python_bindings/`: the python bindings
- `scripts/`: useful scripts
- `src/`: the source codes and unit tests
- `tests/`: the functional tests
- `tools/`: the tools
