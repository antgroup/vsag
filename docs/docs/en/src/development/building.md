# Building

This page documents how to build VSAG from source.

## Prerequisites

- **OS**: Ubuntu 20.04+ or CentOS 7+
- **Compiler**: GCC 9.4.0+ or Clang 13.0.0+
- **CMake**: 3.18.0+
- **clang-format / clang-tidy**: exactly version 15 (enforced)
- Optional: HDF5 (for `tools/eval/eval_performance`), libaio (for DiskANN async IO), Intel MKL.

We recommend using the official Docker dev image, which already contains the matching toolchain:

```bash
docker pull vsaglib/vsag:ubuntu-latest
```

## Makefile Targets

Running `make help` prints a concise list; the most common targets are:

```text
debug       Build debug binaries (with sanitizers)
release     Build release binaries
dev         Developer build: debug + tests + examples + tools
test        Run all unit and functional tests
cov         Generate coverage report
asan        Build and run with AddressSanitizer
fmt         Run clang-format
lint        Run clang-tidy
fix-lint    Apply clang-tidy fix-its
pyvsag      Build pyvsag for a specific Python version (PY_VERSION=...)
pyvsag-all  Build pyvsag wheels for all supported Python versions
dist-pre-cxx11-abi  Build redistributable tarball (pre-C++11 ABI)
dist-cxx11-abi      Build redistributable tarball (C++11 ABI)
dist-libcxx         Build redistributable tarball (libc++)
clean       Remove build trees
```

## Step-by-Step

```bash
git clone https://github.com/antgroup/vsag.git
cd vsag
make release
```

Resulting binaries:

- Library: `build-release/src/libvsag.{a,so}`
- Examples: `build-release/examples/cpp/`
- Tools: `build-release/tools/`

## Environment Variables / CMake Options

| Option | Default | Effect |
|--------|---------|--------|
| `VSAG_ENABLE_INTEL_MKL` | `ON` when MKL detected | Use Intel MKL for BLAS kernels |
| `VSAG_ENABLE_LIBAIO` | `ON` on Linux | Enable DiskANN async IO via libaio |
| `VSAG_ENABLE_TOOLS` | `ON` | Build utilities under `tools/` |
| `CMAKE_BUILD_TYPE` | driven by Makefile target | Debug / Release |

Pass options through CMake directly when not using `make`:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DVSAG_ENABLE_INTEL_MKL=ON
cmake --build build-release -j
```

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
