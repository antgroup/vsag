# Installation

VSAG can be installed as a C++ library, a Python package (`pyvsag`), or a Node.js/TypeScript
package (`vsag`).

## Using Docker (Recommended for Development)

The official development image includes the full toolchain (GCC 9.4+, CMake 3.18+,
`clang-format`/`clang-tidy` 15, HDF5, etc.):

```bash
docker pull vsaglib/vsag:ubuntu-latest
docker run -it --rm -v $(pwd):/work -w /work vsaglib/vsag:ubuntu-latest bash
```

## Building from Source

### Requirements

- **Operating System**: Ubuntu 20.04+ or CentOS 7+
- **Compiler**: GCC 9.4.0+ or Clang 13.0.0+
- **CMake**: 3.18.0+
- **clang-format / clang-tidy**: exactly version **15** (enforced by `make fmt` / `make lint`)

### Build

```bash
git clone https://github.com/antgroup/vsag.git
cd vsag
make release
```

Other common Makefile targets:

- `make debug` — debug build, includes sanitizers and tests.
- `make dev` — developer configuration: debug + tests + examples + tools.
- `make test` — run unit and functional tests.
- `make cov` — generate coverage report.
- `make pyvsag PY_VERSION=3.10` — build the Python wheel.
- `make dist-pre-cxx11-abi` / `dist-cxx11-abi` / `dist-libcxx` — build redistributable tarballs.

See [Building](../development/building.md) for details.

## Python (pyvsag)

```bash
pip install pyvsag
```

## Node.js / TypeScript

```bash
npm install vsag
```

The bindings source lives under `typescript/` and the npm package name is `vsag`.

## Optional Features

Enable or disable at configure time:

- `VSAG_ENABLE_INTEL_MKL=ON` — Intel MKL acceleration.
- `VSAG_ENABLE_LIBAIO=ON` — Linux AIO for DiskANN async IO.
- `VSAG_ENABLE_TOOLS=ON` — build tools under `tools/` (including `eval_performance`).
