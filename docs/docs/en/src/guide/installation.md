# Installation

VSAG can be installed as a C++ library, a Python package (`pyvsag`), or a Node.js/TypeScript
package (`vsag`).

## Using Docker (Recommended for Development)

The official development image includes the full toolchain (GCC 9.4+, CMake 3.18+,
`clang-format`/`clang-tidy` 15, HDF5, etc.):

```bash
docker pull vsaglib/vsag:ubuntu
docker run -it --rm -v $(pwd):/work -w /work vsaglib/vsag:ubuntu bash
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

- `make debug` — plain debug build (no sanitizers; tests/tools/examples disabled by default).
- `make dev` — developer configuration: debug + tests + tools + examples.
- `make test` — build with tests enabled and run the unit + functional suites.
- `make cov` — build with coverage instrumentation; run tests afterwards to generate the report.
- `make asan` / `make tsan` — sanitizer-enabled builds.
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

Enable or disable at CMake configure time with these cache options:

- `ENABLE_INTEL_MKL=ON` — Intel MKL acceleration.
- `ENABLE_LIBAIO=ON` — Linux AIO for DiskANN async IO.
- `ENABLE_TOOLS=ON` — build tools under `tools/` (including `eval_performance`).
- `ENABLE_EXAMPLES=ON` — build sample programs under `examples/cpp/`.

If you build through the project Makefile, the corresponding environment variables are
`VSAG_ENABLE_INTEL_MKL=ON`, `VSAG_ENABLE_LIBAIO=ON`, `VSAG_ENABLE_TOOLS=ON`, and
`VSAG_ENABLE_EXAMPLES=ON`.
