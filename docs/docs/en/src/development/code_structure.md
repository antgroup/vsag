# Code Structure

This page gives a quick tour of the VSAG repository layout.

## Top-Level Directories

| Path | Contents |
|------|----------|
| `include/vsag/` | Public C++ headers (`index.h`, `engine.h`, `resource.h`, `constants.h`, ...) |
| `src/` | Core implementation and unit tests |
| `tests/` | Functional tests (Catch2) |
| `examples/cpp/` | C++ end-to-end examples |
| `examples/python/` | Python examples |
| `python/` | `pyvsag` packaging |
| `python_bindings/` | pybind11 bindings |
| `typescript/` | Node.js / TypeScript bindings (npm package `vsag`) |
| `tools/` | Utilities such as `eval_performance`, `analyze_index`, `check_compatibility` |
| `extern/` | Third-party dependencies (do not modify unless necessary) |
| `docs/` | Documentation (this site) and blog posts |
| `cmake/` | CMake modules |

## Core Subsystems (inside `src/`)

- **index**: concrete index implementations (HNSW, HGraph, DiskANN, IVF, Pyramid, SINDI, ...).
- **quantization**: FP32 / FP16 / BF16 / SQ4 / SQ8 / PQ quantizers with SIMD dispatch.
- **graph**: shared graph data structures used by HNSW/HGraph/DiskANN.
- **storage**: binary/reader sets, streaming serialization.
- **allocator / thread pool**: user-pluggable resource management.
- **simd**: cascaded SIMD dispatch for x86_64 and AArch64.

## Naming Conventions

- Public API: `vsag` namespace, in `include/vsag/`.
- Implementation: `src/`, same namespace unless the file explicitly needs otherwise.
- File extension: `.cpp` (not `.cc`).

## Build Artifacts

`make debug` / `make release` / `make dev` produce build trees:

- `build-debug/`
- `build-release/`
- `build-dev/`

Each contains the test binaries, example executables, and libraries.
