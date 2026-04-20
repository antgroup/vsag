# VSAG Documentation

VSAG is a high-performance, production-grade vector indexing library for similarity search. It
powers vector retrieval in OceanBase and other projects at Ant Group, and is released under the
Apache 2.0 license.

## Features

- **Multiple index types**: `hnsw`, `hgraph`, `diskann`, `ivf`, `pyramid`, `sindi`, `brute_force`,
  covering in-memory, memory-disk hybrid, sparse and multi-tenant scenarios.
- **Rich quantization**: fp32 / fp16 / bf16 / int8 / sq8 / sq4 / pq, with SIMD dispatch on x86_64
  and AArch64.
- **Advanced capabilities**: range search, filtered search, serialization, conjugate graph
  enhancement, online `Tune`-based optimization, custom allocator / thread pool.
- **Language bindings**: native C++, Python via `pyvsag`, Node.js / TypeScript via the npm package
  `vsag`.

## How to Read This Documentation

- **User Guide** — start here if you are new to VSAG.
- **Developer Guide** — building from source, running tests, contributing.
- **Advanced Features** — deep dives into specific capabilities.
- **Resources** — release notes, community links, parameter reference, benchmarks.

The Chinese version of the same documentation is available under `docs/docs/zh/`.

## Project Links

- Source: <https://github.com/antgroup/vsag>
- Issues: <https://github.com/antgroup/vsag/issues>
- Releases: <https://github.com/antgroup/vsag/releases>
