# Release Notes

This page is the canonical changelog for the VSAG 1.x line. For
pre-1.0 history (the 0.15 / 0.16 / 0.18 lines), see
[Releases on GitHub](https://github.com/antgroup/vsag/releases).

VSAG follows [Semantic Versioning 2.0](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- `MAJOR` carries incompatible API or serialization changes.
- `MINOR` adds functionality while remaining backward compatible.
- `PATCH` contains only bug fixes and performance improvements.

The compatibility contract that 1.x will hold to is described in a
dedicated *API Stability* page (planned as a follow-up PR, tracked in
[#2069](https://github.com/antgroup/vsag/issues/2069)). If you are
upgrading from 0.18, start with the
[Migration to VSAG 1.0](migration_to_1_0.md) guide; it covers every
breaking change in one place.

---

## VSAG 1.0.0 — *target: 2026, exact date TBD*

VSAG 1.0 is the first stable major release. It locks in the public
C++/Python/Node.js API surface, the on-disk serialization format, and
the supported index families, so the rest of the 1.x line can ship new
features without breaking your code.

### Highlights

- **Two production-ready index families** — `hgraph` for graph-based
  search, `ivf` for inverted-index search. Both cover in-memory and
  memory-plus-disk hybrid retrieval. Legacy `hnsw` and `diskann` indexes
  are deprecated; see [Migration to VSAG 1.0](migration_to_1_0.md).
- **Comprehensive quantization** — RabitQ (BQ) for extreme compression,
  PQ for flexible compression ratios, SQ4 / SQ8 for standard
  quantization with minor recall loss. All quantizers can be combined
  with HGraph or IVF.
- **First-class non-FP32 inputs** — INT8, BF16, FP16 and sparse vectors
  are accepted as primary input types, not just emulated on top of FP32.
- **Multi-platform SIMD** — x86_64 (SSE / AVX / AVX2 / AVX-512 / AMX) and
  ARM (NEON / SVE) backends, plus optional Intel MKL and OpenBLAS for
  matrix kernels.
- **Per-tenant resource isolation** — per-index allocators and
  injectable thread pools make it practical to host multiple tenants in
  the same process.
- **New unified search API** — `Index::SearchWithRequest(SearchRequest)`
  replaces the deprecated `KnnSearch(query, k, SearchParam&)` overload,
  with explicit per-search allocator and reasoning support.
- **Stable public headers** — every header under `include/vsag/` is now
  guaranteed self-contained; the 1.x line will not silently change
  public ABI surface within a minor version.

### Indexes

- **HGraph** — recommended graph index for most workloads.
  - Reverse-edge support, optional duplicate-distance threshold, and
    `hops_limit` search parameter for tightly bounded latency budgets.
  - `Remove` is now supported on graph indexes (mark-remove plus
    `ShrinkAndRepair` reclamation with timeout).
  - Built-in `Train` API plus ODescent builder for offline graph
    construction; see [Build and Train](../advanced/build_and_train.md).
  - Reasoning instrumentation: pass a `QueryContext` to collect
    per-search diagnostics (visited nodes, hop count, distance
    computations) without changing the result format.
- **IVF** — recommended inverted index for batched / large-K queries.
  Supports the same set of quantizers as HGraph and integrates with the
  per-search allocator.
- **SINDI** — sparse inverted index with built-in term-ID remapping for
  sparse vocabularies, vector update support, and analyzer hooks.
- **Pyramid** — hierarchical inverted index with deduplication support,
  static optimization, `topk_factor` parameter on the base search
  parameter class, and a `PyramidAnalyzer` for index statistics.
- **BruteForce** — exact baseline with parallel range search.
- **WARP** — multi-vector brute-force backend, migrated to the new
  MultiVectors API.

### Quantization

- **RabitQ (BQ)** with extend-bit and split-base reorder support, plus
  dedicated SIMD kernels.
- **PQ / SQ4 / SQ8** as standard memory/recall trade-offs.
- **Scalar quantizer** hardened against NaN encoding.
- **Quantization Transform** advanced page documenting the full
  pipeline; see [Quantization Transform](../advanced/quantization_transform.md).

### Data types and dataset support

- **FP32 / INT8 / BF16 / FP16** vector inputs as first-class formats.
- **Sparse vectors** end-to-end (SINDI + sparse HDF5 dataset helpers in
  `pyvsag`).
- **MultiVector datasets** as a first-class type; eval tooling and
  WARP both consume the new MultiVectors API directly.
- **`extra_info`** payload stored alongside vectors; see the user
  guide on `extra_info` for HGraph.

### Search API

- New `SearchRequest` / `Index::SearchWithRequest` pair as the primary
  search entry point. Carries the query dataset, k, optional filter,
  reasoning hook, and a per-search allocator in a single struct so the
  hot path no longer mixes positional and out-parameter arguments.
- `SearchParam` and the old `KnnSearch(query, k, SearchParam&)` overload
  remain available but are marked `[[deprecated]]`. The full mapping is
  in [Migration to VSAG 1.0](migration_to_1_0.md).
- `CalDistanceById` (batch) is being renamed to `CalcDistancesById` with
  consistent return semantics; the legacy name remains as a wrapper. See
  [Calculate Distance by ID](../advanced/calc_distance_by_id.md) and
  issue [#2068](https://github.com/antgroup/vsag/issues/2068).
- Range search variant (`SearchWithRequest` with radius semantics) is
  available across HGraph, IVF, and BruteForce.

### Platforms and packaging

- **x86_64 SIMD:** SSE, AVX, AVX2, AVX-512, plus AMX backends (SQ8U INT8
  IP and BF16 GEMM for KMeans).
- **ARM SIMD:** NEON and SVE.
- **macOS (Darwin)** is a supported build platform.
- **Intel MKL** is now opt-in (`VSAG_ENABLE_INTEL_MKL=OFF` /
  CMake `ENABLE_INTEL_MKL=OFF` by default).
- **OpenBLAS** can be linked from the system instead of the bundled
  copy (`VSAG_ENABLE_SYSTEM_OPENBLAS=ON`).
- Third-party downloads support custom mirror URLs for environments
  without direct GitHub access.

### Resource isolation and observability

- **Per-index allocators** — pass a custom `Allocator` through
  `IndexCommonParam` and every container under that index honors it.
- **Injectable thread pools** — supply your own thread pool for both
  build and search.
- **Per-search allocator** — see
  [Per-Search Allocator](../advanced/search_allocator.md).
- **Search statistics** — `io_cnt`, `io_time_ms`, and other counters
  exposed through `SearchRequest` reasoning.
- **Memory and introspection** — see
  [Memory](../advanced/memory.md) and
  [Index Introspection](../advanced/introspection.md).
- **Index lifecycle** — [Index Lifecycle Management](../advanced/index_lifecycle.md)
  documents how to
  add, remove, mark-remove, and rebuild safely under load.

### Tooling and ecosystem

- **`pyvsag`** Python bindings extended to cover the full index
  surface, including sparse HDF5 helpers and pyramid export.
- **Node.js / TypeScript bindings** — `vsag` npm package with
  quickstart examples in `examples/typescript/`.
- **`eval_performance`** tool supports multi-vector datasets and a
  configurable search query count.
- **HTTP monitor server** built on `cpp-httplib` for exposing live
  index metrics.

### Breaking changes (vs. 0.18)

The full list with code-diff examples lives in
[Migration to VSAG 1.0](migration_to_1_0.md). Headline items:

1. `hnsw` and `diskann` index types are deprecated. Use `hgraph` (or
   the hybrid memory-disk configuration) and `ivf` respectively.
2. `SearchParam` and `Index::KnnSearch(query, k, SearchParam&)` are
   deprecated in favor of `SearchRequest` /
   `Index::SearchWithRequest(SearchRequest)`.
3. `CalDistanceById` (batch) returns `-1` for invalid IDs and is being
   renamed to `CalcDistancesById`. The old name continues to work for
   one minor cycle.
4. `VSAG_ENABLE_INTEL_MKL` defaults to `OFF`. Set it explicitly if you
   were relying on MKL.
5. Several HGraph defaults changed (`max_degree=64`,
   `ef_construction=400`, `graph_type="nsw"`); `support_remove`,
   `support_duplicate`, and `store_raw_vector` default to `OFF`.

Serialization: 0.18 snapshots are **not** guaranteed to deserialize on
1.0; rebuild on the new release. See *Migration*.

### Known issues

- *To be filled in during the 1.0 RC cycle.*

### Acknowledgments

VSAG 1.0 is the result of contributions from the Ant Group VSAG team
and the wider open-source community. Full per-release contributor
credits remain on the
[GitHub Releases page](https://github.com/antgroup/vsag/releases).

---

## Getting a specific version

### C++ / source

```bash
git checkout v1.0.0
make release
```

### Python

```bash
pip install pyvsag==1.0.0
```

### Node.js / TypeScript

```bash
npm install vsag@1.0.0
```

## Upgrade guidance

- Read [Migration to VSAG 1.0](migration_to_1_0.md) before upgrading
  from any 0.x release.
- Read the **Breaking Changes** section of each future major release
  before crossing major versions.
- When the serialization format changes, validate deserialization
  compatibility in a staging environment first.
- Roll out gradually in production and use the
  [performance evaluation tool](eval.md) to compare recall and latency
  against your existing baseline.

## See also

- [Migration to VSAG 1.0](migration_to_1_0.md)
- [Roadmap](roadmap_2025.md)
- [Best Practices](best_practices.md)
- [Performance](performance.md)
