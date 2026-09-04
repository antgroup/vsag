# VSAG Lite

VSAG Lite is a reduced-footprint build for dense-vector search. It preserves the
public C++ API, so applications using retained indexes can switch between Full
and Lite without source changes.

## Feature set

| Area | Lite status |
| --- | --- |
| HGraph, LazyHGraph, BruteForce and WARP | Included |
| Add, build, search, update and remove | Included |
| Serialize and deserialize | Included |
| Mark-deletion persistence | Included |
| IVF, Pyramid, SINDI and SINDI v2 | Excluded |
| SINDI and sparse-vector-only DataCells | Excluded |

Creating an excluded index through `Factory::CreateIndex` returns an error.

## Build and test

```bash
make lite
make test-lite
```

Equivalent CMake configuration:

```bash
cmake -S . -B build-lite -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_LITE=ON -DENABLE_LITE_TESTS=ON
cmake --build build-lite --parallel
ctest --test-dir build-lite --output-on-failure
```

`ENABLE_TESTS` is the complete Full suite and cannot be combined with Lite.
Use `ENABLE_LITE_TESTS` for the standalone compatibility smoke test.

## Compatibility and persistence

Lite retains the same headers, factory, datasets, result objects and BinarySet
API as Full. Marked removals are stored in an optional HGraph label-metadata
block. Readers remain compatible with older files that lack the block, although
old deletion information that was never stored cannot be recovered.

HGraph stores dense codes, graph adjacency, labels, visited-list state and
optional precise-reorder data. LazyHGraph starts flat and transitions to HGraph
at its configured threshold. Lite removes unused factories and dedicated
DataCells; it does not change retained indexes' runtime layout or accuracy.

## Benchmark

```bash
make benchmark-lite
./build-lite-release/benchs/lite/vsag_lite_benchmark 10000 64 100
```

Arguments are vector count, dimension and query count. JSON output contains
build/serialize/load time, serialized bytes, QPS, P50/P99 and Recall@10. Compare
identical Release builds and report the median of repeated runs. Measure peak RSS
externally with `/usr/bin/time -v`. Sparse/SINDI deployments require Full.
