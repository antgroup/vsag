# HGraph prefetch benchmark

This synthetic benchmark creates deterministic 128-dimensional, clustered vectors and 16,384
queries. Save the index once, then reuse that file when comparing VSAG libraries. Each measured
round prints QPS and a top-10 ID checksum. The checksum must match across variants.

From the repository root, after `make release`:

```sh
g++ -std=c++17 -O3 -Iinclude tools/hgraph_prefetch_benchmark/main.cpp \
  -Lbuild-release/src -Wl,-rpath,"$PWD/build-release/src" -lvsag -pthread \
  -o build-release/hgraph_prefetch_benchmark

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  build-release/hgraph_prefetch_benchmark build sq8 /tmp/hgraph-sq8.index 200000 1 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 24 \
  build-release/hgraph_prefetch_benchmark bench sq8 /tmp/hgraph-sq8.index 200000 1 5
```

Use `fp32` or `rabitq` in place of `sq8` for the other code paths. The RaBitQ case uses fused
`3+5` bit storage and one-bit search. For an eight-thread run use `taskset -c 24-31` and pass `8`
as the `THREADS` argument. Alternate baseline and candidate runs to reduce bias from changing
machine load. Keep `N`, the index file, and thread affinity identical for each pair.
