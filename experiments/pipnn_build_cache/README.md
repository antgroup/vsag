# PiPNN build-cache research prototype

This branch preserves the standalone experiments, not a production `ImportCache` implementation. The targets are disabled by default. No public API, normal PiPNN build path, or persisted index format changes.

The branch starts at `bc2b520`, the engineering optimization submitted in PR #3271. The cache research is a separate commit and does not update that PR.

## Algorithm

1. Freeze the old leader partition and route new vectors with the same leaders. Keep stable old IDs and unchanged construction vectors.
2. Cache old final adjacency and reservoir alternatives. Final neighbors alone cannot supply replacements after deletion.
3. Compute leaf Top-K only for new nodes, using new-to-old and new-to-new dot products.
4. Mark old nodes affected by deleted selected neighbors or incoming new-node candidates. Restore their still-live old reservoir candidates, add the new candidates, and repair locally.
5. Preserve untouched old adjacency. For affected nodes, reuse an identical reservoir or a proven unchanged pruning prefix; otherwise run the original pruning.

The warm-start algorithm allows a different graph. Recall and search cost must be checked. `REUSE_PROOF.md` documents the stricter conditions and their limitations.

## Build

Use the repository's normal dependency/toolchain setup. Enable this development-only option in a Release CMake configuration:

```sh
cmake -S . -B build-pipnn-cache -DCMAKE_BUILD_TYPE=Release \
  -DVSAG_ENABLE_PIPNN_CACHE_EXPERIMENT=ON
cmake --build build-pipnn-cache --target \
  pipnn_cache_derived pipnn_cache_exact pipnn_cache_warm pipnn_cache_search \
  pipnn_cache_correctness pipnn_cache_certificate_check -j4
```

Use the same BLAS/compiler settings for comparisons. `generate.py` creates instrumented builder copies and helper translation units in the build directory. It does not rewrite `src/`; it checks expected source markers and fails on an unsupported builder revision. Internal classes are exposed only in generated experiment copies. These copies are linked against the ordinary static library, as in the original experiments.

Targets are under `build-pipnn-cache/experiments/pipnn_build_cache/`. The prototype borrows internal APIs and is intentionally revision-sensitive. It is not installed with the library.

## Data and smoke validation

Obtain the standard TexMex SIFT archive, unpack `sift_base.fvecs`, `sift_learn.fvecs`, and `sift_query.fvecs` into a local data directory. Archive provenance and hashes are in `results/dataset_source.json` and `results/provenance.json`. Data, binary caches, exported graphs and executables are excluded from this branch.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
BIN=build-pipnn-cache/experiments/pipnn_build_cache
DATA=/path/to/sift
mkdir -p /tmp/pipnn-cache-smoke /tmp/pipnn-cache-warm
python3 -m unittest discover -s experiments/pipnn_build_cache -p test_generate.py -v
"$BIN/pipnn_cache_correctness"
"$BIN/pipnn_cache_certificate_check"
"$BIN/pipnn_cache_derived" 10000 "$DATA" "$DATA/sift_learn.fvecs" /tmp/pipnn-cache-smoke 1
"$BIN/pipnn_cache_warm" 10000 "$DATA" "$DATA/sift_learn.fvecs" /tmp/pipnn-cache-smoke/cache.bin /tmp/pipnn-cache-warm
```

The deletion and tie/K-boundary check covers 228 cases. The certificate check covers 9,000 randomized reservoir mutations, including equal/duplicate vectors and alpha 1.0/1.2/1.5. The 10k derived run also checks frozen old leaf membership/order against the original partition.

## Million-vector reproduction

```sh
mkdir -p /tmp/pipnn-cache-derived1m /tmp/pipnn-cache-exact1m \
  /tmp/pipnn-cache-warm1m /tmp/pipnn-cache-search1m
"$BIN/pipnn_cache_derived" 1000000 "$DATA" "$DATA/sift_learn.fvecs" /tmp/pipnn-cache-derived1m 1
"$BIN/pipnn_cache_exact" 1000000 "$DATA" "$DATA/sift_learn.fvecs" /tmp/pipnn-cache-derived1m/cache.bin /tmp/pipnn-cache-exact1m
"$BIN/pipnn_cache_warm" 1000000 "$DATA" "$DATA/sift_learn.fvecs" /tmp/pipnn-cache-derived1m/cache.bin /tmp/pipnn-cache-warm1m
"$BIN/pipnn_cache_search" 1000000 "$DATA" "$DATA/sift_learn.fvecs" /tmp/pipnn-cache-derived1m/cache.bin /tmp/pipnn-cache-search1m
```

Each command runs 90%, 95%, and 99% overlap. Day one is SIFT base 1M; day two always contains 1M vectors, retaining old source IDs selected with seed 20261104 and adding a prefix of independent SIFT learn vectors. This is cross-build ID overlap, not an intra-dataset duplicate-vector ratio. Changed content under an existing ID is not supported. Source-ID recipes are emitted as native little-endian uint32 files on the tested platform.

Create output directories first. These commands write large cache/graph files; run them separately from timing other workloads. The search executable saves paired graphs for later query experiments. On macOS `ru_maxrss` is bytes; the exploratory RSS field in `derived.cpp` is macOS-specific and must be normalized before interpreting Linux results.

## Recorded results

Apple M4 16 GiB, macOS arm64, AppleClang 21 Release `-O3`, single builder/BLAS thread; degree 32, alpha 1.2, K=5, reservoir 64, max/min leaf=1024/64, fanout=[10,2]. Each construction row is one paired run. Results below are the original measurements before packaging/formatting, not new million-vector reruns of this branch commit.

| Cross-build overlap | Incremental leaf-cache + full prune | Final-graph warm start | Core speedup |
|---|---:|---:|---:|
| 90% | 41.674 s | 20.760 s | 2.01x |
| 95% | 34.069 s | 17.633 s | 1.93x |
| 99% | 25.559 s | 8.887 s | 2.88x |

At ef=256, 512 exact-truth queries and four paired timed search rounds:

| Overlap | Reference Recall@10 | Warm Recall@10 | Reference QPS | Warm QPS |
|---|---:|---:|---:|---:|
| 90% | 99.9414% | 99.9414% | 2,193 | 2,197 |
| 95% | 99.9219% | 99.9219% | 2,188 | 2,197 |
| 99% | 99.9609% | 99.9414% | 2,196 | 2,201 |

QPS changes are small and are not evidence of a stable speedup. At ef=64, 99% overlap loses 0.0781 percentage points of Recall. The raw `results/search_*.json` files include ef=64/128/256/512, per-query hit/distance counts, four batch timings and ranges.

Strict final-graph reuse preserves all adjacency rows exactly, but is slower at 90%/95% overlap; at 99%, it skips pruning for 440,178 nodes and reduces core time from 25.102 s to 21.019 s. See `results/exact_reuse.json`.

## Limits and validation scope

- Construction timings cover memory-resident core work, not complete HGraph import/build, content validation, routing-layer construction or cache file round trips. The old graph is reconstructed from the initial leaf cache in this harness; a product would export it during the original build.
- Search QPS measures a custom single-caller bottom-graph best-first probe with 8 fixed entries, `std::vector` adjacency and standard heaps. Truth, validation, build and graph export are excluded from timing. It is not HGraph API QPS.
- Search correctness checks are independent of timing. AB/BA/BA/AB version order and shuffled query order reduce drift; no affinity or statistical repeatability claim is made.
- The old leaf cache is about 880 MB. Extra fixed old-reservoir key slots are about 512 MB/1M, plus adjacency/source-distance data. The warm algorithm only needs leaf members, final adjacency and reservoirs, but the shared harness still loads the full old leaf snapshot.
- Frozen leaves can exceed max_leaf_size after updates. Local splitting, multi-round aging, ID remapping, quantizer retraining, adaptive pruning, duplicate storage and Pyramid integration remain unimplemented.
- Approximate warm start has no full-graph connectivity guarantee and no extra graph-search refinement. At fixed ef, unchanged aggregate Recall does not imply identical per-query results or equivalent quality everywhere.
- Packaged sources receive format/build/small-run validation. Full library/sanitizer/Python suites and a new million-vector run of the packaged source are not claimed.

## Packaging validation

The packaged five executable variants compiled and linked on macOS against the cached VSAG/dependency archive. The optional CMake configuration generated successfully. Generator tests (4), deletion/tie/K tests (228), certificate mutations (9,000), and a fresh 10k derived/warm smoke run passed. clang-format 20 and whitespace checks passed. This does not claim a fresh full-library rebuild or full test suites.

A clang-tidy 20 scan of the packaged warm driver is **not clean**: the research harness still has naming, implicit-conversion, missing-brace and implementation-file inclusion diagnostics. No checks were disabled to hide them. The standalone include/generation arrangement and these diagnostics need cleanup before a production PR. Recorded million-vector results are retained as historical measurements, not a claim that this branch is merge-ready.
