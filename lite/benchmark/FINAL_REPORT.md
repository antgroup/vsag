# VSAG Lite experiment report

This report consolidates the reproducible evidence collected for the standalone VSAG Lite work. The public Lite implementation is reviewed in PR #2904. Benchmark programs, runners, and experimental evidence are isolated in PR #2926.

## Scope and identities

- Feature base: `25f1d595519a3e4953301021663621909578ebb7` (`feat/lite-first-v01`).
- Experiment head measured here: `fc744fd5e4c91270edaf2813f8e32f8aced1a8d1` (`experiment/lite-baseline-v01`), 21 experiment-only commits above the feature base.
- Host: Ubuntu x86-64 on the project AMD EPYC server; the full `uname` and `lscpu` output are retained with the raw artifacts.
- Build: Release, `ENABLE_TESTS=ON`, Catch2 v3.7.1 from the pinned local dependency source.
- Dataset: prepared 10k and 100k prefixes from ANN-Benchmarks SIFT-128, with 100 independent queries and exact Top-10 ground truth recomputed for each prefix.
- Graph parameters: degree 16 and `ef_search` 128.

The current-head run uses public `Index` APIs for FP32 and FP16 graph storage. Each build probe verifies Save/Load result and distance equality. Each load value is the median of seven fresh processes, with FP32/FP16 order alternated. The operating-system page cache was not evicted.

## Current-head graph results

Build, search, Save, and the same-process Load columns below are one serial observation per configuration. Peak RSS covers the complete build-probe process.

| Prefix | Storage | Recall@10 | Build (ms) | Search P50/P99 (us) | Save (ms) | Same-process Load (ms) | Snapshot (bytes) | Build-process peak RSS (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | FP32 v2 | 0.973 | 916.301 | 113.107 / 163.257 | 27.737 | 2.479 | 6,560,064 | 28,476 |
| 10k | FP16 v3 | 0.973 | 866.179 | 109.268 / 149.666 | 24.030 | 2.490 | 4,000,064 | 21,844 |
| 100k | FP32 v2 | 0.946 | 25,339.1 | 343.773 / 456.691 | 284.252 | 32.394 | 65,600,064 | 211,956 |
| 100k | FP16 v3 | 0.946 | 21,954.3 | 324.073 / 435.341 | 240.532 | 23.833 | 40,000,064 | 171,468 |

FP16 reduces the snapshot size by 39.0% at both scales in this format comparison. Recall is unchanged on these integer-valued SIFT subsets. The latency columns are not a variance study and should not be generalized from one graph build.

### Fresh-process Load

| Prefix | Storage | Load median (ms) | First query median (us) | Follow-up P50/P99 median (us) | Steady RSS median (KiB) | Process peak RSS median (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | FP32 v2 | 3.643 | 222.395 | 141.288 / 233.766 | 11,708 | 11,748 |
| 10k | FP16 v3 | 2.740 | 178.755 | 116.957 / 175.186 | 9,248 | 9,288 |
| 100k | FP32 v2 | 35.966 | 493.259 | 357.123 / 491.379 | 75,772 | 75,872 |
| 100k | FP16 v3 | 27.200 | 443.289 | 321.674 / 433.800 | 50,724 | 50,824 |

All 28 loader processes passed the existing assertions. These values measure process reconstruction with an uncontrolled page cache; they are fresh-process results, not strict cold-start I/O.

### Isolated resident memory

The isolated RSS probe releases the source base vectors, calls `malloc_trim(0)` on glibc, executes all 100 queries, and reads current `VmRSS`. Queries, ground truth, process image, containers, IDs, and graph links remain resident.

| Prefix | Storage | Recall@10 | Current RSS (KiB) |
| --- | --- | ---: | ---: |
| 10k | FP32 | 0.973 | 12,036 |
| 10k | FP16 | 0.973 | 9,604 |
| 100k | FP32 | 0.946 | 77,352 |
| 100k | FP16 | 0.946 | 52,368 |

FP16 reduced current RSS by 20.2% at 10k and 32.3% at 100k in these single fresh-process observations.

### Current library artifact

The exact Release shared library used by the current-head probes was 145,688 bytes before stripping and 125,776 bytes after GNU `strip`. This is a standalone Lite artifact and does not include Full VSAG. Toolchain and enabled-feature differences make it inappropriate to compare this value directly with unrelated release packages.

## Historical controlled comparisons

The following results were measured at the named earlier commits and are retained as historical evidence. They were not relabeled as measurements of `fc744fd5`.

### Full versus Lite exact BruteForce

At experiment commit `d729426`, the generated 100k x 128 workload reported a median Search P50 of 2,401 us for Full and 4,392 us for Lite. Process-lifetime peak RSS was 214,200 KiB for Full and 72,912 KiB for Lite. The stripped shared libraries were 40,463,344 and 39,488 bytes respectively. Lite Search, Save, and same-process warm Load were slower at this point; the result supports the deployment-size and memory goal, not a universal speedup claim.

The subsequent runtime SIMD comparison measured the Lite scalar and dispatched builds in seven alternating runs. Search P50 changed from 436.0 to 100.1 us at 10k and from 4,344.0 to 3,331.8 us at 100k. All 640 inspected Top-10 rows retained the same IDs and order; the maximum absolute distance difference was `9.54e-6`. The stripped Lite library increased from 39,488 to 47,704 bytes.

### Update, Remove, and graph churn

At commit `66f2a18`, the graph stability runner kept 2,000 live 32-dimensional vectors through ten rounds of 100 Update-Remove-Add cycles. Save/Load checks passed in every round. Snapshot size remained between 442,544 and 472,456 bytes, current RSS changed from 4,300 to 5,044 KiB, and process peak RSS ended at 5,044 KiB. Top-1 self-query recall ranged from 0.875 to 1.000. This is bounded-workload churn evidence, not a proof that all allocators or workloads are fragmentation-free.

### Snapshot-load optimizations

The bulk-read change at `fb36355` reduced median fresh-process Load for the 52,000,048-byte v1 BruteForce snapshot from 192.956 to 22.436 ms and for the 65,600,064-byte v2 graph snapshot from 250.835 to 34.652 ms, with result checks unchanged.

The direct FP16 restore change at `5ac36ef` removed the temporary FP32 conversion. On the same v3 snapshots, median Load changed from 27.089 to 2.580 ms at 10k and from 270.248 to 26.168 ms at 100k. Median peak RSS changed from 14,012 to 9,240 KiB and from 100,600 to 50,848 KiB. Snapshot hashes and Recall@10 remained unchanged.

### Offline quantization candidates

The 100k SIFT exhaustive-scan probe measured FP32 Recall@10 1.000, SQ8 0.988, and FP16 1.000. Encoded vector bytes were 51,200,000 for FP32, 12,800,000 plus a 1,024-byte SQ8 model, and 25,600,000 for FP16. The SQ8 and FP16 probe timings use generic scalar exhaustive scans and are not graph-search performance. FP16 is now a public graph storage choice; SQ8 remains an offline candidate. RabitQ is not implemented in the public Lite backend.

## Reproduction and evidence

Build the opt-in experiment targets as documented in [README.md](README.md). The current graph probes can be repeated with:

```bash
VSAG_SIFT_DIR=/path/to/scale-100000 \
VSAG_GRAPH_STORAGE=fp16 \
VSAG_GRAPH_SNAPSHOT=/new/path/graph.snapshot \
build-lite/lite_graph_tests '[lite-sift]'

VSAG_SIFT_DIR=/path/to/scale-100000 \
VSAG_GRAPH_SNAPSHOT=/new/path/graph.snapshot \
build-lite/lite_graph_tests '[lite-sift-load]'

VSAG_SIFT_DIR=/path/to/scale-100000 \
VSAG_SIFT_RSS_BACKEND=fp16 \
build-lite/lite_graph_tests '[lite-sift-rss]'
```

Raw stdout, stderr, `/usr/bin/time -v`, host metadata, four snapshots, snapshot hashes, and exact library copies for the current-head rerun are retained outside Git at:

```text
/home/ubuntu/project/vsag-lite-final-report-20260920-fc744fd
```

The snapshot SHA-256 values are:

| Snapshot | SHA-256 |
| --- | --- |
| 10k FP32 v2 | `dcb3cc656e5663ea8c85ec59b7febb2ec4d950a50951d8ef12e2b32e8792b034` |
| 10k FP16 v3 | `2893269024e05d3a4c9fc5870e0724c681214268fb2ff73a1a80488a3d205932` |
| 100k FP32 v2 | `790619140f7c64556218560524cdeb67386c362d3d69caebc6cf453f1cf8aa20` |
| 100k FP16 v3 | `ee8390305db804d69999640e73c6a3d5bbf53d21507ecf7e6a5dea35ac740e3f` |

## Limitations

- The current graph rerun uses one host and one graph build per configuration. Only fresh-process Load has seven repetitions.
- Page-cache state is uncontrolled; no result is a strict cold-start measurement.
- The loader owns reconstructed memory. mmap and zero-copy loading are not implemented or measured.
- SIFT vectors in these subsets are integer-valued, so unchanged FP16 recall does not establish quality for arbitrary floating-point embeddings.
- BuildGraph temporarily retains input and output representations; its process peak is higher than steady-state RSS.
- The graph is the standalone Lite single-layer implementation. It is not Full `HGraph` or `LazyHGraph`.
- Filtered search, Update, Remove, Save, and Load are covered by tests, but concurrent calls are outside the current API guarantee.
- The Full/Lite BruteForce, SIMD, churn, and before/after load comparisons belong to their recorded commits and must not be presented as current-head reruns.

## Full HGraph RaBitQ reference

Experiment commit `844790629e57bcd89809dbba58b260f8a74d74e1`
adds an opt-in public-API Full HGraph consumer. It uses batch `Build`, because
RaBitQ requires training, and compares FP32, the documented one-bit RaBitQ
configuration with FP32 reorder, and a three-filter-bit plus
five-supplement-bit split configuration. Degree is 16 and `ef_search` is 128
for all modes. Seven fresh processes per scale and mode were run in alternating
order; all 42 serialize/deserialize result-identity checks passed.

| SIFT-128 subset | Mode | Recall@10 median [range] | Build (ms) | P50/P99 median (us) | Snapshot bytes | Final/peak RSS (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | FP32 | 0.997 [0.971, 1.000] | 258.890 | 90.148 / 141.856 | 5,765,755 | 169,320 / 177,052 |
| 10k | RaBitQ 1-bit + FP32 reorder | 0.987 [0.921, 0.990] | 304.969 | 93.278 / 124.655 | 6,013,051 | 300,676 / 308,660 |
| 10k | RaBitQ 3+5 split | 0.983 [0.919, 0.995] | 328.196 | 121.757 / 153.896 | 2,202,806 | 301,480 / 308,896 |
| 100k | FP32 | 0.997 [0.996, 0.998] | 3,227.986 | 190.635 / 265.864 | 57,157,546 | 190,456 / 315,288 |
| 100k | RaBitQ 1-bit + FP32 reorder | 0.915 [0.901, 0.928] | 3,389.983 | 181.305 / 238.453 | 59,567,366 | 321,404 / 446,720 |
| 100k | RaBitQ 3+5 split | 0.986 [0.984, 0.989] | 3,605.873 | 221.003 / 326.071 | 22,235,021 | 322,544 / 461,840 |

At 100k, the 3+5 split reduces snapshot bytes by 61.1% relative to FP32
HGraph, while median Recall@10 changes by -0.011. Median P50 is 15.9% slower,
build is 11.7% slower, and load is 20.3% slower. Its P99 range was
290.072--2,891.121 us because one run was an outlier. The one-bit mode does
not satisfy the Lite storage goal in this form: its FP32 reorder payload makes
the snapshot 4.2% larger than FP32 and it consumes substantially more process
memory.

This is a Full HGraph reference, not a Lite graph result. Full RaBitQ retains
PCA/FHT models, allocator and Full library state, so the RSS values cannot be
projected onto a future standalone Lite implementation. The evidence supports
a bounded Lite-specific 3+5 layout and dependency audit; it does not support
copying the Full quantizer or adding RaBitQ to the public Lite feature PR yet.
Raw artifacts are retained at
`/home/ubuntu/project/vsag-lite-rabitq-reference-20260921-8447906`.
