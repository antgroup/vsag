# Current Lite / native Full equal-parameter acceptance, 2026-10-12

Measured Lite feature `dbb2fe0bec685ae7a343b1d2060d0553a98e20d2`, library
SHA256 `3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330`.
Actual native Full HGraph reference source `d18c82a1f1f23ff84362516e86af5f3cb2e34475`,
installed-library SHA256 `9aeaa550703f116bef2959769d4f2f03da254b07c906ba941d8186224cc70c72`.
The reference is a known installed incremental shared rebuild, **not a fresh Full
all-object build or latest upstream**. Header hashes were rechecked. Four measurement
consumers were freshly compiled against their respective headers/libraries.

## Parameters and boundaries

SIFT128/GIST96010k, existing100query cohort and corresponding-prefix Top10truth.
Both sides use degree16/construction128/**query512**, CPU0, OMP/BLAS/MKL1.
Full small profile: memory_io, compressed graph, build_thread_count1, no redundant
raw FP32/FP16 copy; RaBitQ3+5/FHT/float queries/reorder/error_rate1.9.
Lite uses its public default graph policies and opt-in RaBitQ8. Three alternating
fresh-process pairs per cell, three fresh loaders per builder: **36builders108loaders**.
No query warmup; uncontrolled page cache; rounds0, **not a mutation performance test**.

Unlike the previous512-versus128 report, query ef is numerically equal. It does
not imply equal node visits, identical topology, rotations, quantizer or recall.
The host has AVX2/AVX512; selected instruction paths were not traced or proved
identical. Do not label this strict same-quality or exact-SIMD-path alignment.
Groundtruth/quantiles are independently recomputed from raw neighbor/latency rows;
input hashes are checked, but exhaustive truth preparation is not rerun by the audit.

## Initial-index query and load results

All values are medians, **Full / Lite**. Each query process measures100executions
of100observedqueries; P99 remains exploratory. Full RaBitQ training is randomized;
SIFT recall spans .978–.990, GIST .987–.991 across the three Full repeats. Do not
select the most favorable repeat or claim a deterministic Full snapshot.

| Dataset/storage | Recall@10 | P50 us | P99 us | Fresh-process Load ms | Loaded total RSS KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT FP32 | .987 / .993 | 270.223 / 165.206 | 417.188 / 189.224 | 5.046 / 3.648 | 32,840 / 12,556 |
| SIFT FP16 | .987 / .993 | 272.342 / 163.055 | 416.108 / 188.276 | 3.412 / 2.931 | 30,204 / 10,068 |
| SIFT RaBitQ8 | .980 / .987 | 331.491 / 199.384 | 483.827 / 334.481 | 2.599 / 3.911 | 29,776 / 8,488 |
| GIST FP32 | .992 / .971 | 511.176 / 420.079 | 673.372 / 590.812 | 27.190 / 15.922 | 66,056 / 45,088 |
| GIST FP16 | .992 / .971 | 423.818 / 351.660 | 508.237 / 437.238 | 14.477 / 9.566 | 46,904 / 26,344 |
| GIST RaBitQ8 | .989 / .970 | 516.176 / 437.448 | 619.013 / 687.071 | 8.815 / 6.639 | 38,088 / 16,636 |

Lite P50 is lower in all6cells at these parameters, but GIST recall is lower by
1.9–2.1percentage points. **This is not a same-quality speed victory.** GIST RaBitQ
P99 is worse. Fresh-process load is faster in5/6cells; SIFT RaBitQ remains slower.
It is fresh process, not strict cold cache.

Total loaded RSS is lower in all6cells. Much of this is the runtime baseline:
Full before-load RSS is about25MiB, Lite about3.6MiB. Paired loaded-minus-before-load
RSS medians below are **larger for Lite in all6cells**, preserving an important
negative result rather than attributing total-process wins solely to data layout.

| Dataset/storage | Incremental loaded RSS KiB Full/Lite |
| --- | ---: |
| SIFT FP32 | 7,764 / 8,932 |
| SIFT FP16 | 5,148 / 6,432 |
| SIFT RaBitQ8 | 4,164 / 4,880 |
| GIST FP32 | 41,036 / 41,488 |
| GIST FP16 | 21,772 / 22,708 |
| GIST RaBitQ8 | 12,536 / 13,028 |

## Build, persistence and peak resources

Peak covers the entire builder, including input matrices, queries, persistence
checks and old/loaded state; it is not index-owned bytes or isolated build peak.
Save does not establish fsync/durable-disk behavior.

| Dataset/storage | Build ms Full/Lite | Whole builder peak KiB Full/Lite | Save ms Full/Lite | Snapshot bytes Full/Lite |
| --- | ---: | ---: | ---: | ---: |
| SIFT FP32 | 787.641 / 538.863 | 42,292 / 27,672 | 3.070 / 3.768 | 5,766,017 / 6,560,064 |
| SIFT FP16 | 802.195 / 529.664 | 37,000 / 22,772 | 2.338 / 3.083 | 3,144,565 / 4,000,064 |
| SIFT RaBitQ8 | 917.780 / 601.463 | 38,188 / 21,564 | 1.906 / 2.524 | 2,203,554 / 2,960,672 |
| GIST FP32 | 1502.097 / 1297.444 | 108,932 / 125,412 | 13.593 / 16.000 | 39,808,753 / 39,840,064 |
| GIST FP16 | 1486.169 / 1304.301 | 85,468 / 103,804 | 8.012 / 9.500 | 20,148,057 / 20,640,064 |
| GIST RaBitQ8 | 1373.751 / 1598.127 | 102,192 / 103,816 | 4.456 / 4.654 | 10,691,074 / 11,284,416 |

All Lite Save/snapshot cells are worse. GIST builder peak is worse in all3storages,
while SIFT builder peak is better. GIST RaBitQ build is slower. FP16 Save is much
better than historical Lite before direct-half writing, but this does not turn it
into a Full advantage or reuse the old optimization percentages in this cohort.

## Binary-runtime footprint, not complete package

Identically stripped recursive shared-library DT_NEEDED closure:
Lite **6,147,776B**, Full **98,428,088B**, reduction **93.754%**.
`elf/closure.json` inside the raw archive records every dependency resolution,
original/stripped size and hash. Only scratch copies are stripped; source ELFs are
unchanged. This includes system runtime once per realpath, but excludes consumers,
headers, dlopen plugins, Python bindings, SDK/container packaging, data and snapshots.
**Do not present this as complete deployment-package size.** Full-library build
settings/version remain different; a fresh-library final baseline is still required.

## Evidence, verification and replay

[raw.tar.gz](raw.tar.gz):484,101B,701members, recorded hash in [archive.json](archive.json).
It contains source exports, compiler commands/hashes, provenance, input hash receipts,
all query/neighbor/parameter CSV/JSON, builder/load command/stdout/stderr/peak records,
raw summaries and ELF footprint accounting. Originals remain on the server. Large
snapshot bytes were not retained; each native builder verified Save/Load and snapshot
hashes remain. Only self-created RAM scratch was automatically removed.

Independent auditPASS:36builders,108freshloaders,3,600queryresults/36,000neighbors,
nearest-rank P50/P99, current truth intersections, loader counts and all manifest hashes.
Safe unpacking followed by the same independent audit alsoPASS. Three verifier guard
tests reject unequal ef, wrong reference-library receipts and corrupted manifest.
No Full test-suite, new production sanitizer/coverage or hostedCI success is claimed.

```bash
mkdir unpacked
tar -xzf raw.tar.gz -C unpacked
python3 lite/benchmark/verify_pinned_full_acceptance.py unpacked \
  --dataset sift10k=/path/to/sift/scale-10000 --dataset gist10k=/path/to/gist/scale-10000
python3 -m unittest discover -s lite/benchmark -p test_pinned_full_acceptance.py -v
python3 lite/benchmark/run_pinned_full_acceptance.py \
  --feature /path/to/dbb2fe0 --feature-build /path/to/feature-release \
  --source-repo /path/to/personal-vsag-source-at-a62e62e \
  --full-install /path/to/matching-known-native-full-install \
  --dataset sift10k=/path/to/sift/scale-10000 --dataset gist10k=/path/to/gist/scale-10000 \
  --output /path/to/new-results
```

Full replay requires the matching recorded installation; the wrapper intentionally
rejects a different library/header hash. Rebuilding Full elsewhere may produce a
different binary and requires a separately documented fresh baseline, not changing
this historical fixture's label. Archived source exports are pinned at public personal
sourcea62e62e; no private handoff or benchmark logic is hidden in this comparison.

Remaining acceptance: fresh Full all-object build/config/ISA provenance; same-quality
operating-point comparison; complete deployment package; larger independent queries,
100k/Cohere and sustained resident quality/CPU acceptance. The prior resident CRUD
matrix is complementary and must not be mislabeled as native Full CRUD measurement.
