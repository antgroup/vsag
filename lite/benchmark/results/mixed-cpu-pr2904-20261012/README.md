# Pinned PR2904 real-change mixed CRUD and CPU acceptance, 2026-10-12

This evidence belongs to PR #2926 (`experiment/lite-baseline-v01`). It measures
the feature implementation from PR #2904, **not the older implementation embedded
in the experiment branch**. No production files or experimental policies were promoted.

Measured feature SHA: `dbb2fe0bec685ae7a343b1d2060d0553a98e20d2`.
Measured Release shared-library SHA256:
`3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330`.
Library file size: 412,512 bytes, **not a complete deployment package**.
`identity.json`, archived feature build cache/commands and benchmark `source.cpp`
bind the code, compiler, configuration, machine, library and input hashes.

## Protocol and verification

- SIFT128 and GIST960, each 10,000 base records and the existing 100-query cohort.
- FP32, FP16 and opt-in RaBitQ8; degree16/build128/query512/k10; CPU0;
  OMP/OpenBLAS/MKL thread settings1. These settings do not certify actual thread count.
- Three independent processes per dataset/storage, rotated storage order: 18 total.
- Each process: three rounds, 100 changed-vector cycles per round. Every cycle
  updates a complete vector using `0.875*current + 0.125*original donor`, then
  queries, removes that ID, queries while it is absent, re-adds its changed vector
  and queries again. No restoration to the original vector before mixed queries.
- Per process: 900 mutation API calls and 900 interleaved queries. Across all
  processes: **16,200 mutation calls and 16,200 queries**, not one 16,200-call instance.
- Calls are externally serialized; this is not simultaneous reader/writer support.
- All mixed results check live IDs, finite distances, uniqueness, count and ordering.
  Initial/final recall uses exact double-precision L2 recomputed from the current
  FP32 oracle vectors, with deterministic distance/ID ties. This is not recall
  computed against the old ground truth after mutation.
- Each round saves and loads, compares all 100 queries' ordered IDs and distances
  exactly, then continues mutating the loaded index. Snapshot hashes are repeatable.
- Independently recomputed raw summaries and manifest verification: PASS; repeat
  quality rows and all per-round snapshot hashes are identical for each cell.
- Fresh feature Release CTest5/5, benchmark format15/focused tidy15 and Python
  regression tests3/3 passed. No new sanitizer or coverage claim is made for this benchmark.

## Latency and throughput

Numbers are medians of the three per-process values. Each process contributes
300 samples per mutation kind and 900 mixed-query samples. P99 is the nearest-rank
percentile of executions, **not 900 distinct queries**. Timing includes per-call
CPU instrumentation; do not combine these rows with uninstrumented query-only results.

| Dataset/storage | Add P50/P99 us | Remove P50/P99 us | Update P50/P99 us | Mixed query P50/P99 us | Recall initial/final |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT FP32 | 68.268 / 85.217 | 13.409 / 22.280 | 76.508 / 117.276 | 171.174 / 215.964 | .993 / .993 |
| SIFT FP16 | 66.690 / 81.167 | 13.290 / 21.899 | 74.518 / 108.926 | 166.674 / 205.085 | .993 / .993 |
| SIFT RaBitQ8 | 102.286 / 124.716 | 24.470 / 35.269 | 110.177 / 226.054 | 200.715 / 264.302 | .987 / .985 |
| GIST FP32 | 171.934 / 240.134 | 17.968 / 48.180 | 219.744 / 439.147 | 474.517 / 656.222 | .971 / .976 |
| GIST FP16 | 161.956 / 208.234 | 16.450 / 46.668 | 190.854 / 420.787 | 377.230 / 558.845 | .971 / .975 |
| GIST RaBitQ8 | 294.672 / 340.671 | 41.710 / 102.977 | 342.781 / 1957.086 | 448.198 / 627.212 | .970 / .970 |

| Dataset/storage | Add calls/s | Remove calls/s | Update calls/s | Query calls/s | Mixed API CPU sum ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT FP32 | 14,713 | 74,012 | 12,909 | 5,754 | 204.715 |
| SIFT FP16 | 15,051 | 75,253 | 13,233 | 5,926 | 198.940 |
| SIFT RaBitQ8 | 9,790 | 41,121 | 7,576 | 4,893 | 261.998 |
| GIST FP32 | 5,749 | 51,039 | 4,332 | 2,096 | 557.441 |
| GIST FP16 | 6,177 | 55,853 | 4,956 | 2,613 | 459.480 |
| GIST RaBitQ8 | 3,378 | 21,235 | 2,251 | 2,211 | 643.826 |

Calls/s = calls / sum of measured API wall time. It excludes validation, truth,
persistence and bookkeeping and is **not end-to-end mixed-service throughput**.
Mixed CPU = user+system deltas summed for Add/Remove/Update/Query, not whole-process
CPU. CPU counters have microsecond resolution and include clock/getrusage overhead
outside the wall-time interval; CPU sums may slightly exceed measured call wall time.
Whole-process GNU time resources retain user/system time, average CPU percentage,
page faults and context switches, including oracle computation and build. That CPU
percentage must not be represented as index-only utilization or peak utilization.

## Persistence and memory

Save/load times below are medians of each process's three per-round measurements,
then medians across processes. Load is same-process after Save with uncontrolled
page cache, not fresh-process or cold startup. Native Save/Load preserves results.

| Dataset/storage | Save ms | Load ms | Final snapshot bytes | Initial/final process RSS KiB | Process peak KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT FP32 | 3.802 | 4.064 | 6,560,064 | 24,608 / 33,008 | 33,008 |
| SIFT FP16 | 3.152 | 2.648 | 4,000,064 | 22,088 / 28,168 | 28,168 |
| SIFT RaBitQ8 | 3.049 | 3.788 | 2,960,672 | 21,996 / 26,808 | 26,988 |
| GIST FP32 | 15.750 | 17.915 | 39,840,056 | 122,332 / 126,048 | 163,532 |
| GIST FP16 | 9.439 | 6.867 | 20,640,056 | 103,516 / 125,804 | 141,596 |
| GIST RaBitQ8 | 5.861 | 5.594 | 11,284,408 | 95,388 / 108,380 | 141,752 |

RSS/peak include base, original/oracle copy, queries, graph build, exact-truth
workspace and simultaneous old/loaded indexes. They are not index-owned bytes or
isolated mutation-stage peaks. Reloading every round replaces the object and can
reset capacity; this protocol **does not prove continuously resident index capacity
plateaus, absence of fragmentation, or hours-long stability**.
Temporary snapshots were hashed before automatically removing this run's own RAM
scratch. Hash/size receipts remain; snapshots can be reconstructed by replay. No old
source, datasets or experimental results were deleted.

## Conclusions and acceptance limits

Real-change short mixed CRUD, deletion-aware queries and persistence passed in
all six cells. FP16 is the strongest latency/CPU choice in this measured workload.
RaBitQ gives smaller snapshots but slower CRUD, notably GIST Update P99; SIFT
recall loses two of 1,000 hits after these changes. This is not an unqualified
RaBitQ speed advantage or proof of stable quality for all workloads.

This closes part of requirements3/4: current-feature-version two-dataset CRUD,
CPU, latency, snapshot, load and process-memory evidence. Remaining gates:

1. Fresh native Full corresponding-config and same-quality/SIMD comparison, with
   at least one demonstrated package/load/memory advantage; this report is not Full.
2. Complete deployment package measurement; `.so` bytes are not that metric.
3. Sustained resident-index endurance without reload resetting capacity, larger
   independent query cohort, other workload read/write ratios, and 100k/Cohere.
4. GIST100k routing/maintenance quality, including the already-known lower recall.
5. Final architecture/API boundaries and hosted PR checks pinned to delivery SHA.

## Reproduce

Checkout the #2904 feature SHA separately and build its standalone `lite/` Release
with `ENABLE_RABITQ_LITE_BACKEND=ON`. Do not build the old feature code on #2926
and label that result latest. Prepare the existing two 10k fvecs datasets using the
dataset tools on this branch and verify hashes against `identity.json`.

```bash
python3 lite/benchmark/run_mixed_cpu_acceptance.py \
  --feature-repo /path/to/pinned-feature-checkout \
  --feature-sha dbb2fe0bec685ae7a343b1d2060d0553a98e20d2 \
  --build /path/to/feature-release-build \
  --dataset sift10k=/path/to/sift/scale-10000 \
  --dataset gist10k=/path/to/gist/scale-10000 \
  --output /path/to/new-results --trials 3 --rounds 3 --cycles 100 --cpu 0
python3 lite/benchmark/verify_mixed_cpu_acceptance.py /path/to/new-results
python3 -m unittest discover -s lite/benchmark -p test_mixed_cpu_acceptance.py -v
```

Requires Linux, C++17, `/usr/bin/time`, `taskset`, Python standard library and writable
`/dev/shm`. Results/inputs must not be overwritten; source and measured library must
remain unchanged during the run. `manifest.json` binds measured raw files; this
explanatory README was written after measurement and is not part of that manifest.
