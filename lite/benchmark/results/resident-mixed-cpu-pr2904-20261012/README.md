# Resident-index real-change mixed CRUD acceptance, 2026-10-12

This PR2926 evidence measures unchanged PR2904 feature
`dbb2fe0bec685ae7a343b1d2060d0553a98e20d2`, shared-library SHA256
`3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330`.
No production changes or new Full comparison are claimed.

## Protocol and what changed

The preceding [three-round matrix](../mixed-cpu-pr2904-20261012/README.md)
replaced the index with its loaded copy each round, which could reset capacity.
Here `--resident` keeps the **original index object for all20 rounds**. A separately
loaded copy checks persistence and is destroyed; it never replaces the resident index.
Measurement records are pre-sized/touched then cleared before initial RSS sampling,
so growing measurement-vector capacity does not masquerade as graph memory growth.

- SIFT128/GIST96010k, existing100query cohort, FP32/FP16/RaBitQ8.
- Degree16/build128/query512/k10; CPU0, OMP/OpenBLAS/MKL1; observed `/proc` thread
  counts are1 at all378 checkpoints, not a claim about transient peak thread count.
- Three rotated fresh processes per cell,18 total. Each20rounds×250cycles:
  whole-vector Update→query→Remove→query while absent→Add changed vector→query.
- Each process15,000 mutation API calls and15,000 mixed queries. **270,000 each
  cumulatively**; not one270k-call instance and not simultaneous read/write calls.
- Current-vector oracle, live-ID/size/finite/unique/order checks; exactFP64initial/final
  truth for all100queries. Every round Save/Load ordered IDs/distances match.
- All18processes complete; all repeated quality rows and20snapshot hashes/cell
  match. Independent raw quantile/count/resource/state audit and packed replay PASS.
- Benchmark format15 uses the explicitly selected official feature `.clang-format`;
  focusedtidy15 passes. Seven Python tests include three-storage resident/reload
  snapshot parity, invalid lifecycle rejection and unsafe archive path/link rejection.
  No new product sanitizer/coverage or hostedCI success is claimed.

Each whole-vector replacement is `.875*current + .125*original donor` and must
differ from its previous FP32 input. Quantized storage may map some close inputs to
the same representation; this is not misreported as guaranteed representation change.

## CRUD and query performance

Medians of three per-process nearest-rank quantiles; each process5,000samples per
mutation kind and15,000mixed-query executions. Only100distinct queries are reused.
The20round schedule differs from the earlier three-round schedule; changes in
timings must not be interpreted as a new code optimization or a causal speedup.

| Dataset/storage | Add P50/P99 us | Remove P50/P99 us | Update P50/P99 us | Mixed query P50/P99 us | Recall initial/final |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT FP32 | 67.468 / 82.807 | 11.630 / 16.560 | 72.708 / 99.737 | 165.806 / 196.265 | .993 / .997 |
| SIFT FP16 | 65.538 / 80.047 | 11.500 / 16.270 | 70.688 / 97.727 | 162.385 / 189.274 | .993 / .997 |
| SIFT RaBitQ8 | 100.058 / 116.257 | 21.209 / 32.089 | 105.416 / 172.156 | 194.874 / 225.774 | .987 / .994 |
| GIST FP32 | 149.544 / 203.404 | 15.499 / 40.670 | 186.734 / 383.770 | 395.848 / 570.184 | .971 / .970 |
| GIST FP16 | 146.737 / 178.544 | 12.700 / 31.650 | 163.685 / 325.531 | 332.970 / 420.908 | .971 / .969 |
| GIST RaBitQ8 | 283.532 / 319.852 | 33.179 / 98.056 | 321.131 / 1160.877 | 422.128 / 491.746 | .970 / .963 |

| Dataset/storage | Add calls/s | Remove calls/s | Update calls/s | Query calls/s |
| --- | ---: | ---: | ---: | ---: |
| SIFT FP32 | 14,946 | 83,154 | 13,702 | 5,968 |
| SIFT FP16 | 15,359 | 84,055 | 14,062 | 6,110 |
| SIFT RaBitQ8 | 10,091 | 47,226 | 9,234 | 5,088 |
| GIST FP32 | 6,662 | 57,412 | 5,108 | 2,494 |
| GIST FP16 | 6,861 | 70,343 | 5,849 | 3,015 |
| GIST RaBitQ8 | 3,532 | 26,007 | 2,602 | 2,387 |

Call-only throughput excludes checks, truth, Save/Load and bookkeeping; this is
not end-to-end service throughput. Per-call `getrusage` CPU instrumentation affects
timings and has microsecond resolution. Original sample CSV retains user/system
CPU and context-switch deltas for every operation. CPU intervals include counter
overhead outside the wall interval, so small CPU/wall differences are expected.

## CPU and persistence resources

Whole-process medians below include exact oracle computation, build, extra
persistence-check queries and all bookkeeping, not just indexed API execution.
GNUtime average CPU percentages are99–100% of one CPU, not peak utilization.
Individual processes last about5–20seconds: **this is not hours-long endurance**.

| Dataset/storage | Process user/system seconds | Process wall seconds | Save/Load P50 ms | Final snapshot bytes | Whole-process peak KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT FP32 | 5.21 / .07 | 5.29 | 3.661 / 2.645 | 6,560,064 | 40,400 |
| SIFT FP16 | 5.10 / .05 | 5.16 | 3.097 / 2.278 | 4,000,064 | 32,904 |
| SIFT RaBitQ8 | 6.26 / .04 | 6.30 | 4.033 / 3.246 | 2,960,672 | 30,668 |
| GIST FP32 | 17.83 / .55 | 18.35 | 15.910 / 17.129 | 39,840,056 | 166,320 |
| GIST FP16 | 15.93 / .21 | 16.12 | 9.605 / 6.793 | 20,640,056 | 146,876 |
| GIST RaBitQ8 | 19.41 / .13 | 19.54 | 6.606 / 4.842 | 11,284,408 | 144,552 |

Save/Load are same-process uncontrolled-cache measurements and lack fsync;
neither strict cold startup nor durable-storage performance is established.
Peak includes base+original FP32 buffers, query/oracle workspace, pre-touched
measurement buffer, build and simultaneous resident/loaded indexes.

## Resident process-memory trajectory

The following medians are sampled **after mutations, before that round's Save/Load**.
Prior rounds' check copies can leave allocator pages resident, so even this is not
isolated index-owned capacity. The public API exposes no capacity counter; no
claim about exact fragmentation bytes or allocator page return is made.

| Dataset/storage | Initial KiB | Round5 KiB | Round10 KiB | Round20 KiB | Rounds10–20 range per repeat KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT FP32 | 27,408 | 40,400 | 40,400 | 40,400 | 0 / 0 / 0 |
| SIFT FP16 | 24,872 | 32,904 | 32,904 | 32,904 | 0 / 0 / 0 |
| SIFT RaBitQ8 | 24,784 | 30,140 | 30,404 | 30,668 | 264 / 264 / 264 |
| GIST FP32 | 125,188 | 128,636 | 128,788 | 128,836 | 48 / 48 / 48 |
| GIST FP16 | 106,408 | 146,876 | 146,876 | 146,876 | 0 / 0 / 0 |
| GIST RaBitQ8 | 98,224 | 111,484 | 111,748 | 112,012 | 264 / 264 / 264 |

FP32 SIFT and FP16 show flat late windows, but RaBitQ still has a264KiB late step
in every repeat. This supports bounded observations in the recorded window,
**not proof of zero leaks, zero fragmentation, universal plateau or indefinite stability**.
Checkpoint-load allocator retention can also explain why GIST FP16 process RSS
exceeds FP32 despite smaller stored vectors; the experiment does not identify its
exact cause and must not be used as an index-owned-memory ranking.

## Conclusions and remaining gates

Functional resident CRUD and persistence checks pass on both datasets. FP16 remains
the best latency choice in this workload. RaBitQ retains smaller snapshot/whole
process peak but slower CRUD and GIST Update tail; GIST recall .970→.963 loses7/1000
hits against each state's correct truth. Because these100queries are already
observed, this is not an independent quality replication or a universal stability claim.

Remaining: actual nativeFull corresponding-config and same-quality/SIMD comparison;
complete deployment-package size; longer elapsed-time resident runs without checkpoint
allocation interference; larger independent query cohorts and other read/write ratios;
100k/Cohere and GIST100k route/maintenance quality. Project acceptance is not complete.

## Raw archive and replay

[raw.tar.gz](raw.tar.gz) is4,390,665bytes,152regular members. [archive.json](archive.json)
records its SHA256 and successful independent verification after safe unpacking.
Original raw files remain on the server; only this run's RAM snapshots/binary were
automatically removed after hash receipts. No old sources, data or results were deleted.
The root identity/rows/manifest copies are browsing aids; complete CSV/resource/config/
measured-source evidence is in the archive, with every measured member hashed.

```bash
mkdir unpacked
tar -xzf raw.tar.gz -C unpacked
python3 lite/benchmark/verify_mixed_cpu_acceptance.py unpacked
python3 lite/benchmark/run_mixed_cpu_acceptance.py \
  --feature-repo /path/to/pinned-feature --feature-sha dbb2fe0bec685ae7a343b1d2060d0553a98e20d2 \
  --build /path/to/feature-release-build \
  --dataset sift10k=/path/to/sift/scale-10000 --dataset gist10k=/path/to/gist/scale-10000 \
  --output /path/to/new-resident-results --trials 3 --rounds 20 --cycles 250 --cpu 0 --resident
```

The source on this branch compiles against the explicit pinned feature checkout,
not against the experiment branch's historical feature API. Linux/C++17/Python,
GNUtime/taskset and writable `/dev/shm` are required. Input hashes must match identity.
The old three-round raw source remains untouched; it used the sparse checkout's
default formatter style, whereas this benchmark explicitly uses official format15.
