# Final-version acceptance audit, 2026-10-14

## Verdict against the actual four outputs

| Output | Evidence / verdict | Remaining gate |
| --- | --- | --- |
| 1. Reviewable Lite PR, standalone build, CRUD/Search/SaveLoad, configuration, tests, guide | Implemented in #2904; fresh Release RaBitQ-ON 5/5 and OFF 4/4 in this audit | Community Actions execution/approval and upstream relevant tests are not certified by local tests |
| 2. Architecture, retained/cut modules, index/layout/memory/API scope | Delivered in bilingual `lite_first.md` / `lite_rabitq_design.md` | Maintainer review; not ABI/format compatible with Full |
| 3. Reproducible >=2 datasets/scales and every requested metric | Historical native Full/package evidence plus this fresh SIFT/GIST10k resident CRUD matrix, raw records and verifier | Not a unified newly rerun Full/100k/Cohere/selected-ISA study; no hours-long endurance claim |
| 4. At least one clear package/load/memory gain vs corresponding actual Full with correctness/stable retrieval | Package and loaded-process RSS gains established for explicit profiles; CRUD/SaveLoad checks pass | GIST ANN quality, post-mutation degradation, same-quality comparison and review remain unresolved; no universal performance PASS |

The deliverables exist, but whole-project final acceptance is **not complete**.
The contract does not require every metric to beat Full, a .990 recall threshold,
one million vectors, or PR merge. Nevertheless, documented retrieval weaknesses
must not be hidden behind a smaller package or a passing functional test.

## Identity and source binding

Production feature SHA: `74a9420c771e75dea5e56324775d8db5d1472459`.
The evidence tree normally merges it into #2926; no graph repair candidate is
promoted. The build uses GCC11.4 Release/O3, optional RaBitQ ON, benchmark targets
ON, cached Catch2 v3.7.1 source and two build jobs. No old Lite objects are reused.
`source-closure.json` records Ninja's dependencies for all 15 production objects.
Every repository-owned compiled source/header matches frozen `dbb2fe0`; current
library SHA256 is `3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330`
(412512 bytes), identical to the historical measured Lite library. The new upstream
unused SIMD headers do not change that compiled closure. This binds the actual
Lite artifact; it does **not** rerun Full or package measurements or make the
current root Full implementation equivalent to the historical Full source.

## Fresh resident CRUD protocol and data

`summary.json` gives medians of three per-process summaries for six dataset/storage
cells; `raw.tar.gz` contains complete CSVs, commands, input hashes, manifests,
compiler flags, process resources and per-run audits. `archive.json` binds the
archive SHA256/size. Unpack into a fresh directory and run:

```bash
python3 lite/benchmark/verify_mixed_cpu_acceptance.py /path/to/unpacked/raw
```

Two datasets: SIFT128 and GIST960, 10000 base / 100 supplied queries each; graph
degree16/build128/query512/top10, CPU0, one internal thread, BLAS/OpenMP limits1.
Three storages FP32/FP16/RaBitQ8; three fresh processes per cell, cyclic mode order;
20 rounds x 250 cycles, original index stays live. Each process has 5000 real
updates, 5000 removals, 5000 re-adds, 15000 interleaved queries, 20 saves/loads.
Total: 270000 mutation calls and 270000 mixed queries across 18 processes.
Independent raw verification checks sample completeness, quantiles, hashes,
deterministic quality/snapshots, live counts, one-thread samples and resident mode.
Current FP64 truth is checked initially/finally; mixed calls check live IDs,
uniqueness/order; persistence checks preserve results. PASS means this protocol,
not every possible concurrent/failure/duration scenario. Call-only throughput
excludes validation/oracle overhead; per-call getrusage perturbs timing and has
microsecond resolution. Save/load are same-process uncontrolled-cache operations.

Fresh query results (us; three-process median summaries, not pooled quantiles):

| Dataset/storage | P50 | P99 | Initial -> final recall@10 | Query CPU ms / 15000 calls |
| --- | ---: | ---: | --- | ---: |
| SIFT FP32 | 168.767 | 203.985 | .993 -> .997 | 2574.103 |
| SIFT FP16 | 163.817 | 193.596 | .993 -> .997 | 2485.940 |
| SIFT RaBitQ8 | 198.285 | 237.685 | .987 -> .994 | 3013.544 |
| GIST FP32 | 488.239 | 683.158 | .971 -> .970 | 7412.144 |
| GIST FP16 | 372.233 | 544.429 | .971 -> .969 | 5704.178 |
| GIST RaBitQ8 | 443.992 | 598.468 | .970 -> .963 | 6753.998 |

Call-only CRUD throughput (calls/s; same medians; each operation has 5000 samples
per process; full P50/P99, CPU user/system, Save/Load and Build are in summary.json):

| Dataset/storage | Add | Remove | Update |
| --- | ---: | ---: | ---: |
| SIFT FP32 | 14566.0 | 74682.5 | 13028.8 |
| SIFT FP16 | 15096.0 | 76916.0 | 13457.2 |
| SIFT RaBitQ8 | 9935.5 | 44167.3 | 8878.3 |
| GIST FP32 | 5593.7 | 45020.1 | 4310.9 |
| GIST FP16 | 6460.8 | 54093.9 | 5174.3 |
| GIST RaBitQ8 | 3451.7 | 24345.2 | 2500.9 |

## Historical actual Full comparison and negative results

The [fresh native Full report](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/full-fresh-equal-r1-20261012/README.md)
has all six cells: loaded process RSS lower in six, load faster in five; snapshots
larger and Save slower in six; GIST builder peaks higher in three. GIST recall is
lower than Full, and Lite RaBitQ remains slower than FP16. Equal ef is not equal
quality; Full Ofast versus Lite O3 and selected-ISA boundaries remain explicit.
The [runnable package profile](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/runtime-package-r1-20261012/README.md)
has payload 6689348 / 99246762 bytes (93.260% reduction), gzip 2664623 / 26932509
bytes (90.106% reduction). It uses historical native Full d18c82a, not the fresh
dbb2fe0 Full; it is not a complete SDK/wheel/container or licensed final release.

## Scope and release gates

External serialization is required; no internally concurrent read/write guarantee.
Single-layer graph is not HGraph; RaBitQ has fixed training and is optional/default
OFF. No mmap/zero-copy/atomic-save claim. Both PRs need Actions approval; #2926
also lacks a required kind label as observed before this push. Latest local passes
do not guarantee future community CI or maintainer acceptance. The next technical
priority is permanent-mutation GIST quality, followed by a frozen same-quality,
compiler/selected-ISA controlled production-vs-native-Full matrix and elapsed-time
100k/Cohere resident endurance. Rejected candidates stay experimental.
