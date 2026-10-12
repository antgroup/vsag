# FP32 reciprocal occlusion pilot, 2026-10-12

This is an isolated structural candidate, **not a promoted production optimization**.
Production PR #2904 at `dbb2fe0bec685ae7a343b1d2060d0553a98e20d2` remains unchanged.

## Source evidence and minimal change

Native `src/impl/pruning_strategy.cpp::select_edges_by_heuristic` rejects redundant
edges using pairwise-distance occlusion. The preceding pilot applied alpha=1
occlusion only to FP32 `GraphBackend::nearest`. This pilot additionally prefers
dropping an occluded reciprocal edge when `GraphBackend::link` overflows.
It retains the existing incoming-edge safety check and farthest-edge fallback,
degree cap, mutation journal and incoming synchronization. Both paths can also
be used during updates; this is not a build-only change. FP16/RaBitQ are excluded.

## Paired measurements

CPU0, one thread, degree 16, build budget 128, query budget 512, top-10,
10,000 vectors, three fresh processes per side with cyclic side order.
Medians; SIFT has 100 queries and GIST 1,000. No Full VSAG comparison here.

| Dataset | Side | Recall@10 | Build ms | Query P50 us | Query P99 us |
|---|---|---:|---:|---:|---:|
| SIFT | Production | .9930 | 536.948 | 164.196 | 187.885 |
| SIFT | Nearest-only | .9960 | 612.611 | 165.896 | 193.654 |
| SIFT | Reciprocal candidate | 1.0000 | 680.933 | 176.115 | 197.214 |
| GIST | Production | .9646 | 1280.466 | 443.239 | 588.863 |
| GIST | Nearest-only | .9743 | 1491.108 | 443.287 | 590.714 |
| GIST | Reciprocal candidate | .9978 | 2118.499 | 610.843 | 795.008 |

GIST recall improves 3.32 percentage points, but P50 increases 37.81% and build
time 65.45%. These reused queries are developer data, not a new held-out acceptance
set. This is a quality gain with a cost regression, not overall performance PASS.

## Evidence and remaining gates

`verification.json` independently checks 18 runs and 9,900 recorded queries,
input hashes, per-query IDs, hit counts and P50/P99. `raw.tar.gz` contains
113 members (1,022,313 bytes), SHA256
`728f307895a9fc9f601b41526b47d358f32b4928107737ac080bcfa00990487b`.
`candidate.patch` is the complete source change, SHA256
`e377e6918491d8966c76602ef9fcee6fafd502e5400d0918e7fc82c4f11c33b9`.
`functional-validation.json` and logs record a separate rebuild of that patch:
Release, RaBitQ enabled, existing CTest suite 5/5 PASS, including graph allocation
rollback. This rebuild is not the measured library identity. No sanitizer,
coverage, long-term real-value CRUD or fresh holdout claim is made for this patch.

Reproduce on the documented host using `run_build_diversity_pilot.py --help`,
the same production source/library and dataset paths recorded in `commands.json`
and input receipts; enable `--reciprocal`. Verify using
`verify_build_diversity_pilot.py OUTPUT --gist GIST_DIRECTORY --sift SIFT_DIRECTORY`.
Next: query-budget calibration, fresh held-out validation, then long-term CRUD
and sanitizer gates before considering production promotion.
