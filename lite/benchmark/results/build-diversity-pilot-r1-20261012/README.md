# FP32 initial-neighbor diversity pilot, 2026-10-12

A real isolated code candidate improves measured recall but increases build and
query costs. **Not adopted in PR2904.** No Full run, RaBitQ optimization or complete
quality/CRUD acceptance is claimed. PR2926 contains the runner, patch and evidence.

Source: PR2904 dbb2fe0. A RAM-only copy changes GraphBackend::nearest for FP32:
apply native `src/impl/pruning_strategy.cpp::select_edges_by_heuristic`'s alpha1
distance-occlusion condition to the existing distance-sorted candidate list;
fill remaining slots by closest unselected candidates to preserve Lite's degree
contract. This reuses the official selection logic, **not** its Full-dependent
implementation. Reciprocal `link`, degree16, construction ef128, query ef512,
filtering, format and FP16/RaBitQ behavior remain unchanged. This candidate also
affects FP32 Update's call to nearest; it must not be described as build-only in
production behavior. Mutations were not exercised by this initial-build pilot.

Candidate/control: SHA20dd9b45decafca612d346e05ff152d146f7eff422a3dca2cdc498dbb24e81e4 /
3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330.
Candidate size412400B is not complete-package evidence. Existing control uses
the previous Release library; candidate freshly builds the official standalone
Lite CMake target in RAM. Both consumers use the same pinned helper source.

CPU0/thread limits1, three fresh alternating pairs per dataset, FP32 only.
GIST10k960d uses the prior1000 queries (now development samples, not fresh heldout);
SIFT10k128d uses100 historical queries. 12 runs6600 recorded query results.
No builds overlap timings. Values are medians of three per-process aggregates.

| Dataset | Recall control/candidate | Build ms control/candidate | P50 us control/candidate | P99 us control/candidate | Query win/loss/tie |
| --- | ---: | ---: | ---: | ---: | --- |
| SIFT | .993/.996 | 534.900/610.595 | 162.916/165.324 | 187.454/192.795 | 2/0/98 |
| GIST | .9646/.9743 | 1281.724/1496.883 | 444.298/450.247 | 591.683/602.173 | 134/64/802 |

GIST gains.97 percentage points, build+16.79%, P50+1.34%; SIFT gains.30points,
build+14.15%, P50+1.48%. Snapshot bytes unchanged6560064/39840064, not contents.
GIST still misses the.990 diagnostic target; higher quality is not faster code.
Raw verifier recomputes each query's hits, IDs/ranks/distance validity, quantiles,
complete cells, input/source hashes and observed repeat hit equality. Builder's
Save/Load equality checks pass, but this is not allocation rollback, mixed CRUD,
SAN/OFF/coverage/Full tests or statistical significance. No additional CPU/peak
measurement is claimed. Only the candidate patch is formatted15; no production
lint/coverage claim. Next: evaluate reciprocal edge selection and consistent
pairwise scoring, then run mutation/failure safety and new independent queries
before any promotion. Raw76-member684132B archive SHA-256
0122e1546b60ecfc9406610a0ed1ffbcac494abf696662bebb573006ae846ee1.
Original records/data/source preserved; own RAM build removed after receipt.
