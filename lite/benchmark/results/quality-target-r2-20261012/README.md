# GIST common-recall-target calibration and independent query cohort, 2026-10-12

**Data audit PASS; Lite common-target acceptance FAIL.** A predeclared Recall@10
target of .990 is a diagnostic threshold, not an extra OSPP contractual threshold.
Do not turn a successful raw audit into a successful quality/performance claim.

Both library sources remain `dbb2fe0`: current PR2904 Lite and the fresh native
Full core from [the preceding build](../full-fresh-build-r4-20261012/receipt.json).
No production source, index format or graph/quantization strategy was modified.
Native Full uses root Release -Ofast/-g, Lite -O3; matching selected SIMD execution
and compiler semantics are still open. Full is native HGraph, not an old Lite.

## Protocol

GIST-960, unchanged 10k base prefix, 1,000 distinct source queries. The first
100 queries/truth are byte-identical to the previous experiment. Queries 0-99
alone choose parameters; queries 100-999 (900 distinct) verify frozen choices.
The existing `prepare_gist.cpp` recomputes ground truth. An independent direct
FP64 coordinate-difference oracle, without the Lite distance kernel or fast-math,
agrees on all 1,000 complete ordered Top-10 records. Input hashes are retained.

Three fresh snapshots per side/storage, degree16/build128, six query budgets
128/256/512/1024/2048/4096. Each snapshot is fixed while scanning budgets; its
SHA must remain unchanged. Budget order reverses in the middle trial. Select
the smallest ef meeting .990 in **all three calibration trials**, or mark no
qualified configuration (4096 fallback is diagnostic only). Freeze all six
choices before executing the held-out query phase. Side order alternates.
Builders also execute incidental full-cohort baseline queries; those outputs
are never inputs to selection. Thus held-out means excluded from parameter
selection, not that those vectors were never executed or a new blinded trial.

CPU0, BLAS/OpenMP thread limits1; query workers assert scheduled CPU0. Each
query worker loads its snapshot in a fresh process, warms the entire selected
cohort once, then measures one pass. Native Dataset construction is outside the
API timer; result copying and output IO are outside that timer. Phase getrusage
CPU includes result bookkeeping, not pure API CPU. Peak RSS includes input
vectors, load/warmup and results: it is not isolated index RSS or fresh-load RSS.

108 calibration processes plus 18 held-out processes, 27,000 recorded API
queries / 270,000 neighbors. Warmup adds 27,000 unrecorded executions. There
are **1,000 distinct** queries, not 27,000 distinct queries. Builder baseline
queries are additional and are not included in this query-phase denominator.

## Frozen-budget held-out results

Medians of three processes; P50/P99 are medians of per-process nearest-rank
quantiles over 900 queries, not pooled quantiles. CPU is for the 900-query phase.

| Storage/side | Frozen ef | Recall@10 | P50 us | P99 us | Phase CPU ms | Process peak RSS KiB | All trials >= .990 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 Lite | 2048 | .989556 | 1522.077 | 1728.693 | 1367.453 | 48832 | No |
| FP32 Full | 512 | .992889 | 529.496 | 645.763 | 471.873 | 108708 | Yes |
| FP16 Lite | 2048 | .989222 | 1056.080 | 1261.224 | 949.677 | 30068 | No |
| FP16 Full | 512 | .992444 | 440.098 | 495.046 | 388.921 | 70436 | Yes |
| RaBitQ8 Lite | 2048 | .985111 | 1309.674 | 1373.472 | 1171.145 | 25760 | No |
| RaBitQ3+5 Full | 2048 | .991222 | 1463.168 | 1549.347 | 1295.692 | 52556 | Yes |

Full RaBitQ held-out recall spans .990444-.991222. Other per-side results have
identical hits across trials; this is observed, not a global deterministic claim.
Lite FP32/FP16 P50 is about 2.875x / 2.400x Full after raising budgets to pursue
quality. RaBitQ Lite has lower latency and RSS here, but misses the target and
has lower recall, so that is **not a same-quality win**. Within Lite, RaBitQ P50
is about24% slower than FP16 with worse recall, while its process peak is lower.
All Lite modes passed calibration but failed validation. Do not tune on these
900 results and then relabel them unseen validation. This is not a new CRUD,
endurance, load, package, 100k, or statistical-significance experiment.

## Reproduction and evidence

`run_quality_target.py --output <new-directory>` accepts source/library/cache
and dataset path overrides; it refuses overwriting outputs and pins identities.
`verify_quality_target.py <output> --dataset <prepared-scale-directory>` checks
coverage, manifests, quantiles, IDs/ranks/distances/truth hits, snapshot identity,
CPU budgets and frozen parameter decisions. `check_quality_target_oracle.py`
independently recompiles the direct FP64 oracle; no NumPy installation is needed.
CLI relocation options were added after measurement; the exact measured driver
and matching C++ source are separately archived with hashes. The timed loop is
unchanged. Format/tidy15 and 11 Python tests pass; a dependency-only tidy warning
is disclosed. No fresh production coverage or full upstream suite is claimed.

[Raw archive](raw.tar.gz): 3,729,401 bytes / 597 members, SHA-256
`bac805acedd7032ba2580be616634ca64207078657687ddbc348facc12997498`.
[Supplementary audit/oracle evidence](audit-evidence.tar.gz): 148,029 bytes / 12
members, SHA-256 `69d3fe79766559e9da17cde7b19db6fd611447f05f35ef3936e93e355e67ef80`.
Source vectors are not republished; prepared data/raw records remain on server.
The next optimization must diagnose graph reachability versus quantized ranking,
not claim that larger ef has solved Lite's quality/cost tradeoff.
