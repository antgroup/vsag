# Strict-FP native Full comparison and candidate CRUD gates, 2026-10-12

**Raw/safety gates PASS; candidate GIST post-mutation quality gate FAIL.**
PR2904 remains dbb2fe0. The FP32 reciprocal candidate is not promoted.
This report does not claim overall project completion or RaBitQ improvements.

## Build and protocol

[Fresh native Full strict build](../full-strict-build-r1-20261012/receipt.json):
243 selected core objects from an empty build; ccache OFF; only isolated compiler
configuration changes Ofast to O3. Original official source/library unchanged.
[Patch/compile audit](../full-strict-build-r1-20261012/strict-fp-receipt.json).
Both Lite sides use O3/g, RaBitQ compiled ON but FP32 selected. Consumers O3.
GCC11.4, CPU0, one thread, degree16/build128/top10. Three cyclic fresh runs/side.
This aligns floating optimization semantics, not every flag, algorithm or batching.
[Actual candidate/strict-Full dispatch](../fp32-runtime-dispatch-strict-r1-20261012/receipt.json)
confirms AVX512 scalar on both and AVX512 Batch4 on Full.
Full uses HGraph small profile, memory IO, no raw-vector storage, compressed graph;
Lite uses its own single-level graph/API. Parameter JSON and input hashes retained.

The original1000 GIST query file was already developer data. Fresh validation
uses raw GIST base rows10000:11000 as1000 queries, disjoint from indexed rows0:10000;
this is a separate source cohort, **not the official GIST query-file benchmark**.
Indexed base bytes match the previous cohort. Candidate256/control512/Full512
were frozen before executing these new queries. The .990 minimum is diagnostic,
not an agreed OSPP contract. Independent direct FP64 oracle matches every complete
ordered Top10 record (1000/1000). SIFT100 queries remain development data.

## Frozen query results

Median per-run metrics. RSS includes inputs/loading/warmup/library, not isolated
index memory. CPU includes result bookkeeping; latency excludes Dataset creation
and output IO. Query worker warms the whole cohort once before measuring.

| Dataset / side | Budget | Recall | P50 us | P99 us | Query-phase CPU ms | Peak RSS KiB | Snapshot bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| GIST original Lite |512|.9676|428.238|590.203|429.810|49104|39840064|
| GIST candidate |256|.9946|333.120|439.528|328.828|48956|39840064|
| GIST native Full |512|.9931|505.537|612.042|498.729|109372|39808753|
| SIFT original Lite |384|.9910|124.506|148.226|12.569|30768|6560064|
| SIFT candidate |128|1.0000|55.138|69.288|5.442|30768|6560064|
| SIFT native Full |128|.9580|81.008|92.357|8.047|38880|5766017|

Fresh GIST candidate vs strict Full: P50-34.11%, P99-28.19%, phase CPU-34.07%,
process peak RSS-55.24%; both meet diagnostic .990 in all3trials. Recall .9946
vs.9931 is threshold-qualified, not identical-recall or globally optimal tuning.
Full was not tuned across all configurations on this fresh cohort.
Candidate snapshots are larger: GIST+31,311bytes, SIFT+794,047bytes.
SIFT Full128 fails .990, so do not advertise its timing as equal-quality.
[Development-only followup](../aligned-followup-r1-20261012/README.md) finds Full1024
.996/P50493.277us; this is post-hoc development calibration, not frozen validation.

## Three whole-index mutation passes

Per dataset30,000 genuine vector changes using FP32 .875*current+.125*original
donor, followed by Remove/Add. Three permutations visit every live ID each pass:
90,000 mutation calls/dataset,180,000 total. Changed truth is direct FP64 on the
final vector state. Consumers verify live count and exact pre/post Save-Load
query-result identity. No concurrent access or hours-long endurance claim.

| Dataset | Initial / final recall on100queries | CRUD total ms | Query P50/P99 us | Save / load ms |
|---|---|---:|---|---|
| SIFT |1.000 / .995|5353.697|54.939 /68.509|3.728 /4.137|
| GIST |.992 / .973|16149.431|328.101 /405.398|16.964 /17.024|

Each API has30,000 samples. Timed API P50/P99 us:
SIFT Update88.357/111.447, Remove12.559/16.830, Add78.628/94.188;
GIST Update295.832/440.577, Remove19.379/42.998, Add220.374/268.941.
Per-API throughput (calls divided by summed API time) is in verification.json;
it is not concurrent throughput or an elapsed-process CPU measurement.
Full CRUD was not rerun in this candidate pilot; use the separately pinned
resident mixed-CPU matrix for its own measured implementation, not this candidate.

GIST quality deteriorates1.9 percentage points. Followup candidate budget1024
recovers.997 but P50~1046us, losing the low-budget advantage. Do not conceal this
by presenting only initial-index results or changing the frozen budget afterward.
Next structural work: mutation repair currently fills holes from nearest old
neighbors; test diversity-aware repair rather than claiming increasing ef is a fix.

## Evidence and limits

Original and candidate Release CTest5/5 each; candidate ASan+UBSan5/5 with leak
detection enabled. These are not coverage or native Full full-suite claims.
Raw verifierPASS18queryruns/9900recorded queries plus180000mutation samples;
builder incidental/warmup queries are additional, not included in this denominator.
`raw.tar.gz`:228members/22397608bytes, SHA256
`c38e5a8405cdfda0cd20fd2ef9d2ea9b41f4d11cd70f796b267f6344bd4784e4`.
The archive includes dataset/config/results/build/tests; verification/oracle added
after archive separately retained. Use run_aligned_fp32_full.py, its pinned paths,
verify_aligned_fp32_full.py --sift SIFT_DIR and check_quality_target_oracle.py.
No new deployment-package/load-only/100k/FP16/RaBitQ acceptance claim here.
