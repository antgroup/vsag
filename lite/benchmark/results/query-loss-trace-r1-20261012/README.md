# Fixed-index GIST loss decomposition, 2026-10-12

This is an instrumented **diagnosis**, not a new latency/CPU benchmark or an
implemented production optimization. Current Lite source/library remains PR2904
dbb2fe0 / SHA3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330.
Rebuilt snapshots match the preceding quality-target experiment byte-for-byte,
for all three prior Lite trials per storage. The 30,000 ordered native neighbor
IDs and FP32 distance bits match its ef2048 calibration/validation records.

10k GIST-960, degree16/build128/query2048, same 1000 source queries and FP64-
cross-checked truth. Raw diagnosis covers 1000 queries/storage; the table uses
only the previously defined 900 validation queries (9000 truth IDs/storage).
No data is relabeled as newly unseen, and no parameters are selected here.

| Storage | Formal hits | Unvisited misses | Visited ranking misses | Coarse shortlist misses | Mean visited nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 8906 | 94 | 0 | 0 | 5842.458 |
| FP16 | 8903 | 95 | 2 | 0 | 5841.834 |
| RaBitQ8 | 8866 | 93 | 41 | 0 | 5843.828 |

Loss categories are disjoint and exactly account for all missed truth IDs.
Unvisited means the always-true filter callback never observed that ID. Ranking
loss means visited but not in the complete-distance top10 over visited nodes.
A shortlist loss would mean recovered by reranking all visited nodes but absent
from the formal coarse-candidate rerank result; none occur in this workload.
FP32/FP16 callback and formal results agree in every rank/distance. RaBitQ's
internal unfiltered state result agrees with the public API, and its direct
unfiltered visit count agrees with the callback mask. The callback only adds
accepted-result bookkeeping in the reviewed 10k search path; this is not a
general proof that arbitrary filters or other scales have identical routing.

FP32/FP16 share94 unvisited(query,truth-ID) pairs; FP32/RaBitQ share92, as do
FP16/RaBitQ. This points toward a shared routing-coverage limitation rather than
only a RaBitQ shortlist problem. It does not isolate the causal graph change.

A full scan using the current RaBitQ complete 3+5-bit distance returns8958/9000
truth IDs (Recall .995333). This is an observed achievable full-scan point on
these codes, not a theoretical recall ceiling or efficient ANN solution. It
supports investigating graph coverage before adding precision bits; it does not
prove that a bounded graph search can reach .990 at an acceptable cost.

## Source evidence and next candidate

`src/lite/graph_backend.cpp::nearest` gathers construction candidates then keeps
the closest count. `rabitq_graph_state.h::nearest` also retains nearest candidates;
its existing diversity check is in `repair`, not this initial selection.
Native `src/impl/pruning_strategy.cpp::select_edges_by_heuristic` instead applies
the `alpha * pairwise_distance < query_distance` occlusion rule; native HGraph
calls it during construction and reciprocal connection. Its per-pass distance
computer cache is also relevant. These source differences are real; no candidate
is adopted yet. Test a minimal construction/reciprocal selection change separately,
with fixed degree/budget, CPU/quality and rollback/CRUD regressions, before promotion.

The verifier checks manifests, prior snapshot identities and ordered results,
truth-row identity, raw classification sums, overlaps and complete cell coverage.
It does not separately reimplement the codec to validate the all-code flags.
Format/tidy15 and17 Python tests pass. Full is not rerun; no new package/load/
endurance/coverage claims. Raw40-member archive727288B SHA-256
`2c7a14012f30ba8e98bcd96bb89fd447cfce4522f7260838e54b9c1b1e87170d`.
All original raw files, prepared data and source remain on the server.
