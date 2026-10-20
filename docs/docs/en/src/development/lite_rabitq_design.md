# Lite RaBitQ8: current implementation contract

Updated 2026-10-12. This contract supersedes historical NONE/reverse-disabled
experiment descriptions. The public source is `src/lite/rabitq_backend.cpp`.

## What is implemented

`ENABLE_RABITQ_LITE_BACKEND=ON` enables explicit
`BuildGraph(VectorStorage::RABITQ8, degree, ef)` from a nonempty BruteForce index.
Degree is 2–64; ef is at least degree. Training validates dimension/count limits,
finite inputs and model arithmetic; failure leaves the original backend intact.
The fixed profile uses a deterministic centroid/rotation, 3-bit coarse filter
codes and 5-bit supplement codes. Native SIMD traits/kernels are reused, with
generic fallback and scalar differential tests. It is not the whole native Full
RaBitQ index and cannot reconstruct exact FP32 input vectors.

Add, changed Update, physical Remove, filtered Search, SearchWithOptions and
Save/Load are implemented. The model remains fixed after training: distribution
drift can degrade quantization and graph quality. There is no automatic retraining,
quality-aware budget selection, exact-FP32 rerank or background rebuild promise.

## Public mutation policy: exact source correspondence

The adapter constructor executes:

```cpp
state_.ConfigureIncomingProtection(
    rabitq::IncomingProtection::CACHED, false, false, false, true);
```

Arguments correspond to mode, protect_add, protect_remove,
protect_add_displaced and protect_link. Both Build and Load enable incoming
adjacency. Thus Update protection and cached incoming counts are active;
safe link pruning is active; the three extra stage guards are inactive.
This is an internal fixed public-adapter policy, not a user-exposed tuning option.
Safe pruning preserves a target's last incoming edge when a safe alternative
exists; its fallback does not guarantee complete connectivity or recall.

Incoming slots are checked/narrowed to uint32. Transaction journals capture
changed code/ID/outgoing/incoming rows and cached counts; cold/unsupported internal
states can still use copy fallback. Costs depend on visited/local affected rows,
degree, graph indegree, capacity growth and fallback, not constant-time CRUD.
Update prepares its normalized query/code once and skips a complete identical
stored representation, including all metadata. Equal encoding does not mean equal
raw vector. Failure preserves logical state but can retain increased capacity.
Remove bool conflates missing ID and failed graph allocation.

## Query, storage and persistence

The index stores model, IDs/map, two contiguous code planes, metadata, outgoing
and incoming adjacency; no exact FP32 matrix. Query transform, coarse routing,
supplement-code refinement and result selection remain real costs. Cached query
sums/coarse inner products, batch4 kernels, bounded heaps, large-graph visited
bitmap and routing-only estimates reduce selected costs without changing budgets.
Distances are approximate quantized L2 estimates. A filter can yield fewer than k.
Residual radii retain their actual nonnegative norm, including zero at the centroid;
the safe normalization divisor does not replace the geometric radius. Finite queries
with k=0 or an empty index return before transformation, after input validation.
Ordinary finite scoring keeps its float evaluation order. Non-finite intermediate
distance/bound results are recomputed in double before conversion: representable
answers such as a large-vector self-match remain valid. Estimates outside the float
range are represented by signed infinity; NaN scoring is rejected before heap ordering.

Save streams model/codes/IDs/CSR-format topology without recreating original vectors.
Load restores identical state without training and remains mutable even after
deleting all rows and re-adding. Format `VSAGLQ01` v1 is independent of floating
`VSAGLT01` v1/v2/v3 and native Full. Integer/float bit fields are little-endian;
header sizes, finite metadata, IDs, adjacency, short/trailing input are validated.
Model/code profile changes require explicit format compatibility handling.
The residual-norm correction keeps the v1 field layout and reads existing positive-norm
snapshots without re-encoding. Corrected snapshots can contain zero residual norms and
require the corrected reader; older readers may reject them. Loading an old snapshot
does not repair radii previously clamped to one: rebuild from original vectors when
that case matters.
No checksum, fsync, atomic file replacement or same-instance concurrent calls.

## Test correspondence and completeness

`rabitq_backend_test.cpp` covers API lifecycle, filtering/budgets, mutable round
trips, malformed input, fixed-model encoding, scalar/SIMD plane tails, safe-link
cache invariants, incoming slot/edge validation, bounded heaps and streamed I/O.
Allocation fault tests cover rollback; no test can promise all allocator/platform
failure cases. See [Lite requirements and tests](lite_first.md).

RaBitQ is functionally implemented, not quality/performance-complete. GIST
maintenance recall and fixed-model drift remain open. Existing 10k/100k and real
changed-vector studies do not replace hours-long interleaved reads/mutations,
blind quality cohorts or quality-matched Full comparison. Increasing ef is a
quality/cost tradeoff, not a missing-function fix or universal speedup.
Historical performance reports belong to benchmark PR #2926. Do not reinterpret
their NONE/reverse-disabled configuration as current production or combine
best values from different measured versions.
