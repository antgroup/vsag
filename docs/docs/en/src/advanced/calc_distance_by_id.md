# Calculate Distance by ID

Besides `KnnSearch` and `RangeSearch`, VSAG exposes APIs that compute the distance between a
query vector and **already-indexed vectors referenced by their IDs**. This is useful for
re-ranking external candidate sets, validating recall, or implementing custom retrieval
pipelines on top of VSAG.

Two flavors are provided:

- `CalcDistanceById`  — single ID, returns one distance.
- `CalcDistancesById` — batch of IDs, returns a `DatasetPtr` containing distances.

Each flavor has two overloads: one taking a raw `const float*` (dense vectors) and one taking
a `DatasetPtr` (works for both dense and sparse vectors).

> **Migration note.** `CalDistanceById` is the historical spelling of the batch method. It remains
> available as a deprecated alias through the 1.1 migration window; new code should use
> `CalcDistancesById`. The two names have identical semantics. See
> [issue #2068](https://github.com/antgroup/vsag/issues/2068).

HGraph also supports configured sparse and non-FP32 data types. In those configurations, use a Dataset containing the native query field (`SparseVectors`, `Int8Vectors`, or `Float16Vectors` for FP16/BF16); a raw `float*` distance query is only valid for a float32 index. The Dataset representation must match the index configuration, not merely its vector dimension.

### C++ implementation migration

`CalcDistancesById` is the canonical virtual batch interface, including its `topk` parameter. Public wrappers and internal index implementations dispatch through this name directly. The deprecated public `CalDistanceById` aliases forward to the canonical interface, never the reverse.

Custom index implementations must override the appropriate `CalcDistancesById` overloads instead of relying on an override of the historical spelling. Existing ordinary calls to the deprecated alias remain source-compatible, but overriding only that alias is not sufficient to implement the canonical interface. The virtual interface layout changes: binary compatibility with previously compiled `Index` subclasses is not preserved. Recompile custom subclasses and dependent C++ components against the matching headers and library.

## API Overview

```cpp
// Single, dense float pointer.
tl::expected<float, Error>
CalcDistanceById(const float* vector,
                 int64_t id,
                 bool calculate_precise_distance = true) const;

// Single, DatasetPtr (dense or sparse).
tl::expected<float, Error>
CalcDistanceById(const DatasetPtr& vector,
                 int64_t id,
                 bool calculate_precise_distance = true) const;

// Batch, dense float pointer.
tl::expected<DatasetPtr, Error>
CalcDistancesById(const float* query,
                const int64_t* ids,
                int64_t count,
                bool calculate_precise_distance = true,
                int64_t topk = -1) const;

// Batch, DatasetPtr (dense or sparse).
tl::expected<DatasetPtr, Error>
CalcDistancesById(const DatasetPtr& query,
                const int64_t* ids,
                int64_t count,
                bool calculate_precise_distance = true,
                int64_t topk = -1) const;
```
For `DatasetPtr` queries with `NumElements() > 1`, check
`SUPPORT_BATCH_CALC_DISTANCE_BY_ID` first. `count` is the number of IDs per query, `ids` must
contain `NumElements() * count` row-major IDs, and returned distances use the same layout. When
`topk > 0`, each query returns `min(topk, count)` entries sorted by distance in ascending order,
and the result also contains the corresponding IDs.


Declarations live in
[`include/vsag/index.h`](https://github.com/antgroup/vsag/blob/main/include/vsag/index.h).

### `calculate_precise_distance`

- `true` (default) prefers an available higher-precision/reorder representation. It does **not** reconstruct discarded original vectors. When no separate precise representation exists, it uses the stored representation, which can still be quantized or pruned.
- `false` uses the base representation. Both modes compute the configured distance; they do not run approximate candidate retrieval. A disk-backed representation can incur I/O in either mode.
- BruteForce/WARP and SIMQ use their stored distance backend for both modes. HGraph/Pyramid prefer raw vectors when retained, otherwise reorder codes; IVF uses its configured reorder codes/buckets. SINDI/SINDI_V2 use rerank vectors when configured.

### Native query representations

| Index | Single-ID query | Multi-query batch query |
|---|---|---|
| BruteForce, HGraph, IVF, Pyramid, LazyHGraph | `float*` or one-row `DatasetPtr` with `Float32Vectors` | `DatasetPtr` with `Float32Vectors` |
| SINDI, SINDI_V2 | one-row `DatasetPtr` with `SparseVectors` | `DatasetPtr` with `SparseVectors` |
| WARP, SIMQ | one-row `DatasetPtr` with `MultiVectors` and `MultiVectorDim` | `DatasetPtr` with `MultiVectors` and `MultiVectorDim` |

SINDI immutable storage supports single-ID and batch distances, including after deserialization. WARP/SIMQ calculate the same multi-vector aggregation as their search distance backend, without coarse candidate selection. Raw float pointers cannot represent these native sparse/multi-vector queries.

For N query rows and `count` candidates per row, provide N × count candidate IDs in row-major order (`ids[q * count + j]`); a single candidate list is **not** implicitly broadcast. `CalcDistancesById` is the preferred batch name; `CalDistanceById` remains the compatibility alias.

Missing labels produce `-1`, but a valid inner-product distance can also be negative (including `-1`). Top-k uses label validity, not the sign or value of the distance, to place missing labels last.

IVF PQFS supports point and batch distances using the same packed-code lookup as bucket scanning. A point lookup reads its 32-vector package and selects one lane; this may perform more work than a scalar quantizer lookup, particularly for disk-backed storage.

PQFS is supported as IVF base bucket encoding, not as flat reorder encoding: flat reorder storage does not maintain PQFS packages. Invalid PQFS reorder configurations are rejected at index creation; choose a supported precise quantizer such as FP32 instead.

### Return Semantics

- The single-ID overload returns the distance as a `float`.
- With `topk == -1`, the raw-pointer batch overload returns one row with `count` distances. The
  `DatasetPtr` overload returns `NumElements()` rows with `count` distances each. Both preserve
  input order and do not return IDs.
- With `topk > 0`, the batch overload returns the smallest `min(topk, count)` distances per
  query, sorted ascending, and `GetIds()` contains the corresponding IDs. Missing IDs (represented by `-1`
  distances) are ordered after valid distances regardless of their sign and only appear if there are not enough valid IDs.
- The distance metric (IP / L2 / cosine) follows the `metric_type` chosen at index
  construction; see [Metric Semantics](../resources/metric_semantics.md).

## Basic Usage

```cpp
#include <vsag/vsag.h>

// 1. Build an HGraph index over float32 vectors.
auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();
index->Build(base);

// 2. Single ID.
auto d = index->CalcDistanceById(query_vector.data(), /*id=*/42);
if (d.has_value()) {
    std::cout << "distance to id 42 = " << d.value() << std::endl;
}

// 3. Batch IDs.
std::vector<int64_t> ids = { 1, 2, 3, 4, 5 };
auto result = index->CalcDistancesById(query_vector.data(), ids.data(), ids.size());
if (result.has_value()) {
    const float* dists = result.value()->GetDistances();
    for (size_t i = 0; i < ids.size(); ++i) {
        if (dists[i] == -1.0f) {
            std::cout << ids[i] << " -> invalid ID" << std::endl;
        } else {
            std::cout << ids[i] << " -> " << dists[i] << std::endl;
        }
    }
}
```

## Multiple Queries

The `DatasetPtr` batch overload accepts more than one query when the index advertises
`SUPPORT_BATCH_CALC_DISTANCE_BY_ID`. Candidate IDs and outputs are row-major. For LazyHGraph, the
call is delegated to the active internal index, so actual support can change after a phase
transition even though the wrapper advertises the feature; handle a returned error:

```cpp
// Two dense queries and three candidate IDs per query.
auto queries = vsag::Dataset::Make()
                   ->NumElements(2)
                   ->Dim(dim)
                   ->Float32Vectors(query_vectors.data())
                   ->Owner(false);
std::vector<int64_t> candidate_ids = {
    10, 11, 12,  // candidates for query 0
    20, 21, 22   // candidates for query 1
};

if (index->CheckFeature(vsag::SUPPORT_BATCH_CALC_DISTANCE_BY_ID)) {
    auto result = index->CalcDistancesById(queries, candidate_ids.data(), 3, true, /*topk=*/2);
    if (result.has_value()) {
        auto batch = result.value();
        // batch has NumElements() == 2 and Dim() == 2.
        for (int64_t q = 0; q < batch->GetNumElements(); ++q) {
            for (int64_t j = 0; j < batch->GetDim(); ++j) {
                const int64_t offset = q * batch->GetDim() + j;
                std::cout << batch->GetIds()[offset] << ": "
                          << batch->GetDistances()[offset] << '\n';
            }
        }
    }
}
```

With `topk == -1`, `GetDim()` is `count` and position `q * count + j` corresponds to input ID
`ids[q * count + j]`. With positive `topk`, the row stride is `GetDim()`, and each row contains
the closest valid candidates followed by invalid IDs only when fewer than `topk` valid IDs exist.

A runnable example is provided in
[`examples/cpp/306_feature_calculate_distance_by_id.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/306_feature_calculate_distance_by_id.cpp).

## Sparse Vectors

For sparse-vector indexes such as SINDI, the `const float*` overloads are not applicable. Pass
the query as a `DatasetPtr` carrying sparse vectors via `SparseVectors(...)`, and use the
`DatasetPtr` overloads:

```cpp
auto query = vsag::Dataset::Make();
query->NumElements(1)->SparseVectors(&sparse_query)->Owner(false);

auto d = index->CalcDistanceById(query, /*id=*/42);
```

## Support Matrix

| Index type | Single-ID dense (`const float*`) | Single-ID DatasetPtr | Multi-query DatasetPtr batch | Notes |
|------------|-----------------------------------|----------------------|------------------------------|-------|
| hgraph | yes | yes | when the feature is advertised | Honors `calculate_precise_distance`. |
| ivf | yes | yes | when the feature is advertised | Availability depends on retained precise storage. |
| brute_force | yes | yes | when the feature is advertised | Dense single-vector indexes only. |
| pyramid | yes | yes | yes | |
| lazy_hgraph | yes | no | depends on the active internal index | DatasetPtr batch calls delegate to the active BruteForce/HGraph; availability can change after a phase transition. |
| sindi | no | yes | yes | Sparse vectors only. |

Indexes that do not implement the API surface for a given overload return an
`UNSUPPORTED_INDEX_OPERATION` error.

## Notes

- The query dimension (for dense overloads) must match the index dimension.
- The batch overload has a default implementation that loops over single-ID calls;
  some indexes override it for batch-level optimization.
- Like all VSAG read-only APIs, these methods are safe to call concurrently with other
  read-only operations (e.g. `KnnSearch`).
