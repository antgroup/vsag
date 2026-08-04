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

- `true` (default): the implementation tries to use the **high-precision** representation
  of the stored vector (e.g. full-precision float32). When the index only retains quantized
  codes, obtaining the precise value can be more expensive.
- `false`: the implementation may use the **quantized / approximate** representation that
  the index already keeps in memory. Faster, but the returned distance is approximate.

### Return Semantics

- The single-ID overload returns the distance as a `float`.
- With `topk == -1`, the raw-pointer batch overload returns one row with `count` distances. The
  `DatasetPtr` overload returns `NumElements()` rows with `count` distances each. Both preserve
  input order and do not return IDs.
- With `topk > 0`, the batch overload returns the smallest `min(topk, count)` distances per
  query, sorted ascending, and `GetIds()` contains the corresponding IDs. Invalid IDs (`-1`
  distances) are ordered after valid distances and only appear if there are not enough valid IDs.
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
    auto result = index->CalDistanceById(queries, candidate_ids.data(), 3, true, /*topk=*/2);
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
| hnsw | yes | no | no | Dense batch supports `topk == -1` only. |
| diskann | yes | no | no | Dense batch supports `topk == -1` only. |

Indexes that do not implement the API surface for a given overload return an
`UNSUPPORTED_INDEX_OPERATION` error.

## Notes

- The query dimension (for dense overloads) must match the index dimension.
- The batch overload has a default implementation that loops over single-ID calls;
  some indexes override it for batch-level optimization.
- Like all VSAG read-only APIs, these methods are safe to call concurrently with other
  read-only operations (e.g. `KnnSearch`).
