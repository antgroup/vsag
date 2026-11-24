# Calculate Distance by ID

Besides `KnnSearch` and `RangeSearch`, VSAG exposes APIs that compute the distance between a
query vector and **already-indexed vectors referenced by their IDs**. This is useful for
re-ranking external candidate sets, validating recall, or implementing custom retrieval
pipelines on top of VSAG.

Two flavors are provided:

- `CalcDistanceById`  — single ID, returns one distance.
- `CalDistanceById`   — batch of IDs, returns a `DatasetPtr` containing distances.

Each flavor has two overloads: one taking a raw `const float*` (dense vectors) and one taking
a `DatasetPtr` (works for both dense and sparse vectors).

> **Note on naming.** The batch method is currently spelled `CalDistanceById`
> (missing the `c` in `Calc`). This is a historical typo introduced when the
> batch overload was first added; the two names do **not** indicate any
> semantic difference beyond *single vs. batch*. The current spelling is
> kept for backward compatibility and is expected to be **deprecated** in a
> future release in favor of a correctly spelled name (proposed:
> `CalcDistancesById`). New code is encouraged to centralize calls behind a
> thin wrapper to ease the eventual migration. See
> [issue #2068](https://github.com/antgroup/vsag/issues/2068) for tracking.

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
CalDistanceById(const float* query,
                const int64_t* ids,
                int64_t count,
                bool calculate_precise_distance = true) const;

// Batch, DatasetPtr (dense or sparse).
tl::expected<DatasetPtr, Error>
CalDistanceById(const DatasetPtr& query,
                const int64_t* ids,
                int64_t count,
                bool calculate_precise_distance = true) const;
```

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
- The batch overload returns a `DatasetPtr` whose `GetDistances()` array has `count` entries
  aligned with the input `ids`. An ID that does not exist or has been removed gets a **`NaN`** distance, and the result
  carries a `Valids` mask (`GetValids()`, same length as `GetDistances()`) that is `false` for
  those entries. `GetValids()` is `nullptr` when `count == 0`.
- The single-ID overload returns an `INVALID_ARGUMENT` error (`label <id> does not exist or is removed`)
  when the ID does not exist or has been removed.
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
auto result = index->CalDistanceById(query_vector.data(), ids.data(), ids.size());
if (result.has_value()) {
    const float* dists = result.value()->GetDistances();
    const bool* valids = result.value()->GetValids();
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!valids[i]) {
            std::cout << ids[i] << " -> invalid ID" << std::endl;
        } else {
            std::cout << ids[i] << " -> " << dists[i] << std::endl;
        }
    }
}
```

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

| Index type   | Dense overload (`const float*`) | DatasetPtr overload | Notes |
|--------------|---------------------------------|---------------------|-------|
| hgraph       | yes                             | yes                 | Honors `calculate_precise_distance`. |
| ivf          | yes                             | yes (default loop)  | |
| brute_force  | yes                             | yes (default loop)  | Always precise (no quantization). |
| pyramid      | yes                             | yes (default loop)  | |
| sindi        | no                              | yes                 | Sparse vectors only. |
| diskann      | yes (opt-in)                    | no                  | Requires `support_calc_distance_by_id=true` at build time; otherwise calls return `UNSUPPORTED_INDEX_OPERATION`. |

Indexes that do not implement the API surface for a given overload return an
`UNSUPPORTED_INDEX_OPERATION` error.

### DiskANN (opt-in)

DiskANN only enables this feature when the build parameter
`support_calc_distance_by_id` is set to `true` (default `false`), because the lookup needs an
extra `tag -> internal id` map to be kept in memory.

- When the parameter is `false`, `CheckFeature(SUPPORT_CAL_DISTANCE_BY_ID)` returns `false`
  and both `CalcDistanceById` and `CalDistanceById` return `UNSUPPORTED_INDEX_OPERATION`.
- When the parameter is `true`, DiskANN follows exactly the same contract as the other
  indexes: a missing ID makes the single-ID call return `INVALID_ARGUMENT`, while the batch
  call reports `NaN` plus `false` in `GetValids()` for that slot and never fails because of
  missing IDs. `count == 0` yields an empty result with `GetValids() == nullptr`.
- On an empty or not-yet-built index (with the feature enabled), the single-ID call returns
  `INVALID_ARGUMENT` and the batch call reports every requested ID as invalid.
- Only the dense `const float*` overloads are supported; the `DatasetPtr` overloads return
  `UNSUPPORTED_INDEX_OPERATION`.

## Notes

- The query dimension (for dense overloads) must match the index dimension.
- The batch overload has a default implementation that loops over single-ID calls;
  some indexes override it for batch-level optimization.
- Like all VSAG read-only APIs, these methods are safe to call concurrently with other
  read-only operations (e.g. `KnnSearch`).
