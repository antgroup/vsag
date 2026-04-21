# Range Search

Besides k-nearest-neighbor search (`KnnSearch`), VSAG also supports **range search**
(`RangeSearch`): return every result whose distance to the query vector is less than or equal to
a given radius. It is useful for threshold filtering, de-duplication, and approximate recall
scenarios.

## Basic Usage

```cpp
#include <vsag/vsag.h>

// 1. Create an index (HNSW in this example)
auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_params).value();
index->Build(dataset);

// 2. Prepare the query
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(dim)->Float32Vectors(query_vec)->Owner(false);

// 3. Range search
float radius = 0.5f;
auto result = index->RangeSearch(query, radius, search_params);
if (result.has_value()) {
    auto ids = result.value()->GetIds();
    auto dists = result.value()->GetDistances();
    int64_t n = result.value()->GetDim();
    // ...
}
```

> See `examples/cpp/302_feature_range_search.cpp` for a complete example.

## `limited_size` Parameter

`RangeSearch` accepts a `limited_size` argument that caps the number of returned results:

```cpp
// Return at most 100 results within the radius
auto result = index->RangeSearch(query, radius, search_params, /*limited_size=*/100);
```

- `limited_size = -1` (default): return every result inside the radius (unlimited).
- `limited_size > 0`: return at most this many results.
- `limited_size = 0`: invalid; the implementation explicitly rejects this value
  (`CHECK_ARGUMENT(limited_size != 0, ...)`).

## Combining with Filter

`RangeSearch` has the same signature shape as `KnnSearch` and also accepts a filter (see
`examples/cpp/301_feature_filter.cpp`). The filter is applied during the search, not afterwards,
which is more efficient than post-filtering.

## Support Matrix

| Index type | Supports RangeSearch |
|------------|----------------------|
| hnsw | yes |
| hgraph | yes |
| diskann | yes |
| ivf | yes |
| brute_force | yes |
| sindi | yes (sparse vectors) |

## Notes

- The distance metric (IP / L2 / cosine) defines the semantics of `radius`. Make sure it matches
  the `metric_type` specified at index creation.
- If `radius` is very large, the result set can be huge; combine with `limited_size` to avoid
  unbounded memory usage.
- For graph-based indexes (HNSW / HGraph), runtime parameters like `ef` share the same meaning
  between `RangeSearch` and `KnnSearch`.
