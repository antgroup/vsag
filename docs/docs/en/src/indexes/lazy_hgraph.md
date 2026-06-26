# LazyHGraph

LazyHGraph is an adaptive dense-vector index that starts as an exact
BruteForce index and automatically converts to HGraph after the collection
reaches a configurable `transition_threshold`. It is useful when a dataset
starts small but is expected to grow: early searches stay exact and avoid graph
build overhead, while larger collections get HGraph's approximate-search
latency and quantization options.

- Source: `src/algorithm/lazy_hgraph.{h,cpp}`
- Example:
  [`examples/cpp/111_index_lazy_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/111_index_lazy_hgraph.cpp)

## How it works

1. **Flat phase.** Before the threshold is reached, data is stored in an
   internal BruteForce index using FP32 vectors. Search is exact.
2. **Transition.** When `Build` receives at least `transition_threshold`
   vectors, or `Add` grows the flat phase to that size, LazyHGraph builds an
   internal HGraph from the flat data.
3. **Graph phase.** After transition, new data and search requests are handled
   by the internal HGraph. Search parameters keep using the `hgraph` search
   object.

## Quick start

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "lazy_hgraph": {
        "transition_threshold": 1000,
        "hgraph": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100,
            "build_thread_count": 4
        }
    }
})";
auto index = vsag::Factory::CreateIndex("lazy_hgraph", params).value();

auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Add(base);

auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10, R"({"hgraph": {"ef_search": 100}})").value();
```

## Build parameters

LazyHGraph accepts its build parameters in a top-level `lazy_hgraph` object.
For compatibility with the generic factory shape, the same object may also be
provided as `index_param`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transition_threshold` | uint64 | `1000` | Number of vectors at which LazyHGraph converts from the flat phase to HGraph. Must be positive. |
| `hgraph` | object | `{}` | HGraph build parameters used after transition. See [HGraph](hgraph.md#build-parameters). |

LazyHGraph only supports top-level `dtype: "float32"`. The flat phase is fixed
to FP32 BruteForce storage and does not accept separate flat quantization
parameters.

## Search parameters

Search parameters use the same `hgraph` object as HGraph:

```json
{"hgraph": {"ef_search": 100}}
```

In the flat phase, search is exact. In the graph phase, the internal HGraph uses
the supplied HGraph search parameters such as `ef_search`.

## Lifecycle notes

- `Build` chooses the initial phase from the input size: below
  `transition_threshold` stays flat; at or above the threshold builds HGraph
  directly.
- `Add` can trigger the one-way transition from flat to graph.
- Flat-phase `Remove` always performs physical removal, even if the caller
  passes `RemoveMode::MARK_REMOVE`, so graph transition does not carry
  tombstones.
- `GetExtraInfoByIds`, `UpdateExtraInfo`, and extra-info filtering are
  supported in both phases. See [Extra Info](../advanced/extra_info.md).

## When to use LazyHGraph

- A dense FP32 collection starts small and grows over time.
- Exact results are preferred while the collection is small.
- The same index should automatically switch to HGraph once approximate graph
  search becomes worthwhile.

Use [HGraph](hgraph.md) directly when the dataset is already large at build
time, when you need non-FP32 input types, or when you want graph behavior from
the first insertion.

## See also

- [HGraph](hgraph.md)
- [Extra Info](../advanced/extra_info.md)
- [Creating an Index](../guide/create_index.md)
