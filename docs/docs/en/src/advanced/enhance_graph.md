# Graph Index Enhancement

Graph-based indexes may see recall drops on "hard queries" — queries that are
poorly connected to their true nearest neighbors. VSAG patches these queries online or offline
using a **conjugate graph**, noticeably improving tail recall at almost zero index-size cost.

## Enabling the Conjugate Graph

At build time:

```json
{
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 400,
        "use_conjugate_graph": true
    }
}
```

At search time, toggle it via the `use_conjugate_graph_search` key in the search-parameter JSON
(there is no boolean overload on `KnnSearch`):

```cpp
std::string search_param_json = R"({
    "hgraph": {
        "ef_search": 100,
        "use_conjugate_graph_search": true
    }
})";
auto result = index->KnnSearch(query, k, search_param_json);
```

## How It Works

Call `Feedback(query, k, search_parameters, global_optimum_id)` after a hard query when the exact
nearest label is known. Omitting `global_optimum_id` makes HGraph compute it by an exact scan.
For offline enhancement, `Pretrain(base_ids, k, search_parameters)` generates queries between the
chosen float32 base vectors and their neighbors, then feeds the resulting failures back. Both
methods return the number of newly inserted conjugate edges; redundant feedback returns zero.

The conjugate graph maps local-result labels to known global optima and contributes extra
candidates after HGraph's normal traversal. `use_conjugate_graph` is disabled by default; calling
`Feedback` or `Pretrain` without it returns `UNSUPPORTED_INDEX_OPERATION`. Search enhancement is
enabled by default for an enabled graph and can be disabled per search.

## Example

`examples/cpp/304_feature_enhance_graph.cpp` walks through building, training, and comparing
recall end-to-end.

## When to Use It

- Data distributions with sparse clusters or outliers.
- Online services sensitive to P99 recall.
- You want a recall boost without rebuilding the index.

## Notes

- Build time increases slightly when enabled.
- Conjugate-graph data is serialized together with the index.
- `UpdateId` updates conjugate edges as well as the HGraph label table.
- `Pretrain` currently supports float32 HGraph indexes; `Feedback` supports all HGraph dtypes.
- It can be combined with `Tune` — they target route quality and runtime parameters respectively.
