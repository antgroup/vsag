# Graph Index Enhancement

Graph-based indexes (HNSW, HGraph) may see recall drops on "hard queries" — queries that are
poorly connected to their true nearest neighbors. VSAG patches these queries online or offline
using a **conjugate graph**, noticeably improving tail recall at almost zero index-size cost.

## Enabling the Conjugate Graph

At build time:

```json
{
    "hnsw": {
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
    "hnsw": {
        "ef_search": 100,
        "use_conjugate_graph_search": true
    }
})";
auto result = index->KnnSearch(query, k, search_param_json);
```

## How It Works

The conjugate graph is built by inverting "failure paths" over the training data on the original
graph and then used as additional candidate edges during greedy expansion at search time. It is a
lightweight patch on the main graph, typically below 10% of the main graph's size.

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
- It can be combined with `Tune` — they target route quality and runtime parameters respectively.
