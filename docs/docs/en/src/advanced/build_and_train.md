# Build and Train

VSAG separates index construction into three stages:

1. **Train** — fit any internal quantizers / partitioners on a sample of the data.
2. **Add** — insert vectors into the index using those trained encoders.
3. **Build** — convenience wrapper that does `Train` then `Add` on the same dataset.

Most users only call `Build`. Two situations are worth knowing about explicitly:

- **`Train` + streaming `Add`.** When the corpus is large or arrives incrementally, train on a
  representative sample first and then stream the rest via `Add` (no rebuild). See
  [`examples/cpp/311_feature_train.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/311_feature_train.cpp).
- **ODescent.** An alternative graph-construction algorithm for HGraph / Pyramid that builds the
  whole neighbor graph in batch instead of insertion-by-insertion. See
  [`examples/cpp/312_feature_odescent.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/312_feature_odescent.cpp).

## The `Train` API

```cpp
tl::expected<void, Error> Index::Train(const DatasetPtr& data);
```

Declared in `include/vsag/index.h`. Trains the index on a (typically sampled) dataset
without inserting it. Returns `tl::expected<void, Error>`; check `.has_value()`.

Indexes that perform meaningful training: **HGraph**, **IVF**, **BruteForce**, **WARP**,
**Pyramid**. For all of them, `Build(data)` first trains and then inserts the vectors —
for the default NSW graph it calls the equivalent of `Train(data)` followed by `Add(data)`,
while for HGraph/Pyramid configured with `graph_type: "odescent"` the insertion step is a
batch ODescent graph build instead of `Add`
(see `HGraph::build_by_odescent` / `Pyramid::Build` in `src/algorithm/`).

### When you need to call `Train` explicitly

- The base quantizer requires training. The capability flag
  [`IndexFeature::NEED_TRAIN`](https://github.com/antgroup/vsag/blob/main/include/vsag/index_features.h)
  reflects this on HGraph and IVF: HGraph sets it whenever `base_quantization_type` is **not**
  one of `fp32`, `fp16`, `bf16` (`src/algorithm/hgraph.cpp:1803`); IVF always sets it
  (`src/algorithm/ivf.cpp:316`) because its centroids must be trained. Pyramid does **not**
  currently set `NEED_TRAIN` in `InitFeatures()` even when its underlying HGraph quantizer
  would need training, so do not rely on `HasFeature(NEED_TRAIN)` for Pyramid — call `Train`
  explicitly when you choose a trained `base_quantization_type`. fp32 / fp16 / bf16 do not
  require training (you can still call `Train` — it is a harmless no-op).
- You want to insert vectors in many small batches rather than in one `Build` call.
- You plan to export the trained model and reuse it on another index instance
  (via `ExportModel`).

### Pattern: train once, add in a stream

```cpp
auto params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "max_degree": 32,
        "ef_construction": 100,
        "base_quantization_type": "sq8"
    }
})";
auto index_result = vsag::Factory::CreateIndex("hgraph", params);
if (!index_result.has_value()) {
    std::cerr << "Create index failed: " << index_result.error().message << std::endl;
    return -1;
}
auto index = index_result.value();

// Step 1 — train on the whole base (or a representative sample).
auto train_result = index->Train(base);
if (!train_result.has_value()) {
    std::cerr << "Train failed: " << train_result.error().message << std::endl;
    return -1;
}

// Step 2 — stream vectors in one at a time (or in small batches).
for (int64_t i = 0; i < num_vectors; ++i) {
    auto one = vsag::Dataset::Make();
    one->NumElements(1)
       ->Dim(dim)
       ->Ids(ids + i)
       ->Float32Vectors(vectors + i * dim)
       ->Owner(false);
    auto add_result = index->Add(one);
    if (!add_result.has_value()) { /* handle */ }
}
```

The complete program is
[`examples/cpp/311_feature_train.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/311_feature_train.cpp).

### `Train` vs `Build` vs `Add`

| Call | Trains quantizer? | Inserts vectors? | Use it when |
|------|-------------------|------------------|-------------|
| `Build(data)` | yes | yes (all of `data`) | Bulk-load: you have the whole dataset already. |
| `Train(data)` | yes | no | You want to insert vectors later, possibly in batches. |
| `Add(data)` | no (requires prior `Train` or `Build`) | yes | Incremental inserts after the index is trained. |

## ODescent: an alternative graph builder

By default, HGraph and Pyramid build their graphs **NSW-style** — every vector is inserted one
at a time and connects to the neighbors found by a search-on-insert (`graph_type: "nsw"`).
**ODescent** ("Optimized NN-Descent") is an alternative: it seeds a random k-NN graph over the
entire dataset and then iteratively refines edges using sampled candidate exchanges.

ODescent typically produces graphs with comparable recall to NSW at lower build cost for large
batches, because the refinement loop parallelizes cleanly over the data and avoids per-insert
search.

ODescent is implemented in `src/impl/odescent/odescent_graph_builder.{h,cpp}` and is currently
used by **HGraph** and **Pyramid** (build path).

### Enabling ODescent on HGraph

Add `graph_type: "odescent"` to the HGraph `index_param`:

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 26,
        "ef_construction": 100,
        "graph_type": "odescent",
        "graph_iter_turn": 10,
        "neighbor_sample_rate": 0.3,
        "alpha": 1.2
    }
}
```

Then just call `Build(data)` — no other API change. The complete program is
[`examples/cpp/312_feature_odescent.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/312_feature_odescent.cpp).

### ODescent build parameters

These keys go under `index_param` alongside the usual HGraph keys:

| Parameter | Default (HGraph) | Description |
|-----------|------------------|-------------|
| `graph_type` | `"nsw"` | Set to `"odescent"` to switch on this builder. |
| `graph_iter_turn` | `30` | Number of refinement iterations. Higher → better graph quality, longer build. |
| `neighbor_sample_rate` | `0.2` | Fraction of each node's neighbors sampled per iteration for candidate exchange. |
| `alpha` | `1.2` | α factor used by the diversity-aware edge pruning step. Larger `alpha` → sparser, more diverse edges. |
| `min_in_degree` | `1` | Minimum in-degree enforced when repairing the graph after pruning. |
| `build_block_size` | `10000` | Parallelization granularity (vectors per worker block). |

`max_degree` is inherited from the HGraph top-level setting; you do not need to repeat it under
ODescent. Upper graph layers automatically use half of `max_degree`.

### When to use ODescent vs NSW

- **Use ODescent** when you have the full dataset up front and care about build throughput on a
  many-core machine. The batch refinement parallelizes better than insertion-by-insertion.
- **Use NSW** (the default) when you build incrementally or care about strictly minimal memory
  during the build, or when you have not measured a build-time problem.

Both choices produce a graph that is searched the same way at query time, so search-side
parameters (`ef_search`, `pq_rerank`, …) carry over unchanged.

## See also

- [Creating an Index](../guide/create_index.md)
- [HGraph index parameters](../indexes/hgraph.md)
- [Pyramid index parameters](../indexes/pyramid.md)
- [Index Parameters reference](../resources/index_parameters.md)
