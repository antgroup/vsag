# Incremental Add for MCI

MCI can maintain its clique index while new vectors are appended with `Add()`. This is useful
when a large index is built from an initial snapshot and then receives a batch of new vectors
without rebuilding the full maximal-clique structure from scratch.

The incremental path keeps the original CSR clique index immutable and records newly created
or extended clique memberships in a lazy delta overlay. Search reads both the base CSR data and
the delta overlay, so newly added vectors participate in subsequent KNN and filtered KNN queries.

## Basic Workflow

1. Create an MCI index with the same parameters you would use for full build.
2. Call `Build()` on the initial prefix.
3. Call `Add()` with the remaining vectors, either one at a time or in batches.
4. Search normally.
5. Serialize the index if you need to persist the updated state.

```cpp
auto index = vsag::Factory::CreateIndex("mci", create_params).value();

auto build_result = index->Build(prefix_dataset);
if (not build_result.has_value()) {
    // handle build_result.error()
}

auto add_result = index->Add(tail_dataset);
if (not add_result.has_value()) {
    // handle add_result.error()
}
```

A runnable example is available at
`examples/cpp/323_feature_mci_incremental_add.cpp`.

## How New Vectors Join Cliques

For each new vector `A`, MCI first finds `KNN(A)` using the embedded HGraph sub-index when
available, or MCI's own search path otherwise. It then checks existing cliques that cover at
least one neighbor from `KNN(A)`.

For a candidate clique `C`, MCI computes:

```text
overlap(C, A) = |C intersect KNN(A)| / |C|
```

If the overlap is greater than or equal to `join_ratio_threshold`, the clique becomes eligible
for extension. When more than one clique is eligible, MCI sorts candidates by
`|C intersect KNN(A)|` in descending order and adds `A` to at most `added_mct` cliques.

If no existing clique qualifies, MCI builds a new local clique around `A` from its nearest
neighbors. This local builder starts with `alpha`, doubles the distance threshold when necessary,
and stops once it reaches the target local clique size or the alpha cap.

This maintenance strategy is approximate. It is designed to preserve search quality and avoid
expensive CSR middle inserts during online growth; it does not re-enumerate the global maximal
clique set after every append.

## Incremental Parameters

These parameters live under `index_param`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `join_ratio_threshold` | float | `0.6` | Minimum `|C intersect KNN(A)| / |C|` required before a new vector can join an existing clique. |
| `added_mct` | int | `3` | Maximum number of eligible existing cliques a new vector joins. Candidates are selected by largest intersection with `KNN(A)`. |

Example:

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "base_codes_type": "flatten",
        "max_degree": 32,
        "mcs": 200,
        "clique_max": 20,
        "alpha": 1.2,
        "join_ratio_threshold": 0.6,
        "added_mct": 3
    }
}
```

Lowering `join_ratio_threshold` lets new vectors attach to more existing cliques, which can
improve coverage but may increase the delta overlay. Increasing it makes joins stricter and may
create more new local cliques. Raising `added_mct` gives each new vector more memberships when
several existing cliques qualify, again trading memory for coverage.

## HGraph Hybrid Persistence

When `use_hgraph_hybrid` is enabled, MCI keeps an embedded HGraph sub-index for routing broad
filtered searches and for finding neighbors during incremental add. MCI serialization stores
that embedded HGraph payload together with the MCI data, so a reloaded index can continue to use
the HGraph route without requiring a separate companion file.

You can still use `hgraph_index_path` as an external fallback for older indexes or workflows
that keep the HGraph index as a separate artifact. On deserialization, MCI first restores the
embedded HGraph payload if present; it loads `hgraph_index_path` only when the embedded HGraph is
missing or does not match the MCI element count.

## Evaluating Incremental Build

The standalone evaluation tool `tools/eval/mci_incremental_exp.cpp` compares:

- building on a prefix and then adding the remaining vectors
- optionally, a full-build baseline over the same final vector count

Example command:

```bash
cmake --build build-release --target mci_incremental_exp -j$(nproc)

CREATE_PARAMS='{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"sq8","base_codes_type":"flatten","max_degree":32,"mcs":200,"clique_max":20,"alpha":1.2,"join_ratio_threshold":0.6,"added_mct":3,"build_thread_count":32,"use_hgraph_hybrid":true,"hgraph_valid_ratio_threshold":0.05,"hgraph_ef_search":50,"hgraph_index_param":{"base_quantization_type":"fp32","graph_type":"odescent","max_degree":32,"alpha":1.2,"graph_iter_turn":20,"neighbor_sample_rate":0.2}}}'
SEARCH_PARAMS='{"mci":{"ef_search":20,"seed_count":3600},"hgraph":{"ef_search":50}}'

./build-release/tools/eval/mci_incremental_exp \
  -d /path/to/filter-dataset.h5 \
  -c "$CREATE_PARAMS" \
  -s "$SEARCH_PARAMS" \
  --topk 20 \
  --build_ratio 0.5 \
  --add_batch 10000 \
  --query_count 10000 \
  --skip_full \
  --save_index ./mci_incremental.index
```

Use a small `--base_limit` and `--add_limit` first when trying a new dataset. The incremental
path is much faster than rebuilding cliques from scratch, but high-dimensional datasets can still
make neighbor search and HGraph maintenance expensive.
