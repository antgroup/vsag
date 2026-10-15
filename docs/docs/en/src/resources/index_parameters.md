# Index Parameters

This page summarises the commonly used parameters for every VSAG index type. For the full
enumeration, consult the source:

- Build parameter keys: `src/constants.cpp`
- Public constants: `include/vsag/constants.h`
- Per-index examples: the `examples/cpp/*_index_*.cpp` files (e.g. `103_index_hgraph.cpp`).

## Common Fields

Every index requires these top-level fields at build time:

| Field | Values | Description |
|-------|--------|-------------|
| `dim` | positive integer | Vector dimensionality; cannot change after build |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` | Vector data type; determines internal representation |
| `metric_type` | `l2` / `ip` / `cosine` | Distance metric |

## HGraph

HGraph places its build parameters under the generic `index_param` key (see
`examples/cpp/103_index_hgraph.cpp`); the `hgraph` key is reserved for search-time parameters.

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 400
    }
}
```

| Field | Typical | Description |
|-------|---------|-------------|
| `max_degree` | 16–48 | Maximum out-degree per node |
| `ef_construction` | 200–500 | Candidate set size during build; larger = higher recall, slower build |
| `alpha` | `1.0` | Final robust-pruning factor; PiPNN requires a finite value at least `1.0` |
| `base_quantization_type` | `fp32` / `fp16` / `bf16` / `sq8` / `sq4` / `pq` | Quantization of the base storage — see the [Quantization chapter](../quantization/README.md) for all supported values |

The optional `mci_knng_source` build parameter enables the MCI companion and selects
`hgraph` (default) or `odescent` for its candidate graph. See the
[MCI companion guide](../advanced/hgraph_mci_companion.md) for construction and filter requirements.

MCI maintenance uses `mci_incremental_degree_min`, `mci_incremental_degree_n_divisor`,
`mci_incremental_degree_mcs_divisor`, `mci_delete_clique_size_threshold`, and
`mci_delete_node_mct_threshold`. Physical removal requires `support_force_remove: true`;
see the [maintenance guide](../advanced/hgraph_mci_companion.md) for defaults and restrictions.

The opt-in `use_hybrid_traversal` search mode combines sparse and clique neighbors, with
`hybrid_vob`, `hybrid_filter_cost_ratio`, `mci_seed_coverage`, and `mci_seed_max_count`
controlling per-expansion work and seeds. See the
[MCI companion guide](../advanced/hgraph_mci_companion.md#dynamic-neighbor-traversal).

### PiPNN build parameters

When `graph_type` is `pipnn` on HGraph, the following build parameters are also accepted under
`index_param`. PiPNN is available for HGraph only; the Pyramid index does not support
`graph_type: "pipnn"`.

| Field | Default | Description |
|-------|---------|-------------|
| `pipnn_max_leaf_size` | `1024` | Maximum partition leaf size, in `[2, 2147483647]`. |
| `pipnn_min_leaf_size` | `64` | Target for merging undersized leaves; must be positive and no larger than `pipnn_max_leaf_size`. |
| `pipnn_leader_sample_rate` | `0.005` | Fraction sampled as leaders, in `(0, 1]`; each partition uses `2` to `1000`, bounded by its point count. |
| `pipnn_fanout` | `[10, 2]` | Positive nearest-leader fanout at successive partition levels; deeper levels use `1`. |
| `pipnn_leaf_neighbor_count` | `5` | Positive nearest-candidate count per point and leaf; undersized leaves use at least `4`. |
| `pipnn_hash_plane_count` | `12` | Direction-hash bits in `[1, 15]`; `max_degree <= 2^pipnn_hash_plane_count`. |
| `pipnn_reservoir_size` | `64` | Positive per-point candidate capacity before final pruning; effective capacity is `max(value, max_degree)` and must not exceed `65535`. |

The existing `alpha` field controls PiPNN's final robust pruning. `ef_construction` only tunes NSW
and has no effect on PiPNN.

HGraph additionally supports the experimental L2-only options `pipnn_adaptive_pruning` (default `false`) and `pipnn_adaptive_pruning_adjust_step` (default `0.06`). They adapt the final per-point pruning threshold during initial `Build`; see [HGraph adaptive PiPNN pruning](../indexes/hgraph.md). These options are not exposed by Pyramid.

At search time:

```json
{"hgraph": {"ef_search": 100}}
```

The `hgraph` search-param object also accepts `brute_force_threshold` (a float
in `[0.0, 1.0]`, default `0.0`). When set above zero and the request carries a
filter whose `ValidRatio()` is at most this threshold, HGraph skips the graph
traversal and runs an exact scan over the surviving ids. See the
[HGraph index page](../indexes/hgraph.md#brute-force-fallback-under-highly-selective-filters-brute_force_threshold)
for details.

## LazyHGraph

LazyHGraph can take its build parameters in a top-level `lazy_hgraph` object
(preferred for clarity) or in the generic `index_param` object. The `hgraph`
sub-object is forwarded to the internal HGraph used after transition.

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "lazy_hgraph": {
        "transition_threshold": 1000,
        "hgraph": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100
        }
    }
}
```

| Field | Typical | Description |
|-------|---------|-------------|
| `transition_threshold` | `1000` or workload-specific | Positive vector count at which the index converts from exact flat search to HGraph |
| `hgraph` | HGraph build object | Parameters for the graph phase; see [HGraph](../indexes/hgraph.md) |

LazyHGraph only supports `dtype: "float32"`. Search parameters use the `hgraph`
object, for example `{"hgraph": {"ef_search": 100}}`. See the
[LazyHGraph index page](../indexes/lazy_hgraph.md) for details.

The `hgraph` search-param object also accepts the following filter-related parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_ratio` | float | `0.2` | Controls the ratio of filtered-search candidate checks to skip, in range `[0.0, 1.0]`. Higher values mean more aggressive skipping, faster search, and potentially lower recall. |
| `skip_strategy` | string | `"deterministic_accumulative"` | Skip strategy. Supports `"random"` and `"deterministic_accumulative"`. |

## IVF

```json
{
    "ivf": {
        "nlist": 4096,
        "base_quantization_type": "sq8",
        "nprobe": 32
    }
}
```

## Brute Force

```json
{"brute_force": {}}
```

No extra parameters.

## Pyramid

Pyramid supports organising multiple subgraphs by tag:

```json
{
    "pyramid": {
        "tag_dim": 1,
        "max_degree": 24,
        "ef_construction": 300
    }
}
```

`graph_type: "pipnn"` is not supported by Pyramid; the PiPNN build parameters above apply to
HGraph only.

## SINDI (sparse vectors)

```json
{
    "sindi": {
        "top_k": 32,
        "doc_prune_ratio": 0.1
    }
}
```

## Runtime Parameters

Beyond build-time parameters, `Index::Tune` and `SearchParam` tweak runtime settings such as
`ef_search` and `nprobe`. See [Optimizer](../advanced/optimizer.md) and the
`examples/cpp/3xx_feature_*.cpp` examples.
