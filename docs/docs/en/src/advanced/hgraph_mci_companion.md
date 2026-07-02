# HGraph MCI Companion

HGraph can optionally build an MCI (Maximal Clique Index) companion for filtered KNN
search. The companion stores clique metadata beside the HGraph index and shares HGraph's
vector storage. It is not a standalone index type: create the index as `hgraph`, not `mci`.

Use this feature when filtered search is the main workload and the filter keeps only a
small fraction of vectors. HGraph chooses between normal graph traversal and the MCI
companion by comparing `Filter::ValidRatio()` with a threshold.

## Build Configuration

Enable the companion under `index_param.mci`:

```json
{
  "dtype": "float32",
  "metric_type": "l2",
  "dim": 128,
  "index_param": {
    "base_quantization_type": "fp32",
    "max_degree": 32,
    "ef_construction": 400,
    "alpha": 1.2,
    "mci": {
      "use_mci": true,
      "mcs": 200,
      "clique_max": 50,
      "alpha": 1.2,
      "hgraph_valid_ratio_threshold": 0.2,
      "knng_path": "/path/to/knng_200.bin"
    }
  }
}
```

`knng_path` is optional. If it is present, HGraph builds cliques from the supplied KNNG.
Otherwise it derives a KNN graph from the built HGraph graph and vector data.

| Parameter | Purpose |
| --- | --- |
| `use_mci` | Enables or disables the companion. |
| `mcs` | Candidate neighbor count used when constructing cliques. |
| `clique_max` | Maximum clique size during full build. |
| `alpha` | Clique construction expansion factor. |
| `hgraph_valid_ratio_threshold` | Default search routing threshold: use MCI when `ValidRatio()` is below this value; otherwise use HGraph. |
| `knng_path` | Optional binary KNNG file for clique construction. |
| `incremental_join_ratio_threshold` | Add-time threshold for joining existing cliques. |
| `incremental_added_mct` | Maximum existing cliques a newly added node may join. |
| `incremental_clique_max` | Maximum clique size used by incremental clique creation. |

## Search Configuration

Search parameters live under the `hgraph` search object:

```json
{
    "hgraph": {
      "ef_search": 120,
      "use_mci": true,
      "seed_count": 64,
      "seed_ratio": 0.002,
      "hgraph_valid_ratio_threshold": 0.2
    }
}
```

`seed_ratio` multiplies the current vector count to choose the seed count. If both
`seed_count` and `seed_ratio` are provided, `seed_ratio` takes precedence.

The companion needs a `Filter` object with a meaningful `ValidRatio()` hint. Bitset and
function filters are accepted, but a custom `Filter` gives the search planner better
selectivity information.

## Add, Serialize, and Stats

When `use_mci` is enabled, `HGraph::Add()` updates the companion after inserting each new
node into HGraph. It first tries to join suitable existing cliques, then creates a small
incremental clique when no good join target exists.

The clique data is serialized inside the HGraph index. Loading the HGraph index restores the
companion automatically.

`GetStats()` includes MCI quality fields such as:

- `mci_has_index`
- `mci_total_nodes`
- `mci_covered_nodes`
- `mci_total_clique_count`
- `mci_total_membership_count`
- `mci_avg_membership_per_node`
- `mci_avg_clique_size`
- `mci_max_clique_size`
- `mci_memory_usage`

## Example

See
[`examples/cpp/324_feature_hgraph_mci_companion.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/324_feature_hgraph_mci_companion.cpp)
for a minimal build and filtered-search flow.
