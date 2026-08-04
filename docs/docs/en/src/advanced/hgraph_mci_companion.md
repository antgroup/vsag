# HGraph MCI Companion

HGraph can optionally build an MCI (Maximal Clique Index) companion for filtered KNN
search. The companion stores clique metadata beside the HGraph index and shares HGraph's
vector storage. It is not a standalone index type: create the index as `hgraph`, not `mci`.

Use this feature when filtered search is the main workload and the filter keeps only a
small fraction of vectors. HGraph chooses between normal graph traversal and the MCI
companion by comparing `Filter::ValidRatio()` with a threshold.

## Build Configuration

MCI build parameters are flat fields in `index_param`. The companion is enabled when
`use_mci` is true or when any of `mci_mcs`, `mci_clique_max`, or `mci_alpha` is present.
Omit all four fields to disable it:

```json
{
  "dtype": "float32",
  "metric_type": "l2",
  "dim": 128,
  "index_param": {
    "base_quantization_type": "fp32",
    "max_degree": 32,
    "ef_construction": 400,
    "mci_mcs": 200,
    "mci_clique_max": 50,
    "mci_alpha": 1.2
  }
}
```

HGraph derives the KNN graph for clique construction from the built HGraph graph and vector data.

| Parameter | Purpose |
| --- | --- |
| `use_mci` | Enables MCI with default build parameters when set to `true`. |
| `mci_mcs` | Candidate neighbor count used when constructing cliques. |
| `mci_clique_max` | Maximum clique size during full build. |
| `mci_alpha` | Clique construction expansion factor. |
| `mci_incremental_join_ratio_threshold` | Add-time threshold for joining existing cliques. |
| `mci_incremental_added_mct` | Maximum existing cliques a newly added node may join. |
| `mci_incremental_clique_max` | Maximum clique size used by incremental clique creation. |

## Search Configuration

Search parameters live under the `hgraph` search object:

```json
{
    "hgraph": {
      "ef_search": 120,
      "use_mci": true,
      "mci_seed_ratio": 0.1,
      "hgraph_valid_ratio_threshold": 0.2
    }
}
```

`use_mci` defaults to true for search and can be set to false to disable MCI for a single
query. `hgraph_valid_ratio_threshold` is the search routing threshold: use MCI when
`ValidRatio()` is below this value; otherwise use HGraph. The default is `0.05`.

The seed count is
`ceil(sqrt(current_vector_count) * mci_seed_ratio)`, with a minimum of one seed.
`mci_seed_ratio` defaults to `0.1` and must be finite and non-negative. The resulting
seed count is capped at the number of points that satisfy the filter.

The companion needs a `Filter` object with a meaningful `ValidRatio()` hint. Bitset and
function filters are accepted, but a custom `Filter` gives the search planner better
selectivity information.

## Add, Serialize, and Stats

When MCI is enabled by the flat build parameters, `HGraph::Add()` updates the companion
after inserting each new node into HGraph. It first tries to join suitable existing cliques,
then creates a small
incremental clique when no good join target exists.

Note: MCI indexes should not be built from scratch by calling `Add()` on an empty index.
The incremental add path is intended for appending a small number of vectors to an existing
initial index.

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
