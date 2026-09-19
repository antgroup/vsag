# HGraph MCI Companion

HGraph can optionally build an MCI (Maximal Clique Index) companion for filtered KNN
search. The companion stores clique metadata beside the HGraph index and shares HGraph's
vector storage. It is not a standalone index type: create the index as `hgraph`, not `mci`.

Use this feature when filtered search is the main workload and the filter keeps only a
small fraction of vectors. HGraph chooses between normal graph traversal and the MCI
companion by comparing `Filter::ValidRatio()` with a threshold.

## Build Configuration

MCI build parameters are flat fields in `index_param`. The companion is enabled when
`use_mci` is true or when any MCI build parameter is present. The `mci_knng_source`
parameter selects whether clique construction derives its KNN graph from the completed
HGraph bottom graph or builds a dedicated ODescent graph:

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
    "mci_alpha": 1.2,
    "mci_knng_source": "odescent"
  }
}
```

The default source is `hgraph`, which preserves the existing behavior. Set the source to
`odescent` to build the MCI KNN graph directly from the stored vectors. An internally configured
external KNN graph path takes precedence over this selector.

| Parameter | Purpose |
| --- | --- |
| `use_mci` | Enables MCI with default build parameters when set to `true`. |
| `mci_mcs` | Candidate neighbor count used when constructing cliques. |
| `mci_knng_source` | KNN graph source: `hgraph` (default) or `odescent`. |
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

## Search Traversal and Early Stop

A filtered MCI search fills its candidate queue by scoring seed vectors, then expands the cliques of
the candidates it collected:

1. seeds are sampled from the valid labels, controlled by `mci_seed_ratio`;
2. for every candidate that has not been expanded yet, each of its cliques is walked and every
   member that is valid and unvisited is scored;
3. the traversal ends when the candidate queue has no unexpanded entry left.

Step 2 is what makes MCI useful for selective filters: one clique reaches a whole neighbourhood at
once. It is also where the traversal can waste work, because cliques overlap heavily. On a 5M-vector
quantized index (`mci_mcs=200`, `mci_clique_max=50`) a vector belongs to 22.4 cliques and an average
clique has 102 members, so once the reachable neighbourhood has been scored the remaining expansions
only walk members that are already visited. On that index the traversal expanded 11k-27k cliques per
query while scoring only 0.68 points per expanded clique.

MCI therefore stops expanding after 32 consecutive candidates that discovered nothing new (neither a
new candidate nor a new result). Same index, `ef_search=600`, `mci_seed_ratio=10`:

| | recall@100 | mean latency | cliques expanded per query |
| --- | --- | --- | --- |
| without the early stop | 0.9869 | 66 ms | 11k-27k |
| with the early stop | 0.9856 | 31 ms | 0.7k-2.0k |

The saving is largest for the queries whose seeds already cover every valid vector: there the
recall is unchanged (bit-identical) and the latency drops 4-11x. Queries whose valid set is larger
than the seed budget keep expanding, and they are the ones behind the small recall difference above.

The early stop bounds work, not recall: the traversal still has no lower-bound based termination, so
`mci_seed_ratio` and `ef_search` remain the knobs that decide how much of the valid set is explored.

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
