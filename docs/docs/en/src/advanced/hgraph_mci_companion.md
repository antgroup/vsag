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
| `mci_clique_max` | Minimum size of a maximal clique the full build must keep. Cliques are stored in full, so this is a lower bound, not a cap. |
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

`mci_seed_coverage` raises that budget to `ceil(mci_seed_coverage * valid_count)` whenever the
target fits into `mci_seed_max_count` and does not exceed the vector count; otherwise the
coverage term is dropped entirely rather than truncated. `mci_seed_coverage` defaults to `1.0`
and `mci_seed_max_count` to `32768` (`0` means unlimited). The budget never drops below one, so
seeding cannot be disabled completely -- set both terms to `0` to leave a single seed.

Because that term is dropped rather than clamped, `mci_seed_coverage` has no effect once the
valid set exceeds `mci_seed_max_count`: for those (wide-filter) queries the budget silently
falls back to the `ceil(sqrt(N) * mci_seed_ratio)` floor. Raising the cap is what extends the
exact-seeding regime, at the cost of a larger seed phase. The default of `32768` keeps that
enumeration cheap in absolute terms -- 32768 inner ids are 128 KiB and therefore stay in cache --
so the saturated skip stays a win instead of turning a wide predicate into a scan-sized seed
phase.

The companion needs a `Filter` object with a meaningful `ValidRatio()` hint. Bitset and
function filters are accepted, but a custom `Filter` gives the search planner better
selectivity information.

## Dynamic Neighbor Traversal

`use_hybrid_traversal` (default `false`) replaces the either/or routing above with a single
traversal that handles both neighbor sources of every expanded vector in one pass:

- the sparse HGraph neighbors are scored first and pushed into the candidate heap, so the
  search subgraph stays connected regardless of selectivity;
- the clique members of the same vector are scored predicate-first, and that part is cut short
  by a virtual-overhead budget.

The budget stops the clique part once
`considered * (hybrid_filter_cost_ratio + local_selectivity)` reaches `hybrid_vob`, where
`hybrid_filter_cost_ratio` is `O_filter / O_dist` (the cost of one predicate filtering relative
to one distance computation) and `local_selectivity` is the running predicate hit rate of the
current neighbor traversal. A non-positive `hybrid_vob` disables the early stop. The two
parameters default to `0.0` and `1.0`, and `hybrid_vob` is expressed in units of `O_dist`.

```json
{
    "hgraph": {
      "ef_search": 600,
      "use_hybrid_traversal": true,
      "hybrid_vob": 32.0,
      "mci_seed_ratio": 1.0,
      "mci_seed_coverage": 1.0
    }
}
```

The traversal is seeded from the predicate's valid set using the same budget and samplers as
the MCI route, so the two can be compared without the seed strategy confounding the result.
When that budget ends up covering every valid point, all valid distances are already known and
the traversal answers from the seeds directly; the expansion is skipped because it is provably
redundant, not as an approximation. `GetStatistics()` reports the behavior through
`mci_hybrid_route` (`"hybrid"` when the traversal ran), `hybrid_seed_budget`,
`hybrid_seeded_entries`, `hybrid_expansion_skipped`, `hybrid_expanded_nodes`,
`hybrid_mci_members_considered`, `hybrid_dist_computations` and `hybrid_mci_stopped_early`.

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
