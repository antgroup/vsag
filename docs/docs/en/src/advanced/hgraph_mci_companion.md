# HGraph MCI Companion

HGraph can optionally build an MCI (Maximal Clique Index) companion for filtered KNN
search. The companion stores clique metadata beside the HGraph index and shares HGraph's
vector storage. It is not a standalone index type: create the index as `hgraph`, not `mci`.

Use this feature when filtered search is the main workload and the filter keeps only a
small fraction of vectors. HGraph chooses between normal graph traversal and the MCI
companion by comparing `Filter::ValidRatio()` with a threshold.

## Build Configuration

MCI build parameters are flat fields in `index_param`. The companion is enabled when
`use_mci` is true or when `mci_mcs`, `mci_clique_max`, `mci_alpha`, or
`mci_knng_source` is present. The `mci_knng_source` parameter selects whether clique
construction derives its KNN graph from the completed
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

`use_mci` defaults to true for search and can be set to false to disable the original MCI
route for a single query. The hybrid traversal below has its own enable flag.
`hgraph_valid_ratio_threshold` is the search routing threshold: use MCI when
`ValidRatio()` is below this value; otherwise use HGraph. The default is `0.05`.

The seed count is
`ceil(sqrt(current_vector_count) * mci_seed_ratio)`, with a minimum of one seed.
`mci_seed_ratio` defaults to `0.1` and must be finite and non-negative. The resulting
seed count is capped at the number of points that satisfy the filter.

The MCI route requires a bitset filter or a custom `Filter` that supplies a nonempty
valid-ID list through `GetValidIds()`, plus a meaningful `ValidRatio()` hint. Function-only
filters fall back to HGraph. Sampling resolves the selected labels without filling gaps
from stale or duplicate IDs; if no seeds remain, search falls back to HGraph.

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
current neighbor traversal. A zero `hybrid_vob` disables the early stop. Both cost parameters must be
finite and non-negative. The two
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

The traversal samples the predicate's valid set using the MCI samplers. Its seed budget is
`max(1, ceil(sqrt(N) * mci_seed_ratio))`, raised to `ceil(mci_seed_coverage * valid_count)`
only when that coverage target does not exceed N and fits `mci_seed_max_count`. A target
that misses the cap is dropped entirely, not truncated. Coverage defaults to `1.0` and
the cap to `32768` (`0` means unlimited); both must be non-negative and coverage finite.
Setting both ratio and coverage to zero leaves one seed. These two new parameters affect
only hybrid traversal; the existing MCI route retains its original seed budget.
When that budget ends up covering every valid point, all valid distances are already known and
the traversal answers from the seeds directly; the expansion is skipped because it is provably
redundant, not as an approximation. `GetStatistics()` reports the behavior through
`mci_hybrid_route` (`"hybrid"` when the traversal ran), `hybrid_seed_budget`,
`hybrid_seeded_entries`, `hybrid_expansion_skipped`, `hybrid_expanded_nodes`,
`hybrid_mci_members_considered`, `hybrid_dist_computations` and `hybrid_mci_stopped_early`.

The bitset sampler cannot prove exhaustiveness, so it never enables the expansion skip.
Filters without a usable seed list start at the coarse-search entry point. Hybrid traversal
is independent of the search-time `use_mci` routing flag. It falls back to existing routes
when no companion is published for the current count, for range search, or when an
attribute-filter executor is present. An enabled brute-force threshold takes precedence.
This `1.0` backport retains single-query search; it does not add main's batch-search API.
The exhaustive skip is exact for the distances used during traversal; quantized codes
still have their usual approximation and reranking behavior.

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

In this `1.0` foundation backport, reloading an index grown by incremental `Add()`
falls back to HGraph traversal because the restored companion retains its original
node count. Vector data remains searchable. Restoring incremental MCI availability
requires the separate maintenance backport from #3012. Physical removal is also
unsupported by this foundation.

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

## Optional filter bitmap

A custom filter can expose `GetValidBitmap(uint64_t* size)` with one byte per label,
where zero rejects a label and any nonzero value accepts it. Keep the bitmap alive
and unchanged throughout the search, and make it agree with `CheckValid()`.
MCI uses this optimization only when all live labels equal their inner IDs and
the bitmap covers the index. Otherwise it uses `CheckValid()`. Existing filters
need no changes. The search statistic `mci_bitmap_fast_path` reports whether the
bitmap was used. This applies to both float and quantized MCI searches.
