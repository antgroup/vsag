# HGraph MCI Companion

HGraph can optionally build an MCI (Maximal Clique Index) companion for filtered KNN
search. The companion stores clique metadata beside the HGraph index and shares HGraph's
vector storage. It is not a standalone index type: create the index as `hgraph`, not `mci`.

For implementation details, memory analysis, and 10k/3m results, see
[HGraph MCI Mutation Design and Benchmark Report](hgraph_mci_mutation.md).
For code entry points, configuration, and runnable commands, see the
[code and configuration guide](hgraph_mci_usage.md).

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

Both the float fast path and the generic full-build path share a final coverage pass.
Any node still uncovered after clique enumeration is placed in a fallback clique with graph
neighbors (or a singleton when none are available). This pass relaxes clique distance constraints,
preserves the seed under the clique-size cap, and counts only stored memberships.

| Parameter | Purpose |
| --- | --- |
| `use_mci` | Enables MCI with default build parameters when set to `true`. |
| `mci_mcs` | Candidate neighbor count used when constructing cliques. |
| `mci_knng_source` | KNN graph source: `hgraph` (default) or `odescent`. |
| `mci_clique_max` | Maximum clique size during full build. |
| `mci_alpha` | Clique construction expansion factor. |

### Incremental Maintenance Parameters

These parameters also belong in `index_param`. They control ADD and survivor repair after deletion.
Repair shares ADD's existing-clique join and new-clique construction routines; it does not
reinsert the existing vector into HGraph.

Degree is the size of the deduplicated union of other live members of all effective cliques
covering the current point. It is neither HGraph out-degree nor the point's clique-membership count.
The degree stopping target is:

```cpp
target = N <= 1 ? 0 :
    min(N - 1,
        max(mci_incremental_degree_min,
            min(N / mci_incremental_degree_n_divisor,
                mcs / mci_incremental_degree_mcs_divisor)));
```

`N` is the current live vector count, including the completed HGraph insertion batch but excluding
marked removals; `mcs` means `mci_mcs`. Integer division rounds down. This is a stopping target,
not a guaranteed minimum degree or a hard cap: joining a whole clique can overshoot it,
and exhausted candidates or stalled construction can stop below it.

| Parameter | Default and range | Meaning |
| --- | --- | --- |
| `mci_incremental_join_ratio_threshold` | Default `0.6`; range `[0, 1]`. | Overlap threshold for joining an existing clique: the number of its live members present in the point's KNN candidates, divided by its live member count. For a 10-member clique with 6 members in the candidates, the ratio is `0.6`: threshold `0.6` qualifies, whereas `0.7` does not. The clique must also have room and add new neighbors. This ratio does not revalidate distance constraints against every member. Lowering the threshold relaxes joining; raising it is stricter and may leave more degree deficits for new-clique construction. Compare `0.5 / 0.6 / 0.7` while measuring recall, ADD time, and total memberships. |
| `mci_incremental_degree_min` | Default `50`; positive integer. | Floor in the degree-target formula, still capped at `N-1`. At `N=10000, mcs=200`, the default target is `50`; setting the floor to `70` makes the target `70`. At `N≈3m, mcs=200`, both floors produce target `100`. If many points have low degree, try a higher floor and verify recall; a lower floor can reduce maintenance needed to reach the target but may reduce search connectivity. First check whether the floor actually determines the target. |
| `mci_incremental_degree_n_divisor` | Default `10000`; positive integer. | Controls the size-dependent term `N / divisor`. At `N=800000, mcs=200, degree_min=50`, the default target is `80`. Changing the divisor to `20000` makes this term `40`, so the floor sets the final target to `50`. A smaller divisor raises the size-dependent term; a larger divisor lowers it. Use it to adjust how the target grows with the dataset, noting that the floor or MCS term can make a change ineffective. |
| `mci_incremental_degree_mcs_divisor` | Default `2`; positive integer. | Controls `mcs / divisor`, which is combined with the size-dependent term before applying the floor. At `N≈3m, mcs=200, degree_min=50`, divisor `2` gives target `100`; divisor `4` gives target `50`. A smaller divisor can raise the target and a larger one can lower it, subject to the other terms. This parameter does not change the number of KNN candidates; change `mci_mcs` to change that candidate budget. |
| `mci_incremental_clique_max` | Default `50`; integer at least `2`. | Caps both newly constructed incremental cliques and the size after appending to an existing clique. With cap `50`, a 49-member clique can grow to 50, but a 50-member clique cannot accept another point. Full-build cliques already at size 50 therefore cannot be appended to under the defaults. New cliques can be smaller than the cap. Raising it allows some formerly full cliques to grow and permits larger new cliques; lowering it may require more cliques to fill the degree deficit. Evaluate it together with `mci_clique_max`, total memberships, and query cost. |
| `mci_delete_clique_size_threshold` | Default `30`; positive integer. | Retire a clique affected by the deletion batch only if its remaining live member count after the entire batch is strictly below the threshold: 29 is retired, 30 is retained by default. Retirement removes the clique and its memberships, not surviving vectors; unrelated small cliques are not scanned for retirement. This is not a construction size cap. A higher threshold expands retirement and may increase repair cost and structural change; a lower threshold retains more small cliques. Start at `30` and adjust in steps of `10`, comparing retired cliques, repair counts, deletion time, and recall rather than assuming higher is better. If a construction size cap is below this threshold, every affected clique of that size is retired; evaluate the settings together. |
| `mci_delete_node_mct_threshold` | Default `3`; positive integer. | Selects surviving members of retired cliques whose projected effective clique count is strictly below the threshold. Counts exclude all cliques retiring in the batch, and repair candidates are deduplicated. By default, projected counts 0, 1, and 2 qualify; 3 does not. This counts cliques, not unique neighbors. Raising the threshold expands the candidate set; lowering it narrows the set. Value `1` selects only candidates projected to lose all clique coverage. Compare `3 → 4 → 5 → 6` for repair cost and query quality. Coverage is checked again before repair, so selection does not guarantee creation of a new clique. |

These tuning directions follow the implementation; they are not validated optimal settings.
Keep dataset stages, mutation IDs, ground truth, and other settings fixed while changing one
parameter at a time. Compare quality, throughput, mutation time, and memory together.
For example, at `N≈3m, mcs=100, degree_min=50`, the default target is `50`; changing only the
live-count divisor leaves it unchanged whenever the size-dependent term remains at least 50.
Explicit settings override defaults, and loading an existing index requires matching its stored
parameters. Historical experiments using retirement threshold 3 do not measure the current default 30.

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

## Add, Delete, Serialize, and Stats

When MCI is enabled by the flat build parameters, `HGraph::Add()` first inserts the HGraph
batch, then updates MCI for each successfully inserted row. It first tries to join suitable
existing cliques, then creates incremental cliques if the unique neighbor degree is still too low.

Prefer `Build()` for the initial index, followed by incremental additions. Add on an empty
index can trigger construction, but many tiny Adds are not recommended as a replacement
for bulk Build.

When ADD needs a new clique, it shares the local graph, maximal-clique enumeration and selection
core with full `BuildMCICliques`. The incremental size cap determines the local size threshold;
it no longer accepts two members as the alpha-expansion stopping criterion. JOIN and construction
use the degree target defined by the incremental maintenance parameters above, stopping when
the target is reached, candidates are exhausted, or construction stalls.
High-alpha fallback may still emit smaller cliques.

`MARK_REMOVE` updates the MCI companion as part of the same operation. After removing a node, MCI
keeps every affected clique whose remaining live size is at least
`mci_delete_clique_size_threshold`. It retires only smaller cliques and collects only their live
members whose projected clique count is below `mci_delete_node_mct_threshold`. Those
under-covered nodes are repaired with the same incremental join/build routine used by `Add()`, so
the new memberships and cliques share Add's delta storage. This avoids rebuilding the full
one-hop neighborhood when most members still have sufficient clique coverage.
`MARK_REMOVE` remains the default: logical deletion does not reclaim vector slots.

Set `index_param.support_force_remove: true` to enable
`index->Remove(ids, vsag::RemoveMode::FORCE_REMOVE)`. MCI automatically enables reverse graph
edges and requires flat graph storage. Physical deletion remains incompatible with
`deduplicate_storage`, duplicate groups, and attribute-inverted storage. Indexes built without
force-remove support must be rebuilt to enable it.

Physical deletion repairs HGraph edges and fills removed slots with tail vectors, updating labels
and graph references. MCI rebuilds both CSR directions using the same node-ID mapping, preserves
unrelated soft-deletion markers, and repairs under-covered members of small affected cliques with
Add's incremental clique routine. It then flushes metadata and shrinks vector/graph storage.
Repeated IDs in one request count once; missing IDs count zero. Explicitly requested soft-deleted
IDs can also be physically removed. Subsequent Add starts at the compacted tail.

FORCE_REMOVE is serialized with Add, MARK_REMOVE, and Flush. Queries wait during ID moves and
final shrinking; repair releases the force-remove lock while MCI remains unpublished, so queries
may fall back to HGraph. This is not a transactional operation: deletions completed before an error may
remain, and an unfinished companion is not published to fast search. CSR replacement itself
commits only after allocation succeeds; temporary old/new buffers coexist during compaction.
Allocator retention and IO block granularity mean RSS need not fall proportionally to index
memory accounting.

Call `index->Flush()` to merge the companion's incremental metadata into its two CSR arrays.
Flush includes new delta cliques and members appended to base cliques, drops retired cliques and
deleted members, and rebuilds the node-to-clique mapping with compact clique IDs. It clears the
delta storage but preserves vector inner IDs and deletion markers. The operation is idempotent
and does not rebuild HGraph or persist data to disk; use `Serialize()` for persistence.
Other index types and HGraph without MCI return an unsupported-operation error.

Flush is serialized with Add/Remove. It holds an exclusive clique-storage lock while constructing
and publishing the replacement, so searches may wait. Replacement buffers are allocated before
publication; an allocation failure leaves the old clique representation intact. Temporary memory
includes both the old and new CSR. Do not retain clique IDs across a flush.

MCI search pins CSR, delta, and deletion masks with one shared lock per query and traverses them
without copying membership lists. Contiguous FP32 vectors continue to use direct distance
calculation after Add/Remove, before or after Flush; `mci_raw_float_csr` reports this path, including
queries that traverse delta. Other vector layouts use the same traversal and candidate queue with
their normal distance computer. Deletion filtering pins the deletion set once per MCI query,
checks the user filter first, and avoids acquiring the deletion-set lock for every visited node.

Add/MARK_REMOVE may temporarily unpublish the companion, so concurrent queries still fall back to
HGraph under the existing mutation semantics. This does not guarantee complete recall during a
mutation; fallback results can be empty when the graph entry point has been deleted. Flush alone
does not unpublish the companion; queries wait for the clique storage lock when necessary.

The clique data is serialized inside the HGraph index. Loading the HGraph index restores the
companion automatically.

`GetStats()` includes MCI quality fields such as:

- `mci_has_index`
- `mci_total_nodes`
- `mci_covered_nodes`
- `mci_total_clique_count`
- `mci_retired_clique_count`
- `mci_inactive_node_count`
- `mci_total_membership_count`
- `mci_avg_membership_per_node`
- `mci_avg_clique_size`
- `mci_max_clique_size`
- `mci_memory_usage`

## Example

See
[`examples/cpp/324_feature_hgraph_mci_companion.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/324_feature_hgraph_mci_companion.cpp)
for a minimal build and filtered-search flow.

## Add/Delete Performance Results

See the [historical mutation report](hgraph_mci_mutation.md) for methodology and results.
The benchmark and supporting scripts are not included in this PR.
