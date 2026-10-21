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

Both the float fast path and the generic full-build path share a final coverage pass.
Any node still uncovered after clique enumeration is placed in a fallback clique with graph
neighbors (or a singleton when none are available). This pass relaxes clique distance constraints,
preserves the seed under the clique-size cap, and counts only stored memberships.

| Parameter | Purpose |
| --- | --- |
| `use_mci` | Enables MCI with default build parameters when set to `true`. |
| `mci_mcs` | Candidate neighbor count used when constructing cliques. |
| `mci_knng_source` | KNN graph source: `hgraph` (default) or `odescent`. |
| `mci_clique_max` | Minimum size of a maximal clique the full build must keep. Cliques are stored in full, so this is a lower bound, not a cap. |
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

Migration warning: `mci_incremental_added_mct` is no longer supported. Remove this old
clique-count setting from `index_param` and configure `mci_incremental_degree_min`,
`mci_incremental_degree_n_divisor`, and `mci_incremental_degree_mcs_divisor` instead.
The old key is ignored, not translated; leaving it in the JSON applies the new defaults
unless the degree parameters are explicitly set. There is no equivalent one-to-one mapping
because overlapping cliques contribute different numbers of unique neighbors. Recheck recall
and mutation cost, and rebuild old snapshots with the new configuration rather than assuming
that a removed parameter provides snapshot compatibility.

The default MCS divisor `2` limits the adaptive term to half the KNN candidate budget before
the floor is applied; it does not cap the final target at half a clique size. MCS defaults to
`200` and is distinct from `mci_clique_max`. If `mcs=32` and the floor is `50`, the target is
`min(N-1, 50)` for every N, not `max(50, N/10000)`. This is a tuning heuristic, not a guarantee
that half the candidates become neighbors; the floor can dominate and a whole-clique join
can exceed the target.

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
      "search_no_lock": false,
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
`search_no_lock` is a search-only boolean, default `false`, replacing
`mci_seed_no_lock` (the old name is rejected). It selects empty read-lock guards
throughout HGraph and MCI search, including routing, labels/deletions, vectors,
cliques and result handling. The visited-list pool uses a separate atomic lease
list; it does not access shared deques without their mutexes. Search algorithms,
filtering and missing/deleted-ID semantics are unchanged.

Use this option only with a read-only in-memory index: no concurrent build, add,
remove, vector/label update, tuning, resize, or deserialization. Concurrent independent
queries (including a mixture of locked and unlocked queries) are supported.
`hgraph.parallelism` must be 1, because the intra-query worker pool requires
its own synchronization; parallelize independent queries instead. This flag does not
disable synchronization inside the allocator, OS, external filters or disk caches.
The query-local policy is restored on return or exception and never changes the index's
global mutex objects. Query statistics report `search_no_lock`. Existing serialized
indexes can be reused without rebuilding or rewriting them.

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

## Code Map and Maintenance

Paths below are relative to the repository root.

| File | Entry points / responsibility |
| --- | --- |
| `include/vsag/index.h`, `src/index/index_impl.h` | Public Build/Add/Remove/Search and error wrapping |
| `src/algorithm/hgraph/hgraph_build.cpp` | `Add` → `add_impl`: insert into HGraph, then maintain MCI for successful insertions |
| `src/algorithm/hgraph/hgraph_mci.cpp` | Full construction, `search_mci_knn`, `incremental_update_mci_clique`, `repair_mci_clique`, `force_remove_with_mci`, `maybe_compact_mci` |
| `src/algorithm/mci/mci_local_builder.h` | Shared local graph construction, maximal-clique enumeration/selection and alpha policy for Build and incremental construction |
| `src/algorithm/hgraph/hgraph_modify.cpp` | Removal dispatch, graph repair, tail-slot moves, shrinking |
| `src/datacell/clique_datacell.{h,cpp}` | Bidirectional CSR, delta, deletion snapshots, retirement, remapping, Flush |
| `src/impl/searcher/mci_searcher.cpp` | `search_clique_view`: base CSR + delta + deletion markers |
| `src/impl/label_table/label_table.{h,cpp}` | External-label mapping and a deletion-set read view pinned once per query |
| `src/algorithm/hgraph/hgraph_parameter.{h,cpp}`, `hgraph_param_mapping.cpp` | Defaults, validation, external JSON mapping |
| `src/algorithm/hgraph/hgraph_serialize.cpp` | Format version and restoration |
| `src/analyzer/hgraph_analyzer.cpp` | Coverage, clique sizes, memberships, memory |

The base has `clique → nodes` and `node → cliques` CSR. ADD and deletion repair share:

| Field | Contents |
| --- | --- |
| `delta_cliques_` | Complete membership of new cliques |
| `delta_clique_extra_` | Members appended to base cliques |
| `delta_node_to_cids_` | Incremental node-to-clique relationships |

Node-deletion and clique-retirement markers are separate. Flush merges delta into CSR,
preserves vector inner IDs, and may renumber cliques. FORCE_REMOVE also moves vector inner IDs.

Shared ADD/repair pipeline:

1. Insert an ADD batch into HGraph, then maintain the successful points: each chunk is planned
   in parallel and merged in inner-ID order by a single thread, and every row still follows the
   rules in steps 2-3. `search_mci_knn` calls HGraph `KnnSearch` with `use_mci=false`, not MCI search.
2. Target `min(mci_mcs, visible_total - 1)` neighbors. Internal ef is `max(query_k, 100)`,
   not benchmark ef=320. Remove self, deleted, and out-of-range points; enlarge the request if needed.
3. Try existing cliques with `|KNN ∩ C| / |C| >= join_ratio` and room below the incremental cap,
   stopping at the unique-neighbor degree target rather than a clique-count limit. If degree remains
   below target, invoke the same local clique builder as full Build on not-yet-adjacent KNN candidates,
   repeating until the target is reached, candidates run out, or progress stops. Empty candidates
   yield a singleton. Incremental construction no longer stops alpha expansion at two members.
4. Deletion retires only affected cliques whose surviving size is **less than** `delete_size`.
   Select surviving members whose projected effective coverage is **less than** `delete_mct`;
   recheck coverage before repairing.
5. Pure-FP32 repair reads/decodes the existing point and uses the shared candidate/join/build path.
   It does not insert the vector again through public ADD. ADD sees its insertion prefix;
   repair can see the whole current HGraph.

The ADD clique-count limit has been removed; `delete_mct` still triggers repair.
The degree target is `max(mci_incremental_degree_min, min(N/10000, mcs/2))`, with zero for N≤1 and otherwise
capped at N-1; integer division rounds down. N counts live vectors including the completed graph
insertion batch. The floor defaults to 50, so at N=10000 and mcs=200 the target is 50.
Whole-clique joins may overshoot;
candidate exhaustion or lack of progress can stop below target. Non-FP32 repair retains pair-distance
candidate generation; FP32 findings do not establish RaBitQ behavior.

| Operation | Vector storage | MCI work | Persistence |
| --- | --- | --- | --- |
| MARK_REMOVE, default | Retain physical slots | Retire small cliques, repair under-covered points into delta | None |
| FORCE_REMOVE | Move tail points into holes; reduce slots and attempt shrinking | Snapshot, remap, repair, automatic Flush | None |
| Internal automatic compaction | No vector deletion/movement | Merge delta, remove retired memberships, rebuild CSR | None |

Add and MARK_REMOVE share a 100-successful-vector counter: Add checks per point; deletion checks
after the complete batch is projected and repaired. Failed inserts and duplicate/missing removals
do not count. A successful compaction, full rebuild, load, or FORCE_REMOVE end-of-batch compaction
resets the counter. Allocation failure during automatic maintenance retains delta and retries on
the next successful mutation. Delta remains searchable and serializable below the threshold.

MCI mutations share a serialization mutex. Physical moves and final shrinking hold exclusive
force-remove protection. Repair releases that lock so internal HGraph queries can acquire read
locks; MCI is unpublished and external searches may fall back to HGraph. Queries are not
necessarily blocked for the entire operation. Mutations are not transactionally rolled back.

## Benchmark Profile and C++ Integration

Pass this JSON string to `Factory::CreateIndex("hgraph", config_json)`. MCI build keys are flat
under `index_param`, not in a nested `mci` object. This is the Codefilter benchmark profile,
not the defaults for every library option.

```json
{
  "dtype": "float32",
  "metric_type": "cosine",
  "dim": 384,
  "index_param": {
    "base_quantization_type": "fp32",
    "base_io_type": "memory_io",
    "graph_type": "nsw",
    "max_degree": 32,
    "ef_construction": 200,
    "build_thread_count": 16,
    "support_force_remove": true,
    "use_mci": true,
    "mci_knng_source": "hgraph",
    "mci_mcs": 200,
    "mci_clique_max": 50,
    "mci_alpha": 1.2,
    "mci_incremental_join_ratio_threshold": 0.6,
    "mci_incremental_degree_min": 50,
    "mci_incremental_degree_n_divisor": 10000,
    "mci_incremental_degree_mcs_divisor": 2,
    "mci_incremental_clique_max": 50,
    "mci_delete_clique_size_threshold": 30,
    "mci_delete_node_mct_threshold": 3
  }
}
```

Enable `support_force_remove` at creation. MCI physical removal requires flat graph storage,
enables reverse edges, and rejects incompatible deduplication, duplicate-group, or attribute storage.

| Parameter | Library default | Benchmark CLI / meaning |
| --- | ---: | --- |
| `mci_mcs` | 200 | `--mci-mcs`; benchmark default 50, candidate count |
| `mci_clique_max` | 50 | `--mci-clique-max`; full-build clique cap |
| `mci_alpha` | 1.2 | `--mci-alpha`; expansion coefficient |
| `mci_incremental_join_ratio_threshold` | 0.6 | `--mci-incremental-join-ratio-threshold`; [0,1] |
| `mci_incremental_degree_min` | 50 | Positive degree-target floor; serialized with the index |
| `mci_incremental_degree_n_divisor` | 10000 | Positive live-count divisor for the degree target |
| `mci_incremental_degree_mcs_divisor` | 2 | Positive MCS divisor for the degree target |
| `mci_incremental_clique_max` | 50 | `--mci-incremental-clique-max`; at least 2; benchmark inherits the full-build cap when omitted |
| `mci_delete_clique_size_threshold` | 30 | Positive; retire affected cliques with fewer than 30 live members by default, retain those with exactly 30 |
| `mci_delete_node_mct_threshold` | 3 | `--mci-delete-node-mct-threshold`; positive, strict less-than repair |

For example, `delete_size=4` considers affected cliques with 0–3 survivors, not all cliques
containing a deleted point. Increasing `delete_mct` alone may do nothing if no small clique retires.

Search JSON uses `hgraph`, not `index_param`:

```json
{
  "hgraph": {
    "ef_search": 320,
    "use_mci": true,
    "mci_seed_ratio": 0.1,
    "hgraph_valid_ratio_threshold": 1.0
  }
}
```

The library's route threshold defaults to **0.05**, while this profile and benchmark use **1.0**.
Filtered queries with selectivity below the threshold may use MCI; routing is not guaranteed.
Supply an actual filter with accurate `ValidRatio()`, not an artificially low estimate.
Set query `use_mci=false` for an ordinary-HGraph comparison on the same index.

The following integration fragment assumes JSON strings above, caller-prepared `base`, `added`,
and `query` Datasets, a `filter`, and an external-label vector `removed_labels`.
Include VSAG and standard exception headers. Keep non-owned Dataset buffers alive during calls.

```cpp
auto created = vsag::Factory::CreateIndex("hgraph", config_json);
if (!created.has_value()) {
    throw std::runtime_error(created.error().message);
}
auto index = created.value();
auto check = [](const auto& result) {
    if (!result.has_value()) {
        throw std::runtime_error(result.error().message);
    }
};
auto built = index->Build(base);
check(built);
if (!built.value().empty()) {
    throw std::runtime_error("some initial vectors were not inserted");
}
auto removed = index->Remove(removed_labels, vsag::RemoveMode::FORCE_REMOVE);
check(removed);
auto appended = index->Add(added);
check(appended);
if (!appended.value().empty()) {
    throw std::runtime_error("some added vectors were not inserted");
}
// MCI compaction is automatic; no public Flush call is required or available.
auto result = index->KnnSearch(query, 10, search_json, filter);
check(result);
```

Build/Add also return failed-insertion labels: checking only `expected` is insufficient.
`removed.value()` is the actual removal count. MARK_REMOVE retains physical slots.
Compaction is automatic, including at the end of FORCE_REMOVE; it does not serialize.
Remove accepts external labels, not inner slots; do not retain internal node/clique IDs across compaction.
See `examples/cpp/324_feature_hgraph_mci_companion.cpp` for Dataset and Filter construction.

## Add, Delete, Serialize, and Stats

When MCI is enabled by the flat build parameters, `HGraph::Add()` first inserts the HGraph
batch, then updates the MCI companion for the successfully inserted rows. It first tries to join
suitable existing cliques, then creates incremental cliques if the unique neighbor degree is still
too low.

That update works in chunks. Inside a chunk the candidate KNN, the clique JOIN proposals and the
local clique construction of every row are planned in parallel against one frozen read view, and a
single thread then merges the plans in inner-ID order and applies them. Per-row coverage and the
degree target do not depend on the planning order, but the candidate KNN comes from concurrent
graph search, so as in the full parallel build the resulting clique structure may depend on the
thread count and on scheduling.

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

MCI automatically merges incremental metadata into its two CSR arrays after 100 successful
vector mutations, using one counter shared by Add and MARK_REMOVE. Add checks after each
inserted point, including within a large batch. MARK_REMOVE projects and repairs the complete
deletion batch before checking the threshold, so a large deletion batch compacts once at its end.
Duplicate/missing removal IDs and failed insertions do not count; repair seeds do not count as
new vectors. FORCE_REMOVE retains its mandatory end-of-batch compaction and resets the counter.
Successful compaction resets the counter to zero; a new full build or load also resets this
runtime-only counter. Below the threshold, queries and serialization still include delta.
There is no public Flush interface and no change to the Index virtual interface for compaction.

Flush includes new delta cliques and members appended to base cliques, drops retired cliques and
deleted members, and rebuilds the node-to-clique mapping with compact clique IDs. It clears the
delta storage but preserves vector inner IDs and deletion markers. The operation is idempotent
and does not rebuild HGraph or persist data to disk; use `Serialize()` for persistence.

Flush is serialized with Add/Remove. It holds an exclusive clique-storage lock while constructing
and publishing the replacement, so searches may wait. Replacement buffers are allocated before
publication; an allocation failure leaves the old clique representation intact. Automatic
maintenance defers allocation failures without failing a completed Add/MARK_REMOVE and retries
on the next successful mutation. This does not change FORCE_REMOVE's failure semantics. Temporary memory
includes both the old and new CSR. Do not retain clique IDs across a flush.
Loading a validated compact CSR with no delta entries, empty clique rows, or deletion/retirement
markers preserves the clean state, so its first Flush skips rebuilding the CSR. Other snapshots
are conservatively compacted when flushed.

MCI search pins CSR, delta, and deletion masks with one shared lock per query and traverses them
without copying membership lists. Contiguous FP32 vectors continue to use direct distance
calculation after Add/Remove, before or after Flush; `mci_raw_float_csr` reports this path, including
queries that traverse delta. Other vector layouts use the same traversal and candidate queue with
their normal distance computer. Deletion filtering pins the deletion set once per MCI query,
checks the user filter first, and avoids acquiring the deletion-set lock for every visited node.

Add/MARK_REMOVE may temporarily unpublish the companion, so concurrent queries still fall back to
HGraph under the existing mutation semantics. This does not guarantee complete recall during a
mutation; fallback results can be empty when the graph entry point has been deleted. Automatic
compaction runs within that mutation; queries may wait for the clique storage lock.
Fallback queries recheck candidate IDs against one deletion-set view before packing results.
This removes old candidates deleted during traversal, including an old slot whose label has
been re-added at a new slot. It does not make an entire concurrent search a transactional snapshot.
If candidates are deleted during traversal, KNN fallback may return fewer than `k` results even
when enough live vectors remain. Results are best-effort under concurrent mutations; there is no
secondary fill pass, since another pass could encounter further mutations too.

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

## Evaluation Notes

Development benchmark sources, scripts and raw results are not part of this PR; the field
descriptions below document the CSV produced by that harness.

### Reading results and validating

Five-stage runs protect evaluation top-k neighbors and retain the same truth. Random churn does
not protect neighbors; each checkpoint selects live top-k from exact full-pool distance rankings.
Do not mix their recall conclusions.

| CSV fields | Meaning |
| --- | --- |
| `stage`, `active_vectors`, `index_elements` | Stage, live count, public element count; the last is not physical storage accounting |
| `ef_search`, `recall_at_k`, `qps` | Search width, recall, timed throughput; fixed ef is not fixed quality |
| `build_seconds`, `mutation_seconds`, `flush_seconds` | Historical build/mutation/explicit-flush timings; current automatic compaction is included in mutation time, with no public post-stage Flush |
| `index_memory_bytes`, `vector_memory_bytes`, `graph_memory_bytes`, `mci_memory_bytes` | Index/component accounting, not RSS |
| `mci_route_ratio`, `mci_raw_float_ratio` | Actual MCI/direct FP32 route rates |
| `mci_total_cliques`, `mci_delta_cliques`, `mci_total_memberships` | Clique, delta-clique, and membership counts |
| `avg_dist_cmp`, `avg_hops`, `avg_seed_count` | Search work useful for QPS diagnosis |

MiB = bytes / 1,048,576. Stage memory/timing repeats across ef rows; do not sum it. QPS is
multithreaded throughput: `1000 / QPS` is not per-request latency. CSV has neither latency percentiles
nor per-stage RSS. Raw result files are not included in this PR. These measurements predate adaptation to newer
main and are not a performance rerun of current PR HEAD.

```bash
make debug VSAG_ENABLE_TESTS=ON COMPILE_JOBS=12
build/tests/unittests '[mci],[LabelTable],RaBitQSplitDataCell serialize and methods'
```

Before this documentation update: targeted C++ tests passed 38 cases / 3,294 assertions; Python
suites passed 10+1+3 tests, with the executable integration suites using a Debug benchmark for
correctness. Review follow-up adds reverse-edge restoration and reload-mutation regressions.
Full lint, full tests, and >=90% coverage verification are outstanding. Large-dataset snapshot reuse,
ABI, serialization compatibility, and broader concurrent mutation require separate validation;
targeted passes are not proof of production readiness.

### Bitmap prefetch for static MCI traversal

See also the query-local filtered clique CSR option below.

When an immutable inner-id validity bitmap is available, static MCI traversal reads clique members in blocks of 32 and prefetches the bitmap locations of members up to 96 positions ahead. A temporary validity mask selects members in their original order, after which the existing epoch-based node check and distance evaluation run unchanged. This avoids a dependent filter branch for every rejected member. It does not preprocess the complete valid set, change seeds, or change clique membership. Filters without a directly usable bitmap keep the original visitor.

The development comparison modes are controlled by `VSAG_MCI_MEMBER_MODE` (0 original visitor, 1 rolling bitmap prefetch, 2 block prefetch, 3 block validity mask, 4 mask plus prefetch, default 4), `VSAG_MCI_MEMBER_DISTANCE` (default 96, max 256), `VSAG_MCI_MEMBER_BLOCK` (default 32, max 64), and `VSAG_MCI_MEMBER_LOCALITY` (GCC prefetch locality hint, default 1). They are read once per query; set them before concurrent searches, and do not change the process environment while queries run. These controls do not affect index serialization or require rebuilding an index.

### Query-local filtered clique CSR

The static MCI routing path accepts the experimental search option `hgraph.mci_filtered_cliques=true` (default false), requiring `search_no_lock=true`. With a complete valid-ID list, the query maps, sorts and deduplicates valid inner IDs, then counts their node-to-clique incidences and packs legal clique members using prefix sums and a second fill pass. Each query worker reuses a flat member pool, compact group metadata and an epoch-tagged direct map. There are no per-clique heap allocations or per-query clears of the entire clique map. Singleton filtered cliques store no members.

Both ID preparation and temporary CSR construction occur inside search timing. Expansion visits valid members directly, bypassing bitmap and predicate callbacks while retaining node and clique visit markers. Members are ordered by increasing inner ID, potentially changing traversal order and results at finite search breadth. Exhaustive seeds skip construction. Missing or incompletely mapped valid-ID lists retain the original path. This is a query-private view of the existing saved index; it creates no new filtered-graph connections.

Statistics include `mci_filtered_cliques_requested`, `mci_filtered_cliques_used`, `mci_filtered_id_prepare_ms`, `mci_filtered_clique_build_ms`, `mci_filtered_incidences`, `mci_filtered_nonempty_cliques`, `mci_filtered_kept_cliques`, `mci_filtered_members`, and `mci_filtered_capacity_bytes`. `mci_member_checks` counts scanned filtered member slots; `mci_filtered_source_member_checks` counts the corresponding original clique slots.

### Dynamic candidate-clique scan

The experimental MCI-route option `hgraph.mci_dynamic_clique_scan=true` defaults to false. It requires `search_no_lock=true`, an immutable base CSR, a complete mappable `GetValidIds` list, and `mci_filtered_cliques=false`. The saved index is reused. Missing complete IDs are rejected rather than silently running a different algorithm.

All sampled seeds are scored. The closest `max(k, mci_ef_search)` seeds initialize a max-heap and their clique union X. A reusable thread-private epoch array stores the number of current candidates covering each clique. Remaining legal IDs are scanned in increasing inner-ID order. A point is scored once if any of its inverse-CSR clique IDs has a positive count, and its distance immediately updates the candidate and top-k result heaps. Replacing A with B subtracts A's memberships and adds B's. The implementation commits both updates before examining another point; it adds B first to avoid transient false activations of shared cliques. Candidate distance ties prefer the smaller inner ID.

A new zero-to-one clique activation schedules another pass over unscored legal IDs, including IDs before the current scan position. Search stops after a full pass without new activations; this guarantees that no unscored legal point is covered by the final X. Results depend on scan order and finite ef; this is approximate search. Exhaustive seeds skip union construction and further scanning. There is no all-clique member materialization, clique-member scan, or per-query full state clear.

Dynamic replacement may propagate through more than two graph hops. `mci_hops_limit` is rejected in this mode and the reported `mci_hops_limit=0` denotes no applicable depth limit; scan passes are not graph hops. The original traversal accepts a positive uint32 `hgraph.mci_hops_limit` (default 3) for controlled comparisons. Existing idle and lower-bound traversal stops do not apply to dynamic scanning.

Diagnostics include `mci_dynamic_clique_scan_used`, `mci_dynamic_selected_seeds`, `mci_dynamic_scored_points`, `mci_dynamic_scan_passes`, `mci_dynamic_point_checks`, `mci_dynamic_clique_checks`, `mci_dynamic_count_updates`, `mci_dynamic_activations`, `mci_dynamic_replacements`, `mci_dynamic_active_cliques`, and ID-preparation/initial-union/scan timings with the `mci_dynamic_` prefix. `mci_member_checks=0` refers to original clique-member slots; inverse clique tests are counted separately. All preparation is inside native query timing.

### Ordered one-pass and two-pass clique scans

With `mci_dynamic_clique_scan=true`, setting `hgraph.mci_clique_scan_passes` to 1 or 2 selects a bounded ordered scan. Omitting it preserves the original fixed-point scan. The immutable node-to-clique CSR rows must contain strictly increasing clique IDs; an older saved CSR without this invariant is not supported by this experimental mode. The saved pruned Cohere10M index used for validation has been checked across all nodes and memberships. Index serialization is unchanged.

Each worker reuses metadata for legal points: original inner ID, CSR start/end, minimum and maximum clique ID. Sequence 1 sorts complete clique-ID rows lexicographically, breaking equal rows by inner ID. Sequence 2 orders sequence-1 positions by maximum clique ID. Stable radix sorting handles the minimum/maximum keys; only groups with equal minima need lexicographic tail comparisons. A suffix-minimum position table on sequence 2 provides the first sequence-1 position whose maximum can reach the current lower bound. Blocks of 32 rows store their maximum endpoint.

The current candidate union X retains exact per-clique reference counts plus an epoch-tagged hierarchical bitmap. This supports cached minimum/maximum IDs, fast membership tests, and successor lookup without clearing all words per query. Candidate replacement is committed before testing the next point. Points with a disjoint clique-ID range are rejected, consecutive impossible blocks are skipped, and the scan stops when sequence-1 minima exceed the current X upper bound. A successor lookup skips inactive prefixes of a point's row; increasing member IDs permit stopping above the active upper bound. Immutable CSR reads are prefetched ahead of the row scan.

The second pass uses the envelope of cliques newly activated during pass 1 together with current X bounds. Sequence 2 jumps to its safe starting position. Initially it ends before the last first-pass activation position: after that position coverage could only shrink. A new activation in pass 2 widens its envelope and restores the full possible suffix. Every point is scored at most once. The pass count is capped: unlike the unbounded mode, this does not guarantee that the final X covers no unscored legal point. Neither scan passes nor sequence order denote graph hops.

`hgraph.mci_clique_scan_gallop` (default false) is an experimental alternative that performs repeated successor/binary-search jumps inside a row; false uses sequential bitmap probes after its initial prefix skip. It requires a bounded scan when true. Both variants produce identical results at the same pass cap and ef. All sorting, bitmap maintenance and search work is included in native query timing. Diagnostics use `mci_bounded_` for the pass limit, intersection variant, ordering time, range/gap rejections, prefix/block/tail skips, second-pass start and initial end, binary comparisons, and initial/final X bounds. Ordering time is also included in `mci_dynamic_scan_ms`.

### Navigation for the routed MCI branch

The non-iterator KNN path tries MCI with its filter-derived seeds before traversing HGraph routing layers. A completed MCI route does not consume an HGraph entry point and records zero routing-layer distances. If MCI is unavailable or does not accept the request, the ordinary HGraph entry-point traversal remains active. This optimization does not change stored indexes or the MCI seed budget.

### HGraph neighborhoods for high-incidence scan candidates

The experimental query-only option `hgraph.mci_hub_clique_threshold` defaults to 0 (disabled). With a positive threshold, a candidate whose number of clique memberships is at least that threshold remains eligible for both heaps, but activates no outgoing cliques. Its existing base HGraph neighbors replace that candidate's dense neighborhood. Normal candidates still maintain the exact reference counts of their clique union. Complete stored memberships are retained, so another candidate can still discover the high-incidence point through a shared clique.

`hgraph.mci_hub_hgraph_hops` is 1 by default and accepts only 1 or 2. One hop scores unvisited valid neighbors. Two hops also follows the adjacency lists of invalid one-hop neighbors; these bridge points are not scored and never enter either heap. A newly accepted valid high-incidence candidate is queued for the same processing. Queued candidates that have already left the heap are skipped. Query epoch markers deduplicate graph expansions and distance evaluations.

This option requires the routed dynamic-clique-scan branch, `search_no_lock=true`, immutable CSR, and an existing base HGraph. It supports the fixed-point and bounded one/two-pass scan variants. It does not launch a separate global HGraph search, and `hgraph_ef_search` does not bound these local replacement neighborhoods. It is an approximate neighborhood substitution, with no promise to preserve recall. Exhaustive seeding still skips all graph/clique expansion. Stored indexes and serialization are unchanged.

The `mci_hub_` statistics record the threshold, hop limit, accepted hub candidates, suppressed memberships, expanded hubs, invalid bridge expansions, neighbor checks, and newly scored neighbors. The last count is a subset of the query's total distance evaluations, not an additional count. All work is included in native query time.

### Global MCI correlation router

The experimental API in include/vsag/global_mci_hybrid.h reuses one existing FP32/L2 HGraph with a static global MCI companion. Call GlobalMCIHybrid::Create(index), then its KnnSearch. No partition index, centroids, or workload correlation labels are used. The index must remain immutable throughout the router's lifetime. Only search_no_lock=true and query-internal parallelism=1 are supported; independent queries may run concurrently.

The outer retention router first chooses filtered HGraph when the retained fraction is at least hgraph_valid_ratio_threshold (default 0.05). Otherwise, a complete unfiltered HGraph query, including the bottom graph, retrieves approximate top-10 anchors with correlation_navigation_ef (80). Legal anchors are prepended to the MCI seeds, within mci_seed_max_count.

The predictor probes the global MCI cliques incident to these anchors. It selects at most correlation_max_cliques (256) in anchor round-robin order and checks at most correlation_member_budget (8192) member slots distributed across those cliques. Repeated points count only once using thread-private epoch markers. Local legal-point density divided by the global retained fraction is a local-yield heuristic, not an unbiased estimator over the entire overlapping clique union.

An enrichment at most correlation_negative_ratio (0.5), with at least correlation_min_expected (1) expected legal sample points under independence, selects an exact scan of the complete legal set. An enrichment at least correlation_positive_ratio (2) with a legal hit predicts positive correlation. All other cases keep global MCI. Empty or insufficient samples do not force a scan. Anchors and probes never use ground truth. correlation_routing=false provides the original retention router; correlation_probe_only=true retains navigation, prediction and seed injection while disabling the predicted scan.

Example parameters, passed directly or wrapped in global_mci_hybrid:

    {
      "hgraph_valid_ratio_threshold": 0.05,
      "ef_search": 40,
      "mci_ef_search": 40,
      "hgraph_ef_search": 80,
      "mci_seed_ratio": 0.1,
      "mci_seed_coverage": 0,
      "mci_seed_max_count": 10000,
      "mci_hops_limit": 3,
      "search_no_lock": true,
      "parallelism": 1,
      "correlation_navigation_ef": 80,
      "correlation_max_cliques": 256,
      "correlation_member_budget": 8192
    }

Filters should expose complete GetValidIds; scan falls back to checking every index label when enumeration is unavailable. All navigation, clique probing, ID conversion, scan preparation and search work occur inside the query API and must be included in end-to-end latency. Result statistics under global_mci_hybrid include the route, prediction, sample counts, legal navigation seeds, navigation/search distance counts and phase times. The router does not rebuild, reorder, prune or serialize the index.

### Distance-sample correlation prediction

GlobalMCIHybrid accepts correlation_method="distance_samples" as an alternative predictor on its MCI branch. The predictor performs no HGraph navigation and does not read clique memberships. Existing retention routing still applies before prediction. It reuses the saved index without changing serialization.

Uniform sampling without replacement selects correlation_global_samples (512) global inner IDs and correlation_valid_samples (256) positions in the complete GetValidIds list. A fixed correlation_sample_seed (42) determines the global sample; the valid sample additionally incorporates query-vector bits so the same query/filter is repeatable. Provider ordering must be stable, and the ID list must be complete and unique. Samples are mapped and scored inside query timing; the global sample is independent of the predicate.

The correlation_radius_quantile (0.1) of global distances defines a radius. The fraction of legal samples within that radius divided by the corresponding global fraction estimates enrichment. Ties use the actual inclusive global fraction. Approximate Wilson intervals with correlation_confidence_z (1.96) form a ratio band. Its upper end at most correlation_negative_ratio (0.5) selects a scan; its lower end at least correlation_positive_ratio (2) predicts positive. Other cases remain neutral. These are uncertainty heuristics, not a formal joint confidence guarantee for an adaptive quantile or a guarantee about the much smaller true top-k neighborhood.

Every legal sample becomes an MCI seed using the exact distance already computed. These seeds replace the ordinary ratio-driven seed list and must fit mci_seed_max_count (zero means unlimited). The private backend bridge uses mci_seed_replace, mci_seed_ids and aligned finite mci_seed_distances. Cached seeds enter both candidate and result queues at hops=1 without another distance calculation. The scan branch also reuses these distances. Global/legal sample overlap is scored once. Source-index vectors and graph structures remain immutable.

correlation_method="sample_seeds" is an ablation: it uses the same legal samples and distance reuse, with no global sample, prediction or scan switch. The default clique_probe method retains the earlier HGraph top-10 predictor. Statistics distinguish sample_distances, reused_sample_distances, search_distances and total_distances; reused counts are excluded from new distance evaluations. All selection, mapping, serialization, prediction and search work is timed.


### Static routing compatibility and budgets

The ordinary HGraph/MCI API retains its live/delta traversal. Setting only `search_no_lock=true` does not select the experimental static algorithm. The wrapper supplies the existing static route options; `mci_hops_limit` applies to that path with seed depth 1 and child depth parent+1. A point at the depth limit can enter the result but cannot expand. Building, updating and loading indexes retain the upstream implementation; construction-only pruning and clique reordering from prior experiments are not part of this change.

| Distance-sample parameter | Default | Accepted values |
| --- | ---: | --- |
| `correlation_global_samples` | 512 | Integer 1–65536 |
| `correlation_valid_samples` | 256 | Integer 1–1024; determines the replacement seed budget |
| `correlation_sample_seed` | 42 | Integer 0–1000000000 |
| `correlation_radius_quantile` | 0.1 | Finite value in (0,1] |
| `correlation_confidence_z` | 1.96 | Finite positive value |

`mci_seed_max_count` must be zero or at least the configured `correlation_valid_samples`, even when the filter contains fewer points. The wrapper rejects an insufficient cap. The sampling modes require complete, unique, stable `GetValidIds`; unlike the legacy clique predictor, they do not fall back to enumerating the whole index. Prefer the bitmap interface for large predicates. Use `correlation_probe_only=true` to measure prediction without its scan decision, and `correlation_method="sample_seeds"` to compare the same legal seeds without global sampling. Neither mode promises exact MCI recall.

To enable the new predictor, add `"correlation_method":"distance_samples"` to the wrapper example above. Its default remains `clique_probe`. Parameters passed to ordinary `Index::KnnSearch` do not instantiate a `GlobalMCIHybrid` wrapper.
