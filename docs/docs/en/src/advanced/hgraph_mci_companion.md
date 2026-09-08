# HGraph MCI Companion

HGraph can optionally build an MCI (Maximal Clique Index) companion for filtered KNN
search. The companion stores clique metadata beside the HGraph index and shares HGraph's
vector storage. It is not a standalone index type: create the index as `hgraph`, not `mci`.

For implementation details, memory analysis, and 10k/3m results, see
[HGraph MCI Mutation Design and Benchmark Report](hgraph_mci_mutation.md).

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
| `mci_delete_clique_size_threshold` | Retire an affected clique when its live size after deletion is below this value (default `3`). |
| `mci_delete_node_mct_threshold` | Repair a member of a retired clique when its projected live clique count is below this value (default `3`). |

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
existing cliques, then creates a small incremental clique when no good join target exists.

Prefer `Build()` for the initial index, followed by incremental additions. Add on an empty
index can trigger construction, but many tiny Adds are not recommended as a replacement
for bulk Build.

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

FORCE_REMOVE is serialized with Add, MARK_REMOVE, and Flush; queries wait until ID moves and
repair finish. This is not a transactional operation: deletions completed before an error may
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

## Add/Delete Performance Regression

The repository includes a five-stage regression benchmark for filtered HDF5 datasets. It
measures QPS–recall curves for the initial full index, after deleting 10%, after deleting 20%
in total, after adding 10% back, and after adding all deleted vectors back:

```bash
scripts/perf_reports/run_hgraph_mci_mutation.sh
```

The default dataset is `/root/data/codefilter-10k-384-angular-f32.hdf5`, and results are written to
`/tmp/vsag_mci_mutation/`. Override paths with `MCI_DATASET_PATH`, `MCI_RESULT_DIR`, and
`MCI_BUILD_DIR`. The quick profile evaluates recall on 200 queries, times 10,000 searches at
`ef_search=40,80,160`, uses 16 build and search threads, and builds MCI with `mci_mcs=50`.

Pass `--force-remove` to enable physical deletion support and use FORCE_REMOVE for both deletion
stages. Without this flag, MARK_REMOVE remains the default. CSV `remove_mode` records the mode.
`vector_memory_bytes` and `graph_memory_bytes` record base-vector and graph storage accounting.
Every value can be overridden on the command line; for example, use `--query-count 0` to
evaluate every query.

Pass `--flush-after-mutation` to flush after each mutation stage before measuring search. The CSV
also records `flush_seconds` and `mci_raw_float_ratio` separately from mutation time and MCI routing.

The benchmark loads the complete 10k HDF5 training matrix into memory. It validates that the stored
neighbors match the filtered workload and recomputes exact filtered ground truth when they do not.
To keep that ground truth identical in all five stages, it protects the evaluated top-k vectors and
samples deletions from all remaining vectors with a fixed seed. The same vectors are added back.
With `--max-base`, exact filtered ground truth is always recomputed over the truncated subset.
