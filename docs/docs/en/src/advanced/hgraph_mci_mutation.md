# HGraph MCI mutation design and benchmark report

> Benchmark sources, scripts and raw results are excluded from this PR. Tool names, options
> and outputs below describe historical experiments, not tools shipped in this checkout.

This report documents ADD, MARK_REMOVE, FORCE_REMOVE, Flush, and the Codefilter
10k/3m measurements collected on 2026-09-07. For introductory configuration, see
[HGraph MCI Companion](hgraph_mci_companion.md).

## 1. Summary and scope

- ADD and deletion repair share incremental clique maintenance and three delta structures.
- Deletion retires only undersized affected cliques and repairs selected undercovered members.
  It does not rebuild the entire one-hop neighborhood.
- MARK_REMOVE retains vector slots. FORCE_REMOVE repairs graph edges, moves tail slots,
  remaps MCI, and shrinks storage. Flush alone does not physically delete vectors.
- The full 3m file contains 3,241,378 vectors. After deleting approximately 20%, allocated
  index memory fell from 6,323.25 to 5,215.30 MiB, a 17.52% reduction.
- All FORCE_REMOVE stages passed physical-count, full live-node coverage, zero-inactive-mask,
  and 100% direct-FP32-route checks.
- Restoring all vectors resulted in 6,411.62 MiB. QPS and recall both changed; higher QPS at
  the same ef is not proof of better performance at equal recall.

These results describe uncommitted changes on `feat/mci-delete`, based on commit
`f41d6cd143380886ba193f9cb6d7864ba1c790f8`. That commit alone does not contain the new behavior.

**The benchmark uses HGraph with `graph_type=nsw`: single-layer NSW + MCI, not the separate
`hnsw` index and not a multilayer HNSW configuration.**

## 2. Construction and layout

### 2.1 Construction

```text
Input vectors
  → HGraph graph and vector storage
  → MCI candidate KNN graph
  → clique construction and coverage selection
  → clique-to-node CSR and node-to-clique CSR
```

`mci_knng_source=hgraph` generates candidates from HGraph; `odescent` builds a separate
candidate graph. An external KNN graph file takes precedence. Contiguous FP32 vectors use
`BuildMCICliques`; other layouts have a generic path. MCI shares HGraph vector storage.

The cliques are search structures constrained by candidate sets, expanded distance bounds,
and size limits. Incremental maintenance does not enumerate every strict maximal clique of
a fixed global graph. Full coverage is not a guarantee of exact query recall.

### 2.2 Base and delta

| Structure | Contents |
| --- | --- |
| `p_maxc_`, `maxcs_` | Base clique-to-node CSR offsets and members. |
| `p_node_to_cid_`, `node_to_cids_` | Base node-to-clique CSR offsets and clique IDs. |
| `delta_cliques_` | Complete members of newly created cliques. |
| `delta_clique_extra_` | Members appended to base cliques without changing base CSR. |
| `delta_node_to_cids_` | Incrementally created node-to-clique relationships. |
| `inactive_nodes_` | Inactive vector-slot mask. |
| `retired_cliques_` | Retired-clique mask. |

For `B` base cliques, a clique with `cid < B` reads base members plus extras; otherwise it
reads `delta_cliques_[cid - B]`. Node memberships combine inverse CSR and node delta rows.
Effective views exclude retired cliques and inactive nodes, including when Add and Delete
are interleaved. New cliques created by repair participate in subsequent operations.

External labels, vector inner IDs, and clique IDs are different identities. FORCE_REMOVE
may move inner IDs; Flush may renumber clique IDs. Applications must not retain either
internal ID as a stable external handle.

## 3. ADD strategy

### 3.1 Batch ordering

`HGraph::Add()` serializes MCI mutations, validates/prepares the batch, marks an existing MCI
unavailable, inserts successful rows into HGraph, and then incrementally updates MCI for each
successful row. It publishes the new MCI total and recalculates memory after the loop.

Graph insertion can use the build thread pool. The outer incremental MCI loop is currently
serial. A batch does not publish a complete new MCI after each graph insertion.
If no usable MCI exists, Add builds one; force-enabled indexes reapply surviving soft-delete
masks. Empty-index recovery is supported, but many tiny Adds are not recommended as a
replacement for initial bulk Build.

### 3.2 Candidate neighbors

`incremental_update_mci_clique(u, vector)` calls `search_mci_knn`, using normal HGraph search with
`use_mci=false`. It requests up to `mci_mcs` candidates, excludes the node itself and IDs
outside the current inner-ID visibility boundary, expands the requested result count when
needed, and uses search ef of at least 100. Later rows of the same batch are not all treated
as already processed nodes. Candidates are deduplicated and, if oversized, distance-sorted
and truncated. This is approximate candidate generation, not exhaustive KNN.

### 3.3 Prefer existing cliques

For neighbors `K(u)` and an effective clique `C`:

```text
join_ratio(u, C) = |K(u) ∩ C| / |effective members of C|
```

Nonempty cliques below the incremental size cap qualify when the ratio reaches
`mci_incremental_join_ratio_threshold` (default 0.6). Targets are ordered by overlap count
descending, then clique ID ascending. JOIN continues until the seed's unique neighbor degree
reaches its target, rather than stopping after three cliques. A successful append writes base extras or complete delta-clique members,
respectively, and updates the node's inverse delta row. Invalid, duplicate, retired, or
over-capacity additions are rejected.

Degree is the union of all live members of cliques containing the seed, excluding the seed
itself and counting overlapping neighbors once. Cliques adding no new neighbors are skipped.
For N live vectors, the target is `max(degree_min, min(N / n_divisor, mcs / mcs_divisor))`, capped at
N-1; a zero/one-vector index has target zero. Integer division rounds down. The parameters
`mci_incremental_degree_n_divisor` and `mci_incremental_degree_mcs_divisor` default to 10000
and 2. `degree_min` is configured by `mci_incremental_degree_min` (positive integer, default 50).
N includes the completed Add batch but excludes marked removals. Thus N=10000 and mcs=200
gives target 50, whereas N=3241378 gives target 100. The configured floor applies even when mcs/2 is smaller;
the existing candidate and coverage limits still apply. This is a stopping target, not
a hard maximum: joining an entire clique can overshoot. The obsolete ADD clique-count
parameter and greedy construction path have been removed. This branch still uses
overlap ratio and does not revalidate every pair of clique members.
The floor is serialized and checked for compatibility when loading. Missing fields in old
configurations default to 50. Set the floor to 100 when creating a new index to request the
previous fixed-floor behavior; loading still requires matching stored parameters.

### 3.4 Fill a degree deficit with the full-build local clique algorithm

`BuildMCICliques` and incremental construction both call `MCILocalCliqueBuilder::Build`.
The shared core collects candidates, constructs the distance-threshold local graph, enumerates
maximal cliques, and selects cliques covering required nodes. The size threshold is
`max(2, min(clique_max, candidate_limit + 1, visible_total))`; incremental construction uses
`mci_incremental_clique_max` for the cap. Alpha expansion uses the same `next_mci_alpha` policy;
a two-node result does not end expansion when the configured threshold is larger.

ADD remaps only its seed and KNN to local IDs, so scratch coverage storage is O(mcs), not O(total).
Only the seed is marked as requiring fresh coverage, including when it is an existing deletion
repair point. Distance callbacks read vector codes (including block-memory storage); full Build
retains its SIMD batch distance kernel and parallel round scheduling. Empty candidates produce a
singleton; after alpha exceeds 100, the shared high-alpha fallback can produce a smaller clique.
After JOIN, any remaining degree deficit triggers construction using KNN candidates that are
not already neighbors of the seed. Repeat until the target is reached, candidates are exhausted,
or construction makes no degree progress. This avoids repeatedly constructing the same neighbor
set. Coverage limits and insertion-prefix visibility can prevent reaching the target; a seed is
still covered by a singleton when no usable neighbors exist. Deletion repair uses this same
degree policy after its existing small-clique/low-membership candidate selection.

`AppendNewClique` writes complete members and inverse delta memberships for every member.
This is bounded local construction, not a guarantee of a globally maximum or unique maximal clique.
Historical measurements below predate the shared-builder and degree-target changes and are not their performance results.

## 4. Shared deletion policy

### 4.1 Selective repair replaces full one-hop rebuilding

The earlier proposal retired every clique covering a deleted node and rebuilt its complete
one-hop neighborhood. The current implementation instead uses strict less-than thresholds:

- `T_size = mci_delete_clique_size_threshold`, default 30.
- `T_mct = mci_delete_node_mct_threshold`, default 3.

With the default size threshold, an affected clique with 29 remaining live members is retired;
one with 30 is retained. Unaffected small cliques are not scanned for retirement. This is a
retirement threshold, not the construction size cap; the survivor-membership threshold remains 3.
If the construction size cap is below 30, every affected clique of that size is retired;
evaluate or adjust the retirement threshold together with that cap.
Explicitly configured values (including 3) are preserved, and serialized parameters must match
when loading an existing index. Historical experiments below retain their original configurations
and do not measure the new default of 30.

`PrepareDelete(D, T_size, T_mct)` produces a read-only snapshot:

1. Find effective cliques covering `D`, including base, extras, and delta cliques.
2. Select those with fewer than `T_size` remaining members after removing `D` for retirement.
3. Deduplicate their surviving members as repair candidates.
4. Retain candidates with fewer than `T_mct` projected memberships after those retirements.

`CommitDelete` sets node/clique masks. Before repairing each selected node, the code checks
that it is still live and undercovered: an earlier repair may already have covered it.
MARK_REMOVE passes the complete removed-ID batch to one PrepareDelete/CommitDelete pair,
including when reapplying tombstones after a full MCI rebuild. Repair candidates are deduplicated
before repair begins, so shared cliques are evaluated against the final batch state.

### 4.2 What sharing ADD means

For pure FP32 configurations (float32 input, FP32 base and precise codes):

```text
repair_mci_clique(v): recover the existing query vector
  → incremental_update_mci_clique(v, vector, total)
  → search_mci_knn(v, vector, total): HGraph search, use_mci=false
  → incremental_update_mci_clique(v, knn_ids, total)
  → try_join_mci_clique / build_incremental_mci_clique
```

This invokes shared internal clique maintenance, **not public `Index::Add()` on an existing
vector**. It neither reinserts the vector nor duplicates its external label. Repair writes
the same delta structures as Add.

Repair now shares the complete MCI pipeline with Add, from KNN generation through delta
updates, instead of scanning and fully sorting all live vectors for each repair point.
The KNN limit remains `mci_mcs`: initially request `k+1` to exclude self, use
`ef_search=max(query_k,100)`, and reuse Add's request expansion when filtering leaves too
few candidates. This is approximate graph search, not exact KNN or parallelized repair.

Existing points use contiguous FP32 data when available. Non-contiguous storage such as
block_memory_io reads and decodes the point's FP32 codes before entering the same pipeline.
Quantized configurations and other input dtypes retain exact full-scan candidates, but a bounded
top-K heap retains only mcs entries (O(mcs) scratch space), with the same distance/ID ordering.
This still performs O(N) distance evaluations per repaired node, or O(RN) for R repair nodes;
replacing this exact scan with quantized approximate search is outside the scope of this change.

New additions retain insertion-prefix visibility. Existing FP32 repair points search the entire
current index, excluding removed points and self. Shared clique construction also receives
this explicit bound, rather than constraining repaired low IDs by `node_id + 1`.

After graph repair and ID remapping, FORCE_REMOVE releases its exclusive physical-deletion
lock before shared search. The mutation lock remains held and MCI remains unpublished.
Search acquires its own read lock; deletion reacquires the exclusive lock before Flush and
storage shrinking. This avoids recursive locking. MARK_REMOVE remains logical deletion.

Historical measurements in this report predate this change. The old repair scanned and
sorted all live candidates per repaired node, costing approximately
`O(R × (N × d + N log N))`. Historical latency and recall do not measure the new pipeline;
new benchmarks are required.

### 4.3 Guarantees not provided

`T_mct` triggers repair; it is not an enforced post-repair minimum. An update stops when its
unique-neighbor degree target is reached or its usable candidates are exhausted. The code
does not recount all one-hop degrees, split every remaining large
clique into connected components, or prove global graph connectivity after deletion.
FORCE_REMOVE graph repair and MCI coverage repair are separate operations.

## 5. MARK_REMOVE

The default Remove mode marks labels, updates live/deleted counts, temporarily unpublishes
MCI, snapshots/commits the complete removed-ID batch before selective survivor repair, and
republishes MCI and memory accounting.

It does not move vector slots, shrink their capacity, or perform FORCE_REMOVE graph repair.
Base CSR may retain stale relationships filtered by masks; repairs populate delta. Later
Add appends at the physical tail rather than automatically reclaiming soft-deleted slots.

`GetNumElements()` and benchmark `index_elements` are live counts. In this workload,
`mci_total_nodes` exposes the corresponding physical slot count, which may be larger.

## 6. FORCE_REMOVE

### 6.1 Enabling it

Set `index_param.support_force_remove=true` when constructing the index:

```cpp
auto result = index->Remove(ids, vsag::RemoveMode::FORCE_REMOVE);
if (!result.has_value()) {
    // Inspect result.error(); do not assume automatic rollback.
}
```

MCI force removal requires flat graph storage and automatically enables reverse edges for
bottom/hierarchical graph parameters. Flat storage does not mean single-layer topology.
Deduplicated storage, duplicate-vector groups, and built-in attribute storage are incompatible.
The custom label filter used here is not built-in attribute storage. Existing indexes without
force-removal support must be rebuilt.

### 6.2 Batch operation

1. Acquire the MCI mutation mutex, then the exclusive force-removal lock, blocking queries.
2. Resolve labels, including explicitly targeted soft-deleted nodes; deduplicate and ignore misses.
3. Sort by original inner ID descending and allocate `old_to_new`/`current_to_old` maps.
4. Prepare a deletion snapshot for the entire set, unpublish MCI, and commit masks.
5. For each target, repair graph edges and move the current tail into the removed slot.
6. Update the maps: deleted IDs become invalid; survivors map into `[0, N')`.
7. `RemapNodes` rebuilds both CSRs and remaps untargeted soft-delete masks.
8. Map repair candidates to current IDs and invoke shared incremental clique repair as needed.
9. Automatically Flush repaired metadata into CSR.
10. Shrink vectors, graph, and related per-node storage; update capacity and publish MCI.

Descending original IDs prevent earlier tail moves from relocating pending targets.
For example, deleting IDs 2 and 7 from ten slots can move 9 to 7, then 8 to 2, leaving `[0, 8)`.
Labels move with their actual vectors, not with the old destination slots.

MARK_REMOVE computes MCI snapshots one node at a time; FORCE_REMOVE snapshots the set in one
call. Changing mode or batch size can therefore change repair order and resulting cliques.

### 6.3 Graph and identity repair

Graph repair combines incoming and outgoing neighbors of the removed node. For each affected
neighbor, it merges that neighbor's candidates with the removed node's outgoing neighbors,
excludes deleted/self/out-of-range references, and applies the existing degree-limited edge
selection heuristic. It then clears the removed node's adjacency.

`move_id` moves base vectors, separately owned raw vectors, precise codes, extra information,
bottom/upper graph references, labels, and entry-point information as applicable. Label-table
handling includes soft-delete state, label reuse, and optional source IDs, so moving an old
tombstone does not hide a newer live instance with the same external label.

### 6.4 Scope and failure behavior

Only explicitly targeted slots are physically removed; other soft-deleted slots survive and
are remapped. Future Add starts at the smaller tail but may reserve capacity in growth units.

The operation is not transactional. Completed deletions, graph repairs, and moves may remain
after an error. Incomplete MCI is not republished to fast search. Allocating replacement CSR
buffers before swapping provides local replacement atomicity, not whole-operation rollback.
Old and new buffers coexist temporarily, so peak memory may increase during compaction.

## 7. Flush, search, locking, and persistence

### 7.1 Flush

`index->Flush()` merges base members, extras, and new cliques; removes retired cliques,
inactive members, and empty rows; renumbers cliques; and builds both CSR directions. It
allocates replacement buffers before publishing, then clears delta/retired contents.

Plain Flush preserves vector IDs and inactive masks. It neither rebuilds HGraph nor removes
vectors nor writes to disk. Repeat Flush preserves effective relationships, but clique IDs
must not be cached across it. Empty per-node/per-clique delta row descriptors still consume
metadata memory. `RemapNodes` shares the compaction core and additionally applies an ID map.
FORCE_REMOVE automatically flushes repair output; Add and MARK_REMOVE do not.

### 7.2 Direct traversal

Filtered queries choose MCI according to `ValidRatio()` and the route threshold:

```text
Node → inverse base CSR + node delta → skip retired cliques
Clique → base CSR + extras, or delta clique → skip inactive nodes
Candidate → user filter → label deletion set → distance and candidate queue
```

`CliqueDataCellSearchView` holds one shared lock over base, delta, and masks, avoiding
per-clique list copies and lock acquisition. Contiguous FP32 vectors use direct distances;
other encodings use their original distance interfaces with the same traversal and queue.
The `mci_raw_float_csr` statistic also includes direct FP32 traversal over delta.

A query-scoped deletion-set read view removes per-visited-node deletion-lock acquisition.
User filtering precedes deletion-set filtering. This is reduced lock frequency,
not lock-free search.

### 7.3 Concurrency

| Protection | Purpose |
| --- | --- |
| `mci_mutation_mutex_` | Serialize Add, both Remove modes, and Flush for MCI. |
| `force_remove_mutex_` | Shared by queries, exclusive for physical movement. |
| `persistent_codes_mutex_` | Protect vector-buffer lifetime during distance computation. |
| Clique shared/exclusive lock | Pin query views; protect delta writes and CSR replacement. |
| Label/deletion-set locks | Protect identities, masks, and query deletion views. |

FORCE_REMOVE orders the mutation lock before the force lock and local storage locks.
Add initially holds a shared force lock, but releases it before MCI candidate generation
enters public search. The mutation mutex still excludes force deletion, avoiding recursive
shared locking.

Add/MARK_REMOVE can temporarily unpublish MCI and let queries fall back to HGraph. Full
recall during these mutations is not guaranteed; an invalid fallback entry point can yield
empty results. FORCE_REMOVE blocks queries through movement and repair. Flush keeps MCI
published but readers may wait for its exclusive CSR lock. These are not cross-batch
transaction or universal snapshot-isolation guarantees.

### 7.4 Persistence

Base CSR, delta, and deletion-related state are serialized with HGraph. Force-enabled MCI
restoration reconstructs label deletion state/counts from inactive masks, checks the physical
range, and preserves live mappings when a label also has an older tombstone slot. Flush is
not Serialize, and saving base CSR alone does not preserve all incremental state.

## 8. Memory accounting and 128 MiB blocks

| CSV field | Meaning |
| --- | --- |
| `index_memory_bytes` | Allocated index capacity reported by `GetMemoryUsage()`, not RSS. |
| `vector_memory_bytes` | `basic_flatten_codes` capacity for this FP32 workload. |
| `graph_memory_bytes` | Bottom plus route graph accounting, including related graph metadata. |
| `mci_memory_bytes` | MCI arrays, delta rows, and masks measured by capacity. |
| `stage_rss_bytes` | External process RSS sampled near the end of a stage's queries. |

Index totals also contain labels, locks, and other structures, so they do not equal the sum
of the three component columns. The vector column is not a universal total for every raw,
quantized, or refinement-vector configuration. MiB means bytes divided by `2^20`.

Vectors use `memory_io`; the graph retains default `block_memory_io`. Its default block is
128 MiB, accounting is block count times block size, and Shrink releases only whole blocks.
Both 10k and 8k nodes still require a block. This allocation floor masks vector shrinkage;
it is not merely an operating-system RSS-return issue.

Large datasets can cross block-release boundaries. However, graph repair changes reverse
edges and container capacities, so graph accounting can increase in an individual delete
stage. The same restored point count need not recreate initial graph capacity. RSS also
includes the full input dataset, query filters, scratch buffers, and allocator-retained pages.

## 9. Test protocol and reproduction

### 9.1 Environment and parameters

Measurements were collected on 2026-09-07 using a Release build on Intel Xeon Platinum 8269CY,
52 cores/104 logical CPUs, approximately 754 GiB RAM. Build and search each use 16 threads.

| Parameter | Value |
| --- | --- |
| Data / distance | float32, cosine, dimension 384 |
| Graph | HGraph NSW, max_degree=32, ef_construction=200 |
| Vector / graph IO | memory_io / default block_memory_io |
| MCI candidates | mcs=50, knng_source=hgraph |
| Clique caps / alpha | initial and incremental caps 50, alpha=1.2 |
| Historical join policy | ratio threshold 0.6, at most 3 cliques (superseded by degree targets) |
| Delete thresholds | clique_size=3, node_mct=3 |
| Queries | top-k=10, 200 queries, ef=40/80/160 |
| Timed searches | 50,000 per ef, cycling over those 200 queries |
| Warmup / seeds | 64 warmup searches, mci_seed_ratio=0.1 |
| Route / random seed | route_threshold=1.0, random_seed=20260907 |
| Flush | No extra Flush; FORCE_REMOVE internally compacts its output |

These are single trials, not medians of repeated runs or confidence intervals. Repeated
queries do not represent 50,000 distinct queries or cold-cache throughput.

### 9.2 Dataset and stages

The HDF5 dataset is loaded into memory before full Build, without `--max-base` truncation.
Queries filter on matching train/test labels. The full 3m file has 1,278 test queries; this
run uses 200. The 10k run recomputed exact filtered ground truth; the 3m run used provided
neighbors after range/label checks.

Deletion candidates exclude the selected queries' ground-truth top-k points, preserving a
common ground truth across stages. Remaining candidates are shuffled with a fixed seed.
This is not unrestricted random deletion. Add restores the same vectors and external IDs.

| Stage | Operation | 10k live nodes | 3m live nodes |
| --- | --- | ---: | ---: |
| S0 | Full Build | 10,000 | 3,241,378 |
| S1 | Delete approximately 10% of original data | 9,000 | 2,917,240 |
| S2 | Delete another approximately 10% of original data | 8,000 | 2,593,102 |
| S3 | Restore first batch | 9,000 | 2,917,240 |
| S4 | Restore second batch | 10,000 | 3,241,378 |

Each 3m stage mutates 324,138 nodes, the rounded 10% of the original size, not 10% of the
remaining size. Each 10k API call mutates 10 nodes; each 3m call mutates the entire stage.
This changes compaction frequency and repair decisions, so the runs are not a controlled
mutation-latency scaling experiment.

10k compares independently built FORCE_REMOVE and MARK_REMOVE indexes. Force enables
reverse edges and starts with a different memory footprint; prefer within-run comparisons.
3m measures FORCE_REMOVE only, without MARK_REMOVE or extra-Flush controls.

### 9.3 Reproduction

Run from the repository root. The wrapper builds the Release benchmark by default;
`MCI_SKIP_BUILD=1` reuses a binary only if it already matches the current source.







These commands reproduce benchmark metrics. `stage_rss_bytes` is an externally added column,
not emitted by the C++ benchmark. 3m sampled RSS every 100 ms and used the median over roughly
the final second of ef=160. 10k sampled roughly every 5 ms and used the final 200 ms median.
No allocator trim was forced. Sampled peak RSS can miss short-lived allocation peaks.

## 10. Results

### 10.1 3m allocated memory

| Stage | Physical nodes | Index MiB | Vector MiB | Graph MiB | MCI MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 3,241,378 | 6,323.25 | 4,749.00 | 926.83 | 349.19 |
| S1 | 2,917,240 | 5,692.63 | 4,273.30 | 862.39 | 262.76 |
| S2 | 2,593,102 | 5,215.30 | 3,798.49 | 900.01 | 229.55 |
| S3 | 2,917,240 | 5,762.74 | 4,273.50 | 905.89 | 321.77 |
| S4 | 3,241,378 | 6,411.62 | 4,749.00 | 1,043.43 | 332.09 |

S2 releases **1,107.94 MiB (17.52%)** of index accounting relative to S0. Removed vector
payload is approximately 949.62 MiB; vector capacity decreases by approximately 950.51 MiB
because initial spare capacity also shrinks.

S4 restores the original count and 4,749.00 MiB vector capacity. Index memory is approximately
88.37 MiB higher, graph accounting 116.60 MiB higher, and MCI 17.10 MiB lower than S0.
The remaining difference primarily resides in graph and other structures, not accumulated
copies of the physically removed vectors.

### 10.2 3m RSS, mutation time, and delta

| Stage | RSS MiB | Mutation seconds | Total cliques | New delta cliques |
| --- | ---: | ---: | ---: | ---: |
| S0 | 12,301.81 | — | 408,484 | 0 |
| S1 | 11,697.05 | 393.30 | 408,424 | 0 |
| S2 | 11,210.89 | 338.56 | 408,364 | 0 |
| S3 | 11,709.60 | 551.62 | 523,628 | 115,264 |
| S4 | 12,353.88 | 513.30 | 601,107 | 192,743 |

Build took 593.99 seconds; total wall time was 2,572.34 seconds, approximately 42.9 minutes.
Sampled peak RSS was 13,088.96 MiB. Live counts, physical counts, and covered counts matched
at every stage; all inactive counts were zero. New delta-clique counts exclude base-clique
extras and do not measure the entire delta workload.

The first deletion and first Add each included one short read-only debugger attachment;
their mutation wall times include these pauses. No debugger was attached during timed
queries. Samples showed graph adjacency repair and Add's MCI candidate KNN search,
respectively. Hardware perf sampling was unavailable due to permissions, so these are not
complete CPU time-share profiles.

### 10.3 Complete 3m QPS-recall@10 sweep

| Stage | ef_search | QPS | Recall@10 |
| --- | ---: | ---: | ---: |
| S0 | 40 | 10,030.73 | 84.90% |
| S0 | 80 | 6,540.38 | 88.90% |
| S0 | 160 | 4,004.39 | 92.25% |
| S1 | 40 | 10,369.07 | 84.30% |
| S1 | 80 | 6,988.16 | 89.65% |
| S1 | 160 | 4,524.68 | 92.50% |
| S2 | 40 | 12,228.89 | 84.15% |
| S2 | 80 | 8,140.06 | 88.75% |
| S2 | 160 | 5,222.28 | 92.35% |
| S3 | 40 | 12,238.82 | 84.25% |
| S3 | 80 | 7,897.30 | 88.75% |
| S3 | 160 | 5,006.40 | 92.65% |
| S4 | 40 | 12,308.17 | 84.10% |
| S4 | 80 | 8,294.05 | 87.80% |
| S4 | 160 | 5,345.38 | 90.90% |

All 15 points used MCI and direct FP32 traversal, including nonempty delta stages. This
does not prove delta traversal has zero overhead. S4 recall is lower than S0 by 1.10
percentage points at ef=80 and 1.35 points at ef=160. Denser ef sweeps and repeated trials
are needed to establish performance at equal recall.

### 10.4 10k FORCE_REMOVE versus MARK_REMOVE

| Stage | FORCE slots | MARK slots | FORCE index MiB | MARK index MiB |
| --- | ---: | ---: | ---: | ---: |
| S0 | 10,000 | 10,000 | 146.57 | 145.34 |
| S1 | 9,000 | 10,000 | 144.65 | 145.36 |
| S2 | 8,000 | 10,000 | 143.14 | 145.36 |
| S3 | 9,000 | 11,000 | 145.22 | 147.24 |
| S4 | 10,000 | 12,000 | 146.87 | 148.84 |

Both runs have live counts 10k → 9k → 8k → 9k → 10k, but their physical counts differ.

| Stage | FORCE vector MiB | MARK vector MiB | FORCE RSS MiB | MARK RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| S0 | 15.00 | 15.00 | 205.46 | 202.34 |
| S1 | 13.18 | 15.00 | 205.51 | 202.49 |
| S2 | 11.72 | 15.00 | 204.09 | 202.49 |
| S3 | 13.50 | 16.50 | 205.51 | 205.72 |
| S4 | 15.00 | 18.00 | 207.69 | 206.79 |

FORCE graph accounting changes from 129.33 MiB at S0 to 129.62 MiB at S2; at least one
128 MiB block remains. Total accounting falls only approximately 3.43 MiB even though
vectors have been physically removed and shrunk.

| Stage | FORCE QPS | FORCE recall | MARK QPS | MARK recall |
| --- | ---: | ---: | ---: | ---: |
| S0 | 59,700.36 | 99.15% | 59,658.00 | 98.85% |
| S1 | 64,371.72 | 97.45% | 59,928.16 | 98.85% |
| S2 | 69,515.67 | 97.75% | 60,476.66 | 97.00% |
| S3 | 67,242.11 | 97.45% | 55,990.72 | 98.85% |
| S4 | 66,083.50 | 99.15% | 58,258.47 | 98.85% |

This table uses ef=80. Archived CSV includes ef=40/80/160. Physical ID permutations, seed
selection, and clique changes can affect approximate traversal; equal recall between modes
is not guaranteed.

## 11. Validation and remaining work

Implementation validation recorded 81 related unit tests and 8,657,906 passing assertions:

- FP32/INT8, vector distances and labels after movement.
- Duplicate/missing IDs, mixed soft/physical removal, label reuse, delete-all and re-add.
- Remapping base/extras/delta/masks, serialization, and count restoration.
- Queries concurrent with FORCE_REMOVE/Add/Flush, and CSR replacement allocation failures.
- Deleted entry points, moved tombstones, and logical counts after storage shrinkage.

```bash
./build/tests/unittests \
    '[ut][hgraph],[ut][MCISearcher],[ut][LabelTable],[ut][HGraphParameter],[ut][FlattenDataCell]'
```

An earlier force-removal suite also passed 15 repeated executions. Relevant clang-format-15
and targeted clang-tidy-15 checks passed. These are implementation-stage records; this
documentation-only change did not rerun the entire C++ suite. Full coverage and the full
functional suite were not measured, so the repository's 90% coverage requirement is not
claimed as verified.

Future work, not implemented in this report:

1. Separate parallelizable candidate generation from ordered incremental Add publication.
2. Evaluate partial top-k selection, batched exact distance, or approximate deletion-repair
   candidates; revalidate quality rather than assuming equivalent behavior.
3. Compact graph tail blocks and reverse-edge containers to reduce retained capacity.
4. Add equal-recall comparisons, more queries, repeated trials, 3m MARK/Flush controls,
   and unrestricted deletion with recomputed ground truth.
5. Define and implement stronger connectivity, minimum post-repair coverage, or error
   recovery guarantees separately if required.

## 12. Archived results and source map

This report retains the measurement summaries; raw CSV result files are not included in this PR.

Original continuous RSS, logs, and local helper scripts are in
`/tmp/mci-force-3m-wO5WR3/` and `/tmp/mci-force-verify-5G5UPQ/final/`.
These temporary machine-local paths are not portable artifacts. The prior interactive
chart's numbers were checked against CSV, but browser preview could not be verified because
dependency downloads timed out. Tables and archived CSV remain independently readable.

| File | Responsibility |
| --- | --- |
| [hgraph_build.cpp][src-build] | Build/Add batches and MCI publication. |
| [hgraph_mci.cpp][src-mci] | Candidates, joining/building cliques, repair, force removal, Flush. |
| [hgraph_modify.cpp][src-modify] | Remove dispatch, edge repair, tail movement, shrinking. |
| [clique_datacell.cpp][src-cell] | Delta, deletion snapshots/commit, bidirectional compaction. |
| [clique_datacell.h][src-view] | Pinned base/delta/mask views and traversal. |
| [mci_searcher.cpp][src-search] | Direct distances and candidate queue. |
| [label_table.cpp][src-label] | Label moves and deletion-state restoration. |
| [hgraph_serialize.cpp][src-serialize] | Persistence and component memory accounting. |
| [memory_block_io.cpp][src-block] | Whole-block allocation and shrinking. |
| `mci_mutation_benchmark.cpp` | Loading, ground truth, five stages, CSV. |

[src-build]: ../../../../../src/algorithm/hgraph/hgraph_build.cpp
[src-mci]: ../../../../../src/algorithm/hgraph/hgraph_mci.cpp
[src-modify]: ../../../../../src/algorithm/hgraph/hgraph_modify.cpp
[src-cell]: ../../../../../src/datacell/clique_datacell.cpp
[src-view]: ../../../../../src/datacell/clique_datacell.h
[src-search]: ../../../../../src/impl/searcher/mci_searcher.cpp
[src-label]: ../../../../../src/impl/label_table/label_table.cpp
[src-serialize]: ../../../../../src/algorithm/hgraph/hgraph_serialize.cpp
[src-block]: ../../../../../src/io/memory_block_io/memory_block_io.cpp

## 13. ADD / Delete threshold sweep

Use `sweep_mci_thresholds.py` to test existing thresholds without changing the
index algorithm. Start with one-factor-at-a-time (OAT) experiments on 10k, rather than a
full parameter grid on 3m.

| Parameter | Baseline | Default candidates |
| --- | ---: | --- |
| `join_ratio` | 0.6 | 0.4, 0.6, 0.8 |
| Historical maximum joined cliques (removed) | 3 | 1, 3, 6 |
| `clique_max` (incremental only) | 50 | 25, 50, 100 |
| `delete_size` | 3 | 3, 4, 5, 6 |
| `delete_mct` | 3 | 3, 5, 8 |

Cliques are retired when their size is strictly below the deletion threshold.
Increase the deletion size threshold from baseline 3 in steps of one: 4, 5, 6, not 20/40.
Initial 10k cliques can have approximately 50 members, so these thresholds may never
activate small-clique repair; report unchanged results as such.
Increasing `delete_mct` may have no effect unless a small clique is retired; subsequent
grid experiments are needed to test this interaction.

The incremental size cap is independent of the full-build `mci_clique_max=50`, keeping
the initial build configuration fixed. `mci_mcs` remains 50: an incremental cap of 100
does not guarantee 100-member cliques. Candidate counts and appending to existing cliques
also constrain realized sizes. The historical maximum-joined-cliques sweep is no longer
supported; current ADD uses the degree target. `delete_mct` gates repair, not minimum degree.

### 13.1 Running experiments

1. The baseline plus non-baseline values on each axis produces 12 configurations.
   Three repetitions per configuration produce 36 independent runs.
2. Each run executes all five stages at ef=40/80/160/320: 720 measurements in total.
3. Configurations share the deletion random seed within a repetition; seeds increase
   between repetitions.
4. Configuration order is deterministically shuffled per repetition. Runs are sequential
   to avoid competing for CPU resources.
5. Defaults use one build thread to reduce initial-build variation and 16 search threads.
   This does not guarantee byte-identical indexes or seed HGraph construction with the
   deletion random seed.
6. Defaults use mutation batches of 1,000, 200 recall queries, and 10,000 timed searches
   per ef value.

Build threads, mutation batches, and query counts differ from the historical experiments
in section 10. Compare against this sweep's own baseline, not historical QPS.
Filtering and ground-truth protection follow the existing benchmark. Search seed sets
are not held fixed across stages.

Build the current benchmark, inspect the plan, then run:



The script does not build automatically; it checks that the binary supports the new
threshold options. Omitting the output directory creates a unique temporary directory.
Existing nonempty directories are not overwritten. Each run has a 600-second timeout;
failures preserve logs, remaining configurations continue, and the script exits nonzero.

To isolate deletion size thresholds 3/4/5/6 with other parameters fixed
(4 configurations, 12 runs):



After OAT experiments, explicitly narrow the ranges for an interaction grid:



The original baseline is retained even outside the grid, with duplicates removed.
This example has 25 configurations and 75 runs, also exceeding the default run limit.
The full default grid has 324 configurations and 972 runs, exceeding `--max-runs=64`.
Inspect the plan before explicitly increasing this limit. There is no automatic
"best configuration" selection.

Validate only shortlisted configurations on 3m, with full mutation batches and a longer
timeout, for example:



This is a reproduction command, not a claim that this 3m threshold comparison was run.
Configuration names for `--only` come from `--dry-run`. Removal defaults to FORCE_REMOVE;
use `--remove-mode mark` for logical deletion. Treat `--flush-after-mutation` as a separate
experiment, not samples to merge with unflushed runs.

### 13.2 Outputs and interpretation

| File | Contents |
| --- | --- |
| `manifest.json` | Experiment plan, dataset identity, binary and dynamic VSAG fingerprints. |
| `config/repeat-N/attempt-N/` | Raw CSV, complete command, logs, success/failure status. |
| `all.csv` | Successful measurements, with recall changes paired to each run's initial stage. |
| `summary.csv` | Medians by configuration/stage/ef, recall/QPS ranges, sample counts. |
| `target-recall.csv` | Fastest measured QPS meeting each target; no interpolation/extrapolation. |
| `failures.json`, `status.json` | Failure details, completed counts, planned total. |

Target-recall QPS is filled only when every planned repetition succeeds and meets the
target. Otherwise it is blank; higher QPS at lower recall is not a substitute.
At a fixed ef, `recall_delta_pp` is calculated against each run's own initial recall
before aggregation, in percentage points.

New CSV fields include `avg_dist_cmp`, `avg_hops`, `avg_seed_count`, average clique size,
total memberships, and delta-extra memberships. The first three come from recall queries
outside the timed QPS loop. MCI hops here count visited cliques, not shortest-path length.
`memberships_per_live_node` uses live nodes, avoiding confusion with MARK's physical slots.
These metrics summarize search work and structure; they do not prove connectivity or
ground-truth reachability.

Validation checks five-stage live/physical counts, full MCI coverage, deletion masks,
100% fast-path use, echoed thresholds, and measurement completeness. Invalid runs are
excluded from aggregates. Resume an interrupted experiment with identical parameters:



Only validated measurements with a completion marker are skipped. Incomplete attempts
are preserved and retried in a new attempt directory. Changed parameters, dataset
identity, or binary fingerprints reject resume, preventing mixed experiments.

Script tests:




## 14. Random churn from an 800k initial index

`run_mci_stress.py` uses a separate stress mode without the five-stage
benchmark's ground-truth protection. From the 3,241,378-vector pool, sample 800,000 IDs
without replacement and fully build the initial index. Each round samples
`round(3,241,378 / 14) = 231,527` IDs without replacement from the complete pool:

- FORCE_REMOVE IDs present at the start of the round; ADD IDs absent at its start.
- IDs are unique within a round but can reappear across rounds. These are not 14 partitions.
- Delete selected live IDs first, then add selected absent IDs; measure both checkpoints.
- Defaults run 14 rounds, usually giving 29 checkpoints and 116 ef measurements.
- Only the initial count is 800k; subsequent membership and live counts change.

This historical run used deletion size threshold 3, ADD join ratio=0.6, at most 3 joined
cliques and clique max=50. Current ADD uses the configurable degree target instead.
Build/search use 16 threads, the first 200 fixed queries, ef=40/80/160/320, and 10,000
timed searches per ef. Mutation batch size equals the round sample size.



The dataset is loaded into memory. Complete exact label-filtered distance rankings are
cached for the fixed query cohort. At each checkpoint, selecting the first live top-k
IDs from each ranking is equivalent to recomputing exact neighbors on that live set.
Ground-truth neighbors are not protected; original full-pool top-k lists are not used
to evaluate a subset. A fixed query with fewer than top-k live matches explicitly fails,
rather than silently changing the query cohort. Dataset and ranking caches are benchmark
overhead, excluded from index accounting; plotted index memory is not process RSS.
Since the live dataset changes, recall drift cannot be attributed solely to index degradation.

Outputs:

- `statistics.png`: live counts, recall, QPS, index/component memory, cliques, mutation times.
- `qps-recall.png`: initial and up to four evenly sampled subsequent checkpoints.
- `curve.csv`: all checkpoint/ef measurements; `.events.csv` records operations and truth costs.
- `curve.csv.ids.csv`: initial members and every operation's IDs for sampling/state audits.
- `curve.csv.truth.csv`: exact ground-truth IDs per checkpoint and query.
- `manifest.json`, `benchmark.log`, `status.json`: parameters, logs, final completion status.

Plots refresh after each checkpoint passes search and index count/coverage checks;
in-progress plots are partial results. Nonempty output directories are rejected.
There is no resume from an intermediate index; failures preserve logs and completed curves.
To redraw existing measurements or run synthetic integration tests:



Optional `--mode alternate` adds from absent IDs on odd rounds and removes from live IDs
on even rounds, using 231,527 IDs each round. This is a different workload; do not mix
its results with toggle runs. Synthetic tests independently verify live-set exact truth,
ID existence, sample counts, and seeded reproducibility. They do not establish completion
of the 800k experiment; check its output `status.json`.


## 15. FP32 five-stage rerun and initial-index snapshots

The completed 2026-09-08 run used 3,241,378 vectors, FP32, 16 build/search threads,
and FORCE_REMOVE batches of 324,138 vectors. At ef_search=320:

| Stage | Live vectors | Recall@10 | QPS | Mutation seconds | Index MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 3,241,378 | 94.50% | 2,691.89 | 0 | 6,323.55 |
| Delete 10% | 2,917,240 | 94.75% | 2,863.30 | 349.26 | 5,684.00 |
| Delete 20% | 2,593,102 | 95.80% | 3,255.91 | 337.23 | 5,232.58 |
| Add back 10% | 2,917,240 | 96.35% | 3,277.14 | 534.94 | 5,783.94 |
| Add back 20% | 3,241,378 | 94.65% | 3,346.91 | 528.11 | 6,432.85 |

Initial construction took 553.47 seconds. Index MiB is reported index allocation, **not
process RSS**; per-stage RSS was not recorded. Raw result files are not included in this PR.
These are measurements of the development implementation before its adaptation to the newer
main branch, not a rerun of the PR rebased onto main. Higher QPS at a fixed ef does not alone
establish an equal-recall performance improvement.

After switching FP32 deletion repair to HGraph KNN, run:



The detached runner snapshots the existing Release executable and libvsag. Live counts follow
100%, 90%, 80%, 90%, 100%, with each mutation batch equal to 10% of the initial population.
It measures ef=40/80/160/320, recall on 200 queries, and QPS on 10,000 queries. Construction
and search use the requested threads; deletion batches and MCI repair remain sequential.
Deletion protects evaluation top-k neighbors, matching the historical five-stage test rather
than the unprotected random-churn workload. Ensure no other performance jobs contend for
resources; this runner does not automatically queue.

Before any deletion, it saves `initial.index` with `initial.index.json` recording the dataset
path, size, modification time, count, and build parameters. The benchmark executable accepts
`--save-initial-index` outside stress mode and refuses to overwrite snapshots. The file uses
the existing `Serialize(std::ostream&)` format. Keep both files together; serialization is
excluded from build/mutation timing.

Historical preflight found a memory error after loading a 400-point FP32 index through either
ostream or BinarySet, so the performance run above built afresh. Review follow-up identified
and fixed missing reverse-edge restoration: tail moves left stale incoming references, and
GetStats degree counting subsequently corrupted the heap. Flat bottom graphs and sparse upper
graphs now rebuild reverse edges from decoded, valid forward edges without changing the wire
format. New regressions cover both serialization interfaces, both memory IO layouts, and two
deletions followed by two additions and searches. **Benchmark load/reuse remains unexposed**;
these tests are not a new performance run using the saved 3m snapshot.

Monitor `status.json`, `benchmark.log`, and `curve.csv`. Completion validates all 20 stage/ef
measurements and generates `qps-recall.png`.
