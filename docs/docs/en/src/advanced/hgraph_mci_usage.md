# HGraph MCI: code and configuration

This guide covers the development branch's FP32 Build, ADD, MARK_REMOVE, FORCE_REMOVE,
and Flush workflows. The index type is `hgraph`; MCI shares its vector storage and is not
a separate HNSW index. See the [mutation report](hgraph_mci_mutation.md) for algorithms and measurements.

> The reproduced reload-then-FORCE_REMOVE memory error was traced to missing graph reverse-edge
> restoration and has a fix and regression coverage. Historical measurements used a fresh Build;
> no new large-dataset snapshot-reuse benchmark is claimed.
> The new virtual `Index::Flush()` requires ABI review; MCI serialization advances to v2.
> Compatibility with older binaries must be tested separately.

## 1. Code map and maintenance

Paths below are relative to the repository root.

| File | Entry points / responsibility |
| --- | --- |
| `include/vsag/index.h`, `src/index/index_impl.h` | Public Build/Add/Remove/Flush/Search and error wrapping |
| `src/algorithm/hgraph/hgraph_build.cpp` | `Add` → `add_impl`: insert into HGraph, then maintain MCI for successful insertions |
| `src/algorithm/hgraph/hgraph_mci.cpp` | Full construction, `search_mci_knn`, `incremental_update_mci_clique`, `repair_mci_clique`, `force_remove_with_mci`, `Flush` |
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

1. Insert an ADD batch into HGraph, then maintain successful points. `search_mci_knn` calls
   HGraph `KnnSearch` with `use_mci=false`, not MCI search.
2. Target `min(mci_mcs, visible_total - 1)` neighbors. Internal ef is `max(query_k, 100)`,
   not benchmark ef=320. Remove self, deleted, and out-of-range points; enlarge the request if needed.
3. Try existing cliques with `|KNN ∩ C| / |C| >= join_ratio` and room below the incremental cap,
   selecting at most `added_mct`. Build a new clique only if none was joined; empty candidates yield a singleton.
4. Deletion retires only affected cliques whose surviving size is **less than** `delete_size`.
   Select surviving members whose projected effective coverage is **less than** `delete_mct`;
   recheck coverage before repairing.
5. Pure-FP32 repair reads/decodes the existing point and uses the shared candidate/join/build path.
   It does not insert the vector again through public ADD. ADD sees its insertion prefix;
   repair can see the whole current HGraph.

`added_mct` is an upper bound on joins, while `delete_mct` triggers repair; neither guarantees
that every point ultimately belongs to that many cliques. Non-FP32 repair retains pair-distance
candidate generation; FP32 findings do not establish RaBitQ behavior.

| Operation | Vector storage | MCI work | Persistence |
| --- | --- | --- | --- |
| MARK_REMOVE, default | Retain physical slots | Retire small cliques, repair under-covered points into delta | None |
| FORCE_REMOVE | Move tail points into holes; reduce slots and attempt shrinking | Snapshot, remap, repair, automatic Flush | None |
| Flush | No vector deletion/movement | Merge delta, remove retired memberships, rebuild CSR | None |

MCI mutations share a serialization mutex. Physical moves and final shrinking hold exclusive
force-remove protection. Repair releases that lock so internal HGraph queries can acquire read
locks; MCI is unpublished and external searches may fall back to HGraph. Queries are not
necessarily blocked for the entire operation. Mutations are not transactionally rolled back.

## 2. Configuration and C++ integration

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
    "mci_mcs": 50,
    "mci_clique_max": 50,
    "mci_alpha": 1.2,
    "mci_incremental_join_ratio_threshold": 0.6,
    "mci_incremental_added_mct": 3,
    "mci_incremental_clique_max": 50,
    "mci_delete_clique_size_threshold": 3,
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
| `mci_incremental_added_mct` | 3 | `--mci-incremental-added-mct`; positive join limit |
| `mci_incremental_clique_max` | 50 | `--mci-incremental-clique-max`; at least 2; benchmark inherits the full-build cap when omitted |
| `mci_delete_clique_size_threshold` | 3 | `--mci-delete-clique-size-threshold`; positive, strict less-than retirement |
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
check(index->Flush());
auto result = index->KnnSearch(query, 10, search_json, filter);
check(result);
```

Build/Add also return failed-insertion labels: checking only `expected` is insufficient.
`removed.value()` is the actual removal count. MARK_REMOVE retains physical slots.
Explicit Flush is optional; FORCE_REMOVE already flushes internally. Flush does not serialize.
Remove accepts external labels, not inner slots; do not retain internal node/clique IDs across compaction.
See `examples/cpp/324_feature_hgraph_mci_companion.cpp` for Dataset and Filter construction.

## 3. Benchmark scope

Development benchmark sources and scripts are excluded from this PR. Historical results
remain in the mutation report; these tools are not available in this checkout.

## 4. Reading results and validating

Five-stage runs protect evaluation top-k neighbors and retain the same truth. Random churn does
not protect neighbors; each checkpoint selects live top-k from exact full-pool distance rankings.
Do not mix their recall conclusions.

| CSV fields | Meaning |
| --- | --- |
| `stage`, `active_vectors`, `index_elements` | Stage, live count, public element count; the last is not physical storage accounting |
| `ef_search`, `recall_at_k`, `qps` | Search width, recall, timed throughput; fixed ef is not fixed quality |
| `build_seconds`, `mutation_seconds`, `flush_seconds` | Build, stage mutation, explicit post-stage Flush; internal FORCE_REMOVE Flush is part of mutation time |
| `index_memory_bytes`, `vector_memory_bytes`, `graph_memory_bytes`, `mci_memory_bytes` | Index/component accounting, not RSS |
| `mci_route_ratio`, `mci_raw_float_ratio` | Actual MCI/direct FP32 route rates |
| `mci_total_cliques`, `mci_delta_cliques`, `mci_total_memberships` | Clique, delta-clique, and membership counts |
| `avg_dist_cmp`, `avg_hops`, `avg_seed_count` | Search work useful for QPS diagnosis |

MiB = bytes / 1,048,576. Stage memory/timing repeats across ef rows; do not sum it. QPS is
multithreaded throughput: `1000 / QPS` is not per-request latency. CSV has neither latency percentiles
nor per-stage RSS. Historical five-stage measurements are summarized in report section 15;
raw result files are not included in this PR. These measurements predate adaptation to newer
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
