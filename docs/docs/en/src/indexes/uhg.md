# UHG

UHG (Unified Hybrid Graph) retrieves paired dense and sparse representations under a query-specific weight. Create it with `Factory::CreateIndex("uhg", parameters)` and provide both `Float32Vectors` and `SparseVectors` in the same Dataset. Both representations share the external document IDs.

The returned distance is `alpha * (1 - dense_dot) + (1 - alpha) * (1 - sparse_dot)`. Smaller distances rank first. Normalize dense embeddings before ingestion if cosine-equivalent dense distances are desired; the index does not normalize either representation or calibrate their score scales.

## Construction

```json
{
  "dtype": "float32",
  "metric_type": "ip",
  "dim": 1024,
  "index_param": {
    "max_degree": 64,
    "neighbors_per_alpha": 32,
    "construction_candidates": 200,
    "alpha_intervals": 10,
    "hgraph": {"max_degree": 64, "ef_construction": 200},
    "sindi": {"term_id_limit": 500000, "window_size": 10000}
  }
}
```

HGraph and SINDI generate candidate pools offline. For every document, UHG merges the candidates, evaluates their exact component distances, selects the best `neighbors_per_alpha` neighbors at each weight `a / alpha_intervals` for `a=0,...,alpha_intervals`, and retains at most `max_degree` neighbors with the highest selection frequencies. Frequency ties are resolved by external ID. Each guide retrieves up to `construction_candidates + 1` documents, including the center if returned. The graph remains fixed across query weights.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_degree` | 64 | Maximum UHG outgoing degree |
| `neighbors_per_alpha` | 32 | Neighbors selected per sampled weight |
| `construction_candidates` | 200 | Candidate count per guide, excluding the center allowance; must be at least `neighbors_per_alpha` |
| `alpha_intervals` | 10 | Sampling intervals; 10 produces weights 0, 0.1, ..., 1; range 1–1000 |
| `hgraph` | `{}` | Dense guide options: `max_degree`, `ef_construction`, `build_thread_count` |
| `sindi` | `{}` | Sparse guide options: `term_id_limit`, `window_size`, `remap_term_ids` |

HGraph and SINDI remain guide indexes. UHG additionally stores FP32 dense and sparse forward vectors for batched graph distances, trading storage for avoiding per-node public API calls and inverted-list point lookups. It also stores adjacency lists and sparse norms. Exact FP32 component distances are required: HGraph uses `base_quantization_type="fp32"`, SINDI uses `use_quantization=false` and `doc_prune_ratio=0`, and both use `use_reorder=false`. Conflicting values are rejected.

## Search

```cpp
auto index = vsag::Factory::CreateIndex("uhg", build_parameters).value();
auto base = vsag::Dataset::Make()
                ->Owner(false)
                ->NumElements(count)
                ->Dim(1024)
                ->Ids(ids)
                ->Float32Vectors(dense_vectors)
                ->SparseVectors(sparse_vectors);
auto built = index->Build(base);
if (!built.has_value()) {
    throw std::runtime_error(built.error().message);
}
auto query = vsag::Dataset::Make()
                 ->Owner(false)
                 ->NumElements(1)
                 ->Dim(1024)
                 ->Float32Vectors(query_dense)
                 ->SparseVectors(&query_sparse);
auto result = index->KnnSearch(query, 10, R"({
    "uhg": {
        "alpha": 0.3,
        "ef_search": 300,
        "entry_points": 8,
        "guide_ef_search": 100,
        "use_pruning": true,
        "pruning_scale": 1.0
    }
})");
```

For `alpha <= 0.5`, SINDI supplies entry points; for `alpha > 0.5`, HGraph supplies entry points. UHG then runs beam search using the requested hybrid distance. Empty sparse queries use the dense guide; a guide with no matching entries falls back to the first graph vertex. Guide retrieval does not truncate the final hybrid candidates: traversal may discover documents absent from the guide results.

| Query parameter | Default | Meaning |
| --- | --- | --- |
| `alpha` | 0.5 | Dense weight in [0,1], independently chosen per query |
| `ef_search` | 100 | Traversal beam size; must be at least requested k |
| `entry_points` | 8 | Number of guide results used to initialize traversal, capped by beam size |
| `guide_ef_search` | 100 | HGraph beam or SINDI candidate budget, raised to at least the entry count |
| `use_pruning` | true | Enable dense-first sparse-distance pruning |
| `pruning_scale` | 1.0 | Gamma in [0,1]; 1 uses the Cauchy–Schwarz lower bound; smaller values are heuristic and may reduce recall |

The pruning estimate is `alpha * dense_distance + (1-alpha) * (1-gamma*query_sparse_norm*document_sparse_norm)`. A float32 rounding margin is applied before pruning. Exact sparse evaluation is skipped only after the traversal beam is full and this estimate exceeds its worst distance. At alpha=0, dense distance calculation is skipped; at alpha=1, sparse distance calculation is skipped. Setting `use_pruning=false` disables this estimate test.

## Supported operations and limits

This implementation supports Build on an empty index, single-query KNN, external-ID filters, concurrent reads, ID export, memory usage, and BinarySet/stream/ReaderSet serialization. Load with the same construction parameters and dimension. Build and load publish state only after the complete operation succeeds. It does not support Add, Remove, UpdateId/UpdateVector, range search, quantized vectors, extra info, or attribute filters.

Base sparse vectors must be nonempty, with strictly increasing unique term IDs and finite nonnegative FP32 values. Empty sparse queries are accepted. Dense values must be finite. Base IDs must be unique and cannot be -1. Filtering keeps excluded vertices available for traversal, but highly selective filters may return fewer than k matches; increasing `ef_search` can improve coverage.

Recall depends on the offline candidate pool, graph degree, query beam, and pruning scale. No full-dataset exact-search fallback is used. The paper's reported speedups are not performance guarantees for this implementation; benchmark with paired embeddings and exact hybrid ground truth for the query weights used in production.

## MS MARCO comparison benchmark

The standalone `tools/uhg_benchmark` executable compares production UHG search with equal-budget HGraph + SINDI retrieval. Each route retrieves 100, 150, 200, 300, or 500 candidates. The baseline unions the returned IDs, calculates both exact component distances for every candidate, and reranks the union to top-100. Timings include retrieval, deduplication, and complete reranking. UHG uses beam widths 100, 200, 300, 500, and 1000, with equally many guide entries and pruning scale 0.3. Both methods use unpruned sparse queries.

```sh
cmake -S tools/uhg_benchmark -B build-uhg-benchmark \
  -DVSAG_BUILD="$PWD/build-release" -DVSAG_DEPS=/tmp/vsag-deps
cmake --build build-uhg-benchmark --parallel 8
OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=64 \
  build-uhg-benchmark/uhg_benchmark \
  /data/datasets/msmarco-1024-8m-ip.hdf5 /data/results/uhg-msmarco-full 0 0 64 3
```

Build VSAG in release mode first. The benchmark additionally requires HDF5, BLAS, and OpenMP development libraries. `VSAG_DEPS` points to extracted VSAG dependency sources. The final arguments select maximum base count, maximum query count (0 means all), offline build threads, and measurement repetitions. Search runs sequentially on one thread. Fixed weights are 0.3, 0.5, and 0.7; the fourth scenario assigns those three weights approximately equally to queries and shuffles them with seed 42.

Ground truth exhaustively scores the entire selected base using FP32 dense/sparse inner products and double precision weighted scores. Recall@100 measures overlap with these exact hybrid neighbors, rather than MS MARCO relevance judgments. Index archives and ground truth are cached in the output directory. Use a new directory when changing data or parameters.

For the large dataset, the benchmark constructs the alpha-frequency graph in parallel using the same neighbor selection rule, then loads it through the production UHG deserializer. It reuses the baseline guide indexes, with offline dense beam 200 and sparse query pruning 0.9; this offline construction is separate from the public serial `UHG::Build` path. Reports distinguish guide build time, graph construction time, and query latency. `metadata.json` records the parameters; `results.json` records Recall@100, QPS, latency percentiles, and mean union size.

## Query implementation and compatibility

Each query creates component computers once, reuses neighbor-batch scratch buffers and a pooled visited list, batches dense evaluation, then evaluates sparse distances for surviving nodes. Unfiltered queries reuse the traversal beam for results; filtered queries retain a separate output heap so excluded nodes remain traversable bridges. Pruning thresholds are fixed within each batch, so heuristic-pruning recall may differ slightly from per-node evaluation.

New indexes use serialization v2 with forward vectors. Version 1 remains readable through the public-distance fallback. The benchmark upgrades cached indexes using the original data while retaining the original graph and guides. Intermediate HGraph guide distances are not reused; HGraph/SINDI source code remains unchanged.
