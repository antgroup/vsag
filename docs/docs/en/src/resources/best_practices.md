# Best Practices

This page gathers practical advice for running VSAG in production, as a companion to the
parameter reference and performance tuning guide.

## Index Selection

| Scenario | Recommended index | Rationale |
|----------|-------------------|-----------|
| Medium scale (≤ 10M), in-memory, recall/latency critical | `hgraph` | Unified high-quality graph index with multiple quantizations and Tune support |
| Compatibility with existing HNSW deployments | `hnsw` | Interface/parameters closest to hnswlib |
| Billion-scale vectors under limited memory | `diskann` | PQ in memory, full vectors on disk |
| Coarse recall / candidate layer | `ivf` | Trains once, parallelizes widely |
| Small scale, 100% precision required | `brute_force` | Exhaustive search; useful as a recall baseline |
| Multi-tenant or partitioned data | `pyramid` | Multiple subgraphs inside one index, supports tag-based retrieval |
| Sparse vectors (BM25 / SPLADE-style) | `sindi` | Dedicated sparse-vector index |

Detailed parameters: [Index Parameters](index_parameters.md).

## Build Time

- **Pick the metric first**: `l2` / `ip` / `cosine` cannot be changed after the index is built.
- **`ef_construction`**: typically 200–500. Too small hurts recall; too large slows builds.
- **`max_degree` / `M`**: typically 16–48. Larger values mean higher recall and memory.
- **Quantization**: latency-sensitive scenarios favor `sq8` or `pq`; accuracy-sensitive ones
  favor `fp32` or `fp16`.
- **Parallel builds**: use a custom `ThreadPool`
  (see `examples/cpp/203_custom_thread_pool.cpp`) to control concurrency.

## Search Time

- **`ef_search`**: commonly `topk` to `topk * 10`; do a QPS/recall grid search to settle on the
  right value.
- **Batch search**: merging multiple queries improves cache utilization; batch at the caller or
  use batch-capable examples.
- **Filter**: use the built-in `Filter` (`examples/cpp/301_feature_filter.cpp`) rather than
  post-filtering.
- **Per-search allocator**: for high-concurrency online services, use a per-thread arena
  allocator; see [Memory Management](../advanced/memory.md).

## Tuning

- Use [`Tune`](../advanced/optimizer.md) against realistic query distributions.
- Enable the [conjugate graph](../advanced/enhance_graph.md) for tail-heavy workloads.
- Treat [`eval_performance`](eval.md) as a continuous regression test.

## Deployment

- The official Docker image is the recommended starting point; see
  [Installation](../guide/installation.md).
- For production binaries, pick the distribution matching your ABI:
  `dist-pre-cxx11-abi`, `dist-cxx11-abi`, or `dist-libcxx` (see [Building](../development/building.md)).
- Enable `VSAG_ENABLE_INTEL_MKL=ON` on Intel CPUs for additional acceleration.
- For DiskANN, use NVMe SSDs and compile with `VSAG_ENABLE_LIBAIO=ON`.

## Observability

- `Index::GetMemoryUsage()` exposes runtime memory usage.
- The search path supports a custom `Logger`
  (`examples/cpp/202_custom_logger.cpp`) to integrate with your logging stack.
- `eval_performance` can write its metrics directly to InfluxDB for long-term monitoring.
