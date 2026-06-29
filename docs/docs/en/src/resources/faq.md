# FAQ

This page collects common questions that VSAG users hit while choosing indexes,
tuning performance, and integrating APIs. Follow the linked pages for details.

## Which index should I choose?

Common VSAG indexes target different workloads. Choose by data type, scale,
and recall / latency target.

`hgraph` is the default choice for dense vectors. It fits text, image, and
multimodal embeddings in online search systems that need high recall and low
latency. It supports multiple quantizers, incremental insertion, deletion,
reranking, and automatic tuning.

`ivf` fits large-scale or high-throughput workloads where memory is tight and
queries can tolerate bucket-based recall tradeoffs. It reduces scanning by
partitioning vectors into buckets.

`sindi` is for sparse vector retrieval, such as BM25, SPLADE, or BGE-M3 sparse
outputs. It only accepts `dtype: "sparse"` and primarily uses
`metric_type: "ip"`.

`pyramid` fits multi-tenant, partitioned, or tag-path workloads. It keeps
multiple subgraphs inside one index and supports tag / path based retrieval.

`brute_force` is for small datasets, functional validation, and exact-recall
baselines. It is exact, but latency and throughput usually do not scale to large
datasets.

Practical guidance:

- If you are unsure and your vectors are dense, start with `hgraph`.
- Use `sindi` for sparse vectors.
- Use `brute_force` for small datasets or recall baselines.
- Compare `ivf` when throughput and memory matter more than single-query latency.
- Consider `pyramid` when your data has clear partition, tenant, or path structure.

Related pages: [Index Overview](../indexes/README.md), [Best Practices](best_practices.md).

## Why does the same parameter set perform very differently on different datasets?

This is common in vector search. Even if the vector count, dimensionality, and
index parameters are identical, different datasets can have very different
search difficulty.

The root cause is data distribution:

- Some datasets have clear neighbor structure, so graph search reaches the right
  region quickly.
- Some datasets have ambiguous neighbor boundaries, so search must expand more
  candidates for the same recall.
- Embedding normalization, clustering, per-dimension distribution, and noise all
  affect search difficulty.

For HGraph-like graph indexes, `ef_search` is a key search-time parameter for
recall and latency. It controls how many candidates the search keeps and
expands:

- Larger `ef_search` usually improves recall.
- Larger `ef_search` usually increases per-query latency.
- When other factors are similar, query latency is often approximately linear in
  `ef_search`.

Therefore, do not compare datasets only by QPS at the same `ef_search`. A more
meaningful process is:

1. Tune `ef_search` separately on each dataset.
2. Make each dataset reach the same target recall, such as 95% or 98% recall.
3. Compare P50 / P95 / P99 latency and QPS at that target recall.

If dataset A reaches 95% recall with `ef_search = 80`, while dataset B needs
`ef_search = 300`, B being much slower is expected. It means B is harder to
search; it does not necessarily mean the index degraded.

When reporting performance, record:

- Dataset name and scale.
- Dimensionality.
- Index parameters.
- Target recall and actual recall.
- `ef_search`.
- QPS.
- P50 / P95 / P99 latency.

Related pages: [HGraph](../indexes/hgraph.md), [Evaluation Tool](eval.md).

## Why is `sq8_uniform` usually faster than `sq8`? When should I enable `use_reorder`?

Both `sq8` and `sq8_uniform` are 8-bit scalar quantizers, but they use different
scaling strategies.

`sq8` is per-dimension quantization:

- Each dimension has its own `min_i` / `max_i` / `scale_i`.
- This adapts better to each dimension's value range.
- Distance computation has to handle per-dimension scales, so the hot path is
  more complex.

`sq8_uniform` is global uniform quantization:

- All dimensions share one `min` / `max` / `scale`.
- Query and base codes can more easily be computed directly in the integer domain.
- SIMD, AVX-512, AMX, and NEON paths are more efficient.
- Distance computation can avoid per-element dequantization and per-dimension
  scale handling.

When the data distribution fits this assumption, `sq8_uniform` is often faster
than `sq8`.

Good use cases for `sq8_uniform`:

- Normalized vectors, especially `cosine` workloads.
- Dimensions have similar value ranges.
- Distance computation is the query bottleneck.
- Throughput and latency matter more than the last bit of recall.
- You can use `use_reorder` to fix coarse-ranking errors.

Less suitable cases:

- Different dimensions have very different value ranges.
- Vectors concatenate heterogeneous feature blocks.
- Some dimensions have heavy tails or strong outliers.
- You do not plan to enable reorder and are very sensitive to recall.

`use_reorder` first uses the compressed base quantizer for coarse ranking, then
reranks candidates with a higher-precision precise quantizer.

Common configuration:

```json
{
    "index_param": {
        "base_quantization_type": "sq8_uniform",
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

Enable `use_reorder` when:

- You use lossy quantizers such as `sq4`, `sq4_uniform`, `pq`, `pqfs`, or `rabitq`.
- Recall is not stable enough with `sq8` or `sq8_uniform`.
- `topk` is small but final ranking quality is important.
- You can afford an additional higher-precision representation.
- Production recall stability matters more than maximum compression.

You can skip `use_reorder` when:

- `fp32` or `fp16` already meets recall.
- `sq8_uniform` reaches recall targets without reorder.
- Memory budget is very tight.
- Latency is extremely sensitive and reranking overhead is unacceptable.

Simple guidance:

- Throughput first: try `sq8_uniform` without reorder and measure recall.
- Safer default: `sq8_uniform` + `use_reorder: true` +
  `precise_quantization_type: "fp32"`.
- Strong compression: `sq4_uniform` / `pq` / `rabitq` usually need reorder.

Related pages: [Scalar Uniform Quantization](../quantization/sq_uniform.md),
[Scalar Quantization](../quantization/sq.md).

## What are the distance semantics of `l2`, `ip`, and `cosine`?

VSAG search results are always sorted by smaller distance first. Even when the
underlying metric is inner product or cosine similarity, VSAG converts the score
into distance semantics.

Specific semantics:

- `l2` returns `L2Sqr`, the squared L2 distance.
- `ip` returns `1 - inner_product`.
- `cosine` returns `1 - cosine_similarity`.

Why does `l2` return squared distance? Squared L2 distance has the same ordering
as L2 distance, and avoiding the square root improves performance. VSAG
therefore commonly uses `L2Sqr` internally and in returned distances.

This affects `RangeSearch` radius settings:

- If you want L2 distance smaller than `2.0`, pass radius `4.0`.
- For `ip`, radius means `1 - inner_product`.
- For `cosine`, radius means `1 - cosine_similarity`.

For example, if you want cosine similarity at least `0.8`:

```text
distance = 1 - cosine_similarity
radius = 1 - 0.8 = 0.2
```

Notes:

- Different systems may return similarity or distance.
- Before comparing with another library or ground truth, confirm the scoring semantics.
- After an index is created, `metric_type` cannot be changed at search time.

Related pages: [Metric Semantics](metric_semantics.md),
[Range Search](../advanced/range_search.md).

## What is the difference between `base_quantization_type` and `precise_quantization_type`?

These two parameters control coarse storage and rerank storage.

`base_quantization_type` is the main storage quantizer:

- It stores the main vectors inside the index.
- It is used for coarse distance computation during graph search or inverted-list scanning.
- It directly affects memory usage, search speed, and coarse-ranking recall.
- Common values include `fp32`, `fp16`, `bf16`, `sq8`, `sq8_uniform`, and `pq`.

`precise_quantization_type` is the higher-precision quantizer for reranking:

- It only takes effect when `use_reorder: true`.
- It reranks coarse candidates.
- It corrects distance errors introduced by lossy quantization.
- The common choice is `fp32`; depending on memory budget, `fp16`, `bf16`, or
  `sq8` may also be used.

A useful mental model:

```text
base_quantization_type    = format used to quickly find candidates
precise_quantization_type = format used to recompute candidate distances
```

High-recall baseline:

```json
{
    "index_param": {
        "base_quantization_type": "fp32"
    }
}
```

Memory / recall tradeoff:

```json
{
    "index_param": {
        "base_quantization_type": "sq8_uniform",
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

Lower memory:

```json
{
    "index_param": {
        "base_quantization_type": "sq4_uniform",
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

More aggressive compression:

```json
{
    "index_param": {
        "base_quantization_type": "pq",
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

Setting guidance:

- Recall first, enough memory: `base_quantization_type: "fp32"`.
- General production choice: `base_quantization_type: "sq8_uniform"`.
- Data distribution does not fit uniform scaling: try `sq8`.
- Tight memory: try `sq4_uniform`, `pq`, or `rabitq`, and enable reorder.
- If `use_reorder` is enabled, start with `precise_quantization_type: "fp32"`.

Note that `dtype` is the input data type, while `base_quantization_type` is the
internal storage / computation format. They are not the same. For example, input
can be `dtype: "float32"` while the index stores vectors with
`base_quantization_type: "sq8_uniform"`.

Related pages: [Quantization Overview](../quantization/README.md),
[HGraph](../indexes/hgraph.md), [IVF](../indexes/ivf.md).

## Should I use Bitset, lambda, `Filter`, attribute filtering, or `extra_info`?

VSAG provides multiple filtering APIs for different workloads.

Bitset filtering fits a known set of ids to exclude, such as tombstones,
blacklists, or permission-denied ids. `Bitset::Test(id) == true` means the id is
filtered out.

Lambda or `std::function<bool(int64_t)>` fits simple filtering logic. Returning
`true` means the id is filtered out.

A `Filter` object fits more complex filtering logic, or cases where you can
provide hints such as `ValidRatio()`. `Filter::CheckValid(id) == true` means the
id is kept.

Attribute filtering fits structured predicates, such as
`category = "book" AND price <= 100`. It is used through `SearchRequest` and fits
vector + structured field hybrid search.

`extra_info` filtering fits fixed-size byte payloads stored beside each vector.
HGraph can filter on those bytes during graph traversal.
`Filter::CheckValid(const char*) == true` means the vector is kept.

How to choose:

- Use Bitset if you only need to exclude a known id set.
- Use lambda for simple ad hoc logic.
- Use a `Filter` object for complex logic and when you can estimate pass ratio.
- Use attribute filtering for named, typed structured fields.
- Use `extra_info` when metadata is a fixed-size byte payload stored with vectors.

The most confusing part is true / false semantics:

- `Bitset::Test(id)` returns `true` to filter out this id.
- A lambda returns `true` to filter out this id.
- `Filter::CheckValid(id)` returns `true` to keep this id.
- `Filter::CheckValid(const char*)` returns `true` to keep the vector.

When using bitset filtering, keep ids in `[0, 2^32)` when possible to avoid
low-32-bit collisions. If the predicate is very selective, graph search may need
to expand more candidates to collect enough valid results. For HGraph, consider
`brute_force_threshold` so highly selective filters can automatically fall back
to brute-force scanning.

Related pages: [Filtered Search](../advanced/filtered_search.md),
[Attribute Filter](../advanced/attribute_filter.md), [Extra Info](../advanced/extra_info.md).
