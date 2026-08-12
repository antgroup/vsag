# Index Parameters

This page summarises the commonly used parameters for every VSAG index type. For the full
enumeration, consult the source:

- Build parameter keys: `src/constants.cpp`
- Public constants: `include/vsag/constants.h`
- Per-index examples: the `examples/cpp/*_index_*.cpp` files (e.g. `103_index_hgraph.cpp`).

## Common Fields

Every index accepts these top-level fields at build time. `dtype` and `metric_type` are required;
`repr` is optional. `dim` is required for non-sparse data and defaults to `4096` for
`dtype: "sparse"` when omitted:

| Field | Values | Description |
|-------|--------|-------------|
| `dim` | positive integer | Vector dimensionality; cannot change after build |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` / `sparse` | Scalar value type. `sparse` is retained for sparse-index compatibility. |
| `repr` | `dense` / `sparse` / `multi_vector` | Optional data layout. When omitted, VSAG infers `sparse` from `dtype: "sparse"` and otherwise uses `dense`. |
| `metric_type` | `l2` / `ip` / `cosine` | Distance metric |

`dtype` and `repr` describe different properties: `dtype` is the scalar encoding, while `repr`
is the record layout. `dtype: "sparse"` requires `repr: "sparse"` when `repr` is explicit.
Use `repr: "multi_vector"` with a supported multi-vector index and a scalar `dtype` such as
`float32`.

## HGraph

HGraph places its build parameters under the generic `index_param` key (see
`examples/cpp/103_index_hgraph.cpp`); the `hgraph` key is reserved for search-time parameters.

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 400
    }
}
```

| Field | Typical | Description |
|-------|---------|-------------|
| `max_degree` | 16–48 | Maximum out-degree per node |
| `ef_construction` | 200–500 | Candidate set size during build; larger = higher recall, slower build |
| `base_quantization_type` | `fp32` / `fp16` / `bf16` / `sq8` / `sq4` / `pq` | Quantization of the base storage — see the [Quantization chapter](../quantization/README.md) for all supported values |
| `use_reverse_edges` | `false` | Track incoming neighbors for O(1) reverse-edge lookup; roughly doubles edge storage and is unsupported with compressed graph storage |
| `label_remap_type` | `pg` | Label-map implementation: `pg` (default) or `robin` |
| `reorder_source` | `precise` | Reorder from the `precise` store or directly from `base`; RaBitQ x+y split, including `tq_chain="mrle, rabitq"`, selects `base` automatically |
| `persist_source_id` | `false` | Include HGraph source-ID metadata in serialization; useful when a restored index must later export a build cache |
| `use_conjugate_graph` | `false` | Enable HGraph feedback/pretraining and persist the auxiliary conjugate graph |
| `mrle_dim` | `0` | MRLE output dimension in `[0, dim]`; `0` means input dimension |
| `fast_encode_rabitq` | `true` | Use fast multi-bit RaBitQ encoding; `false` restores the exact encoder |
| `fast_encode_rabitq_rounds` | `6` | Fast-encoder refinement rounds in `[1, 32]` |

At search time:

```json
{"hgraph": {"ef_search": 100}}
```

`ef_search` accepts any positive signed 64-bit integer. It is no longer capped relative to
`topk`; very large values can substantially increase latency and memory used by the frontier.
`use_conjugate_graph_search` is a boolean (default `true`) that uses learned conjugate edges when
the index was built with `use_conjugate_graph: true`.

The `hgraph` search-param object also accepts `brute_force_threshold` (a float
in `[0.0, 1.0]`, default `0.0`). When set above zero and the request carries a
filter whose `ValidRatio()` is at most this threshold, HGraph skips the graph
traversal and runs an exact scan over the surviving ids. See the
[HGraph index page](../indexes/hgraph.md#brute-force-fallback-under-highly-selective-filters-brute_force_threshold)
for details.

## LazyHGraph

LazyHGraph can take its build parameters in a top-level `lazy_hgraph` object
(preferred for clarity) or in the generic `index_param` object. The `hgraph`
sub-object is forwarded to the internal HGraph used after transition.

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "lazy_hgraph": {
        "transition_threshold": 1000,
        "hgraph": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100
        }
    }
}
```

| Field | Typical | Description |
|-------|---------|-------------|
| `transition_threshold` | `1000` or workload-specific | Positive vector count at which the index converts from exact flat search to HGraph |
| `hgraph` | HGraph build object | Parameters for the graph phase; see [HGraph](../indexes/hgraph.md) |

LazyHGraph only supports `dtype: "float32"`. Search parameters use the `hgraph`
object, for example `{"hgraph": {"ef_search": 100}}`. See the
[LazyHGraph index page](../indexes/lazy_hgraph.md) for details.

The `hgraph` search-param object also accepts the following filter-related parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_ratio` | float | `0.2` | Controls the ratio of filtered-search candidate checks to skip, in range `[0.0, 1.0]`. Higher values mean more aggressive skipping, faster search, and potentially lower recall. |
| `skip_strategy` | string | `"deterministic_accumulative"` | Skip strategy. Supports `"random"` and `"deterministic_accumulative"`. |

## IVF

```json
{
    "ivf": {
        "nlist": 4096,
        "base_quantization_type": "sq8",
        "nprobe": 32
    }
}
```

## Brute Force

```json
{"brute_force": {}}
```

No extra parameters.

## Pyramid

Pyramid build parameters also live under `index_param`:

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 24,
        "ef_construction": 300
    }
}
```

MRLE with split RaBitQ uses `base_quantization_type: "tq"`,
`tq_chain: "mrle, rabitq"`, `mrle_dim`, and the
`rabitq_bits_per_dim_base`/`rabitq_bits_per_dim_precise` pair. It automatically reorders from the
split base codes and retains original FP32 vectors for decode-only operations. See the
[Pyramid page](../indexes/pyramid.md#mrle-with-split-rabitq) for a complete configuration and
storage/recall tradeoffs.

## SINDI (sparse vectors)

```json
{
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "term_id_limit": 30000,
        "doc_prune_ratio": 0.1
    }
}
```

See the [SINDI page](../indexes/sindi.md) for `use_quantization`, immutable builds, and search
parameters such as `n_candidate`.

## SINDI_V2 (sparse vectors)

SINDI_V2 supports all SINDI features with both in-memory and disk-based I/O.

```json
{
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "term_id_limit": 30000,
        "use_reorder": true,
        "term_io": {
            "type": "async_io",
            "file_path": "/path/to/sindi_v2.terms"
        },
        "rerank_io": {
            "type": "async_io",
            "file_path": "/path/to/sindi_v2.rerank"
        }
    }
}
```

See the [SINDI_V2 page](../indexes/sindi_v2.md) for details.

## Runtime Parameters

Beyond build-time parameters, `Index::Tune` and `SearchParam` tweak runtime settings such as
`ef_search` and `nprobe`. See [Optimizer](../advanced/optimizer.md) and the
`examples/cpp/3xx_feature_*.cpp` examples.
