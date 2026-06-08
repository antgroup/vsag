# Product Quantization (PQ)

Product Quantization splits a vector into `pq_dim` equal-sized **subvectors**
and quantizes each one independently against a small learned codebook of
`2^pq_bits` centroids. The stored code is then `pq_dim × pq_bits` bits per
vector — orders of magnitude smaller than `fp32`. Distance computations
use precomputed lookup tables (LUT) per query.

![Product Quantization: sub-vector split and codebook lookup](../figures/quantization/pq-codebook.svg)

> Implementation: `src/quantization/product_quantization/product_quantizer.cpp`,
> parameter file `product_quantizer_parameter.cpp`.

## When to use it

- **High-dim float vectors (≥ 256 dim)** where `sq8` is still too large.
- **Memory-tight, accuracy-acceptable** workloads where ~16× compression
  vs fp32 is required.
- Combined with `use_reorder: true` and a small `fp16`/`fp32` precise
  store, PQ is the standard "compressed graph index" recipe at large
  scale.

For wider SIMD throughput at `pq_bits = 4`, see [PQ FastScan](pqfs.md).

## Memory cost (codes only)

`ceil(pq_dim × pq_bits / 8)` bytes per vector for the codes, plus a small
codebook stored once (`pq_dim × 2^pq_bits × subspace_dim × 4` bytes).
For typical settings (`pq_dim = 32`, `pq_bits = 8`, `dim = 128`):

- code size = `32 × 8 / 8 = 32` bytes per vector (vs `128 × 4 = 512` for
  fp32 → 16× smaller).

## Parameters

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `pq_dim` | int | `1` | Number of subvectors. Must divide `dim`. Larger values give finer quantization at the cost of more codebooks and larger codes (`product_quantizer_parameter.h:38`). |
| `pq_bits` | int | `8` | Bits per subvector (1–8). With `8`, each subvector is one byte. Most reliable with `8`; see [PQ FastScan](pqfs.md) for the 4-bit SIMD variant. |

On HGraph these are exposed as the top-level keys `base_pq_dim` and
`pq_bits` (`src/algorithm/hgraph.cpp:465-472`).

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "base_pq_dim": 32,
        "max_degree": 32,
        "ef_construction": 300,
        "use_reorder": true,
        "precise_quantization_type": "fp16"
    }
}
```

## Training

`NEED_TRAIN` is set. Training runs k-means per subspace to learn the
`2^pq_bits` centroids; this is typically the most expensive training step
of any built-in quantizer. Use a training sample of at least `256 ×
2^pq_bits` vectors per subspace for stable codebooks; `Build(base)`
samples from the input automatically.

## Metric compatibility

`l2`, `ip`, `cosine` — all supported. Query-time distance is computed via
a per-subspace LUT: for `l2` it is squared L2 between the query subvector
and each centroid; for `ip` it is the dot product. Cosine reduces to `ip`
on pre-normalized vectors.

## Tips

- `pq_dim` should divide `dim` evenly. Common ratios are `dim/4` or
  `dim/8`.
- Very small `pq_dim` (e.g. `dim/16`) produces very compact codes but
  loses recall fast; combine with reorder.
- For anisotropic data, a rotation transformer in front improves PQ
  recall noticeably: use [Transform Quantizer](../advanced/quantization_transform.md)
  with a chain like `"rom, pq"`.

## Related pages

- [PQ FastScan](pqfs.md)
- [Transform Quantizer](../advanced/quantization_transform.md)
- [HGraph index](../indexes/hgraph.md)
- [Quantization overview](README.md)
