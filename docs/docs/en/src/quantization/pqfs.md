# PQ FastScan

`pqfs` is a SIMD-accelerated variant of [Product Quantization](pq.md) that
fixes `pq_bits = 4` and uses a memory layout designed for the AVX-2 /
AVX-512 "FastScan" lookup-table kernel. At the cost of being 4-bit only,
it delivers significantly higher distance-computation throughput.

![PQ FastScan: 16-vector 4-bit interleaved block and SIMD LUT lookup](../figures/quantization/pqfs-block.svg)

> Implementation: `src/quantization/product_quantization/pq_fastscan_quantizer.cpp`,
> parameter file `pq_fastscan_quantizer_parameter.cpp`.

## When to use it

- The platform has AVX-2 (and ideally AVX-512); the FastScan kernel is
  the main reason to choose `pqfs` over `pq`.
- Search throughput, not just memory, matters.
- 4-bit subspace codebooks (16 centroids per subvector) are sufficient
  for your recall target — typically yes when combined with reorder.

If your platform does not advertise the required SIMD width, fall back to
plain [`pq`](pq.md).

## Memory cost (codes only)

`ceil(pq_dim / 2) = (pq_dim + 1) / 2` bytes per vector — both even and odd
`pq_dim` are supported (`src/quantization/product_quantization/pq_fastscan_quantizer.cpp:41`).
Codebooks: `pq_dim × 16 × subspace_dim × 4` bytes — significantly smaller
than 8-bit `pq` because the codebook has only 16 centroids per subspace.

## Parameters

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `pq_dim` | int | `1` | Number of subvectors. Must divide `dim`. `pq_bits` is **fixed to 4** internally and not configurable (`pq_fastscan_quantizer_parameter.cpp:28-33`). |

Exposed on HGraph as `base_pq_dim` (`src/algorithm/hgraph.cpp:465-472`).

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pqfs",
        "base_pq_dim": 32,
        "max_degree": 32,
        "ef_construction": 300,
        "use_reorder": true,
        "precise_quantization_type": "fp16"
    }
}
```

## Training

`NEED_TRAIN` is set. Trains 16-centroid codebooks per subspace; cheaper
than the 256-centroid training in `pq`.

## Metric compatibility

`l2`, `ip`, `cosine` — same coverage as `pq`. The LUT layout is metric-
specific but transparently handled by the quantizer.

## Tips

- `pq_dim` should be a multiple of the SIMD-batch width the kernel
  expects (the implementation uses 32 internally on AVX-512). When in
  doubt, choose `pq_dim ∈ {32, 64, 96, 128}`.
- The benefit over `pq` is **throughput at the same recall**, not memory
  (4-bit codes are inherently smaller, but `pq` with `pq_bits = 4` would
  match).
- For maximum recall recovery, pair with `use_reorder: true` and an
  `fp16` or `fp32` precise store.

## Related pages

- [Product Quantization (PQ)](pq.md)
- [HGraph index](../indexes/hgraph.md)
- [Transform Quantizer](../advanced/quantization_transform.md)
- [Quantization overview](README.md)
