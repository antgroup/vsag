# Scalar Quantization Uniform (SQ4 / SQ8 Uniform)

`sq8_uniform` and `sq4_uniform` are scalar quantizers like
[`sq8` / `sq4`](sq.md), except they learn a **single global** `[min, max]`
range that applies to every dimension. This trade-off — slightly less
adaptive per dimension, but a much simpler decode path — unlocks SIMD code
that runs significantly faster on `l2` and `ip` distance kernels and
keeps the code layout tighter.

![Uniform (global range) vs per-dimension Scalar Quantization](../figures/quantization/sq-uniform-vs-perdim.svg)

> Implementation: `src/quantization/scalar_quantization/sq8_uniform_quantizer.cpp`,
> `src/quantization/scalar_quantization/sq4_uniform_quantizer.cpp`.

## Why it is fast: distances stay in the integer domain

This is the core reason to prefer `sq*_uniform` over `sq*` whenever it
applies. Because every dimension shares one `(min, max)` pair, the affine
decode `x = min + code · (max - min) / (2^b - 1)` has the **same scale and
offset for every coordinate**. That has three consequences in the hot path:

- The query is encoded **once** with the same global `(min, max)` into a
  uint8 (or packed nibble) buffer, in `ProcessQueryImpl`
  (`src/quantization/scalar_quantization/sq8_uniform_quantizer.cpp:179`).
- Each base vector code is **never decoded back to fp32**. The kernel
  `SQ8UniformComputeCodesIP(uint8_t* q, uint8_t* x, dim)` /
  `SQ4UniformComputeCodesIP(...)` reads both operands as raw integer codes
  and does the dot product on uint8 / packed nibble lanes using AVX-512 /
  AMX (or NEON on ARM), one cache-line at a time. There is no per-element
  fp dequantization in the inner loop.
- The single shared scale factor and offset are applied **once per
  pair**, after the integer reduction, to recover the fp distance. Some
  metric-specific corrections (a per-vector norm or sum) are also added
  outside the loop; see the trailing metadata noted in
  `sq8_uniform_quantizer.cpp:200` and the
  `SQ8UniformComputeCodesIPBatch` batch kernel.

In the per-dimension `sq*` quantizers, each coordinate has its own
`(min_i, max_i)` so the kernel either has to multiply by a per-dim scale
table inside the loop or decode at least one operand back to fp first.
Skipping that work is what makes uniform variants significantly faster at
the same recall.

## When to use it

- **HGraph / IVF / Pyramid hot paths.** When the bottleneck is the
  base-quantizer distance computation, `sq8_uniform` / `sq4_uniform` are
  almost always faster than their non-uniform counterparts at comparable
  recall.
- **Data with similar coordinate ranges across dimensions.** Normalized
  embeddings (cosine), or vectors that have already been rotated (e.g.
  through a [Transform Quantizer](../advanced/quantization_transform.md)
  chain like `"rom, sq8_uniform"` or `"fht, sq8_uniform"`) are the ideal
  inputs.
- **As the terminal quantizer of a `tq` chain.** The most common chain is
  `"pca, rom, sq8_uniform"`, see example 501.

## SQ4 uniform vs SQ8 uniform

| Type | Bits / dim | Memory vs fp32 | Typical accuracy |
| --- | --- | --- | --- |
| `sq8_uniform` | 8 | ~1/4 | minor recall loss |
| `sq4_uniform` | 4 | ~1/8 | needs reorder for high recall |

## Parameters

| Key | Type | Default | Applies to | Meaning |
| --- | --- | --- | --- | --- |
| `sq4_uniform_trunc_rate` | float | `0.05` | `sq4_uniform` only | Symmetric truncation rate for outliers (`src/quantization/scalar_quantization/sq4_uniform_quantizer_parameter.h:39`). Higher values clip more extreme coordinates, reducing range loss for the bulk of the data at the cost of clipping the tails. |

`sq8_uniform` has no quantizer-specific JSON parameters.

When using HGraph, `sq4_uniform_trunc_rate` is exposed as a top-level key
and mapped into the nested quantization params
(`src/algorithm/hgraph.cpp:409-416`).

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq4_uniform",
        "sq4_uniform_trunc_rate": 0.05,
        "max_degree": 32,
        "ef_construction": 300,
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

Set `"base_quantization_type": "sq8_uniform"` and drop the `trunc_rate`
key for the 8-bit variant.

## Training

`NEED_TRAIN` is set. Training estimates one global `[min, max]` across all
dimensions (with optional truncation for `sq4_uniform`). `Build` will
perform training internally.

## Metric compatibility

`l2`, `ip`, `cosine` — all supported. `cosine` normalizes before quantizing,
which is also what makes uniform scaling close to optimal for that metric.

## Choosing between uniform and non-uniform

- Data is normalized (`cosine` or pre-normalized `l2`) → **uniform**.
- Data has very heterogeneous per-dimension ranges (e.g. mixed feature
  blocks) → start with non-uniform [`sq*`](sq.md), or use uniform behind a
  rotation transformer (`"rom, sq*_uniform"`).
- Throughput matters more than the last bit of recall → **uniform**.

## Related pages

- [Scalar Quantization (SQ4 / SQ8)](sq.md)
- [Transform Quantizer](../advanced/quantization_transform.md)
- [Quantization overview](README.md)
