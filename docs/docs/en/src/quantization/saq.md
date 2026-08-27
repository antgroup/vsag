# SAQ (Segmented Code Adjustment Quantization)

`saq` is a trained low-bit quantizer for high-dimensional dense vectors. It combines a
full-dimensional PCA rotation, variance-aware dimension segmentation, dynamic bit allocation, and
Code Adjustment Quantization (CAQ). The implementation follows the algorithm in
[SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension
Segmentation](https://arxiv.org/abs/2509.12086).

> Implementation: `src/quantization/saq_quantization/saq_quantizer.cpp`; parameters:
> `saq_quantizer_parameter.cpp`.

## Pipeline

Training performs these steps:

1. Learn a full-dimensional PCA rotation and order components by decreasing variance. L2 data is
   centered with the learned mean, matching the reference SAQ preprocessing. Inner product and
   cosine are not centered because translation changes those metrics. The projection keeps every
   dimension.
2. Split the ordered dimensions on 64-dimension boundaries. Dynamic programming jointly chooses
   boundaries and a 1–13 bit width for each segment under the configured record budget. Setting an
   explicit segment count uses evenly spaced boundaries and optimizes only the bit allocation.
3. Create an independent pseudorandom orthogonal rotation for each segment, unless disabled. SAQ
   derives it from the fixed seed `20260825` and the segment offset so repeated training on the
   same input produces the same persisted model.
4. Encode each segment with symmetric scalar quantization, then run CAQ coordinate adjustment to
   improve the direction of the quantized vector.

Search projects the query once and builds byte lookup tables for the projected coordinates. Scalar
codes are stored as bit planes and scanned by dedicated batch-four lookup kernels (AVX512 where
available, with AVX2 and generic fallbacks), without reconstructing an FP32 candidate. L2 threshold
scans visit segments in variance order and can stop as soon as the accumulated non-negative segment
distance exceeds the current threshold.

## Parameters

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `saq_avg_bits` | float | `4.0` | Average payload target per input dimension, in `[1, 8]`. The complete record reserves the same 12-byte overhead as the equal-length multi-bit RaBitQ baseline; SAQ charges 8 bytes per selected segment within that record. |
| `saq_segment_count` | integer | `0` | `0` lets dynamic programming choose the segment count and boundaries. A positive value requests that many 64-dimension-aligned segments. Training fails if the bit budget cannot represent them. |
| `saq_adjustment_rounds` | integer | `6` | Maximum CAQ coordinate-adjustment rounds, in `[0, 32]`. `0` keeps the scalar initialization without adjustment. |
| `saq_use_pca` | bool | `true` | Learn and apply the full-dimensional PCA rotation. Disable only for controlled ablations. |
| `saq_random_rotation` | bool | `true` | Balance coordinates with an orthogonal rotation inside every segment. |

The complete record length matches multi-bit RaBitQ at the same configured average bit count. Each
segment uses 8 bytes of metadata: one adjusted scale and one metric-specific squared norm. L2 stores
the original projected norm required by its asymmetric query-to-code formula; cosine stores the
reconstructed norm so query-to-code and code-to-code paths compute the same decoded-vector cosine.
Multiple segments therefore leave fewer payload bits under the fixed record budget. Bit-plane code
records are fixed-width, so memory and disk layouts remain stable after training.

## HGraph example

```json
{
    "dim": 768,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "saq",
        "saq_avg_bits": 4,
        "saq_segment_count": 0,
        "saq_adjustment_rounds": 6,
        "saq_use_pca": true,
        "saq_random_rotation": true,
        "max_degree": 32,
        "ef_construction": 400,
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

The same five `saq_*` keys are accepted by IVF and Pyramid when
`base_quantization_type` is `saq`. IVF trains SAQ on its base-code input (residuals when
`use_residual` is enabled). HGraph and Pyramid use the quantizer directly through the shared flatten
datacell.

## Training, persistence, and limitations

- SAQ needs at least two training vectors. Use `Build`, or call `Train` before `Add`.
- Serialization includes the PCA matrix, projected L2 mean, segment plan, bit allocation, and
  segment rotations. Loading does not retrain the model; it rejects a rotation whose element count
  does not match the segment dimensions or which contains non-finite values.
- The full PCA model takes O(dim²) space. The quantizer deterministically samples at most 65,536
  evenly spaced inputs for PCA and segment-plan training; all inputs are still encoded into the
  index.
- Disabling PCA for an L2 ablation keeps mean centering enabled, so the comparison isolates PCA
  ordering rather than also changing the origin.
- Segment rotations use a fixed implementation seed for reproducible builds; the learned matrices
  are serialized, so loading never regenerates them.
- All dense metrics (`l2`, `ip`, and `cosine`) are supported. Cosine inputs are normalized before
  PCA and encoding. If either operand has zero norm, cosine distance is defined as `1.0`, matching
  VSAG's finite-distance convention.
- Progressive threshold termination is currently enabled only for L2, where partial segment
  distances form a safe monotonic lower bound on the remaining work.

## Choosing settings

Start with `saq_avg_bits: 4`, automatic segmentation, six adjustment rounds, PCA, and random segment
rotations. Compare against RaBitQ at the same code budget and enable an FP32 reorder store when the
application requires high recall. Use `saq_segment_count` only for reproducible ablations; automatic
segmentation adapts better to the variance profile of each dataset.

## Related pages

- [Quantization overview](README.md)
- [RaBitQ](rabitq.md)
- [HGraph](../indexes/hgraph.md)
- [IVF](../indexes/ivf.md)
- [Pyramid](../indexes/pyramid.md)
