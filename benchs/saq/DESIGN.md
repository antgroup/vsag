# SAQ design and VSAG integration

## Scope

This change adds Segmented Code Adjustment Quantization as the opt-in `saq` base quantizer for
HGraph, IVF, and Pyramid. It does not change the default quantizer or public index behavior. The
implementation uses VSAG's allocator, quantizer CRTP interface, flatten datacell, stream format,
PCA/orthogonal transforms, and runtime SIMD distance functions.

The algorithm is based on [SAQ: Pushing the Limits of Vector Quantization through Code Adjustment
and Dimension Segmentation](https://arxiv.org/abs/2509.12086) and was checked against the authors'
[reference implementation](https://github.com/howarlii/SAQ). The VSAG record layout and distance
interface are native implementations rather than a source-level import of the reference IVF code.

## Training path

1. Deterministically sample at most 65,536 evenly spaced training vectors. For cosine, normalize
   the sample; for L2 and inner product, preserve its input values. This bounds the temporary
   full-dimensional PCA matrix while every base vector is still encoded into the index.
2. Train a `dim × dim` PCA matrix. L2 encoding and search subtract the learned mean before the
   orthogonal projection, matching the reference SAQ preprocessing and preserving L2 geometry.
   Inner product and cosine use the full orthogonal projection without centering, because a shared
   translation would change those metrics. The controlled L2 no-PCA ablation still subtracts and
   restores the training mean, so it isolates PCA ordering rather than conflating PCA with
   centering. Every PCA path retains all components and orders dimensions by decreasing training
   variance.
3. Compute the variance of each projected dimension in one streaming pass.
4. Jointly choose contiguous segment boundaries and a bit width from 1 through 13 for each segment.
5. Instantiate a pseudorandom orthogonal transform per segment when `saq_random_rotation` is
   enabled. The fixed seed `20260825` is combined with the segment offset, making repeated builds
   deterministic without reusing the same matrix for equal-sized segments.

The PCA covariance calculation uses the configured BLAS `SGEMM` backend. This removes the scalar
`O(ND²)` loop from the high-dimensional training hot path while retaining the same covariance
definition and eigendecomposition.

## Segmentation and bit allocation

Candidate boundaries are multiples of 64 dimensions plus the final input dimension. Let `V(i,j)`
be the sum of projected variances for the dimensions in `[i,j)`, and let `b` be a segment bit width.
The dynamic program minimizes the paper's proxy:

```text
DP[j, used + metadata_bits + bitplane_bytes((j-i), b)] =
    min(DP[j, ...], DP[i, used] + V(i,j) / 2^b)
```

The record budget is `round_to_byte(round(saq_avg_bits * dim) + 96)` bits, matching the complete
record length of the comparable multi-bit RaBitQ layout. SAQ stores 64 bits of metadata per segment,
as in the reference implementation, so each additional segment consumes another 64 bits from the
fixed record. Codes are stored as byte-aligned bit planes; the planner charges their exact padded
size. When candidate plans are within one percent of the best modeled error, the implementation
prefers fewer segments to avoid metadata and search overhead.

With `saq_segment_count > 0`, boundaries are evenly distributed over the 64-dimension blocks and a
smaller DP optimizes only the per-segment bit widths. Invalid segment counts or insufficient record
budgets fail training with an argument error.

## Encoding and CAQ adjustment

Each segment applies symmetric uniform scalar initialization. A record stores:

- adjusted scale that combines the scalar range and CAQ rescale factor (`float`);
- a metric-specific squared norm (`float`): the original projected norm for L2 and the
  reconstructed norm for cosine; the field is retained but not read by inner product;
- byte-aligned bit planes at the trained bit width.

CAQ then performs at most `saq_adjustment_rounds` coordinate sweeps. For a coordinate, the code is
moved up or down while the move improves the squared cosine objective
`<x,q>² / ||q||²`. A sweep stops early when no coordinate changes. Finally the rescale factor
`||x||² / <x,q>` is stored. Setting the rounds to zero is a supported scalar-initialization
ablation.

## Search and SIMD behavior

The query is normalized for cosine, PCA-projected once, and transformed once per segment into the
quantizer computer's reusable query buffer. Its per-segment squared norms and coordinate sums are
cached. A 256-entry dot-product lookup table is built for every eight query coordinates. Stored
1–13 bit planes are then scanned with VSAG's existing RaBitQ lookup kernels, including dedicated
four-record AVX512 and AVX2 paths and a generic fallback. This layout avoids reconstructing an FP32
vector for every candidate and is the path used by the flatten datacell's batch-four graph scan.

HGraph neighbor pruning also requests distances between two stored codes. PCA and every segment
rotation are orthogonal, so L2, inner product, and cosine can be evaluated directly in their shared
projected coordinate system. The code-pair path computes code sums, squared sums, and cross-products
from bit-plane intersections with popcount, then evaluates the decoded-vector distance from those
moments. With the bit width bounded by 13 this remains linear in `D`, avoids per-coordinate code
unpacking, and exactly matches a decoded-vector calculation within floating-point tolerance. It
also avoids two inverse dense transforms, which would turn every pruning comparison into `O(D²)`
work and dominate high-dimensional construction.

For L2, each segment contributes `||q_s||² + ||x_s||² - 2<q_s,x_s>`. Contributions are clamped to
zero to protect the monotonic partial sum from floating-point roundoff. Threshold scans can stop
after any segment whose partial sum already exceeds the current graph-search threshold. IP and
cosine scan every segment because their partial sums are not lower bounds. Cosine divides the
accumulated inner product by the query and reconstructed-code norms on both query-to-code and
code-to-code paths. If either norm is zero, both paths return the finite distance `1.0`.

## Persistence and compatibility

The model stream contains the five user parameters, fixed bit budget, PCA matrix, projected L2 mean,
ordered segment plan, and each optional random rotation. Deserialization validates the base metric
and trained state, record budget, fixed-plan consistency, full ordered and aligned dimension
coverage, bit widths, PCA dimensions, metric-appropriate mean dimensions, and record layout before
accepting the model. Every serialized segment rotation must also contain exactly `length²` finite
matrix elements; validation occurs before the transform instance accepts the replacement matrix.
Existing quantizer stream formats and external defaults are unchanged; `saq` is selected only when
explicitly configured.

## Complexity and known boundaries

- PCA model space: `O(D²)`; eigendecomposition: `O(D³)`; covariance: BLAS-backed `O(ND²)`.
- Planning: `O(P²QB)` for `P` aligned boundary positions, bit budget `Q`, and 13 bit choices.
- Encoding: PCA `O(D²)`, segment rotations `O(sum D_s²)`, and CAQ `O(rounds × D)` after transforms.
- Search: query transforms and lookup construction once, then `O(D)` bit-plane lookup per visited
  record.
- Graph construction code-pair distance: `O(B²D/64)` bit-plane moments for segment width `B`, with
  `B <= 13`; this is effectively linear in `D` under the public format bound.

This version uses a native per-record bit-plane lookup layout rather than the reference repository's
32-record transposed FastScan blocks. It does not implement zero-bit tail segments, learned
non-orthogonal dimensionality reduction, or GPU kernels. Those optimizations can be added without
changing the external parameter contract.

## Verification map

Unit tests cover dynamic/fixed plans, exact record budgets, 1–13 bit-plane packing, invalid
parameters, all three metrics, deterministic rotations, CAQ-on versus CAQ-off encoding,
encode/decode, code-pair moments including non-aligned dimensions and widths above 8, scalar versus
batch-four query distance, cosine zero-norm handling and decoded-vector semantics, L2 threshold
termination, PCA ordering, mean restoration, serialization round trips, and rejection of malformed
rotation sizes/non-finite values. A real HGraph SAQ build/search/serialize/restore/search round trip
checks index persistence; separate HGraph, IVF, and Pyramid tests verify that all five external
parameters reach the internal quantizer configuration.
