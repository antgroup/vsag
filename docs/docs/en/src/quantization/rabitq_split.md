# RaBitQ x+y Split

RaBitQ x+y split is an HGraph, IVF, and Pyramid storage and search mode for
low-bit base codes.
Each vector is divided into two records:

- `x` filter bits are read during graph traversal or IVF bucket scanning.
- `y` supplement bits are fetched only for candidates that reach reorder.
- The final reorder distance uses all `x+y` bits.

This layout keeps the traversal record small while retaining a higher-precision
RaBitQ distance for final ranking. It also allows the filter record to stay in
memory while the colder supplement record is stored on disk.

## Enable split mode

HGraph, IVF, and Pyramid select split mode when both quantization types are
`rabitq` and
`rabitq_bits_per_dim_precise` is present:

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 960,
    "index_param": {
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "use_reorder": true,
        "rabitq_bits_per_dim_query": 32,
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5,
        "rabitq_error_rate": 1.9,
        "max_degree": 64,
        "ef_construction": 400
    }
}
```

The relevant parameters are:

| Parameter | Meaning |
| --- | --- |
| `base_quantization_type` | Must be `"rabitq"`. |
| `precise_quantization_type` | Must also be `"rabitq"` to select split mode. |
| `rabitq_bits_per_dim_base` | `x`, the number of filter bits read during traversal. |
| `rabitq_bits_per_dim_precise` | `y`, the number of supplement bits fetched during reorder. |
| `rabitq_bits_per_dim_query` | Must be `32` for split storage. |
| `rabitq_error_rate` | Default positive multiplier applied to the lower-bound error term. |
| `use_reorder` | Should be `true` so candidates are ranked with the `x+y` distance. |

The constraints are:

```text
1 <= x <= 8
1 <= y <= 8
x + y <= 8
```

If `rabitq_bits_per_dim_precise` is omitted, HGraph, IVF, and Pyramid use the standard RaBitQ
path instead of split storage.

Enable the filter/lower-bound search path with:

```json
{
    "hgraph": {
        "ef_search": 200,
        "parallelism": 4,
        "rabitq_one_bit_search": true,
        "rabitq_error_rate": 1.9
    }
}
```

The external search key is named `rabitq_one_bit_search`, but on a split index
it uses all `x` filter bits configured by `rabitq_bits_per_dim_base`.
`hgraph.rabitq_error_rate` overrides the index default for that search. It can
be swept without rebuilding because the stored record contains the geometric
error scale before this multiplier is applied.

IVF bucket scans use the configured `x` filter bits automatically; no
`rabitq_one_bit_search` switch is needed. The default
`rabitq_search_strategy: "candidate_reorder"` uses `ivf.factor` to control
how many filter-stage candidates (`factor * topk`) proceed to supplement
reranking.

IVF also provides an opt-in lower-bound heap strategy:

```json
{
    "ivf": {
        "scan_buckets_count": 32,
        "rabitq_search_strategy": "heap",
        "parallelism": 4
    }
}
```

This KNN-only strategy keeps full `x+y` RaBitQ estimates in the result heap. It
reads the `y`-bit supplement only when an x-bit lower bound is strictly smaller
than the current heap maximum. `factor` is not used by this strategy. It
requires split storage, `use_reorder: true`, and search-time
`enable_reorder: true`.
The heap entries reuse the byte-LUT-quantized x-bit inner product and add the
y-bit supplement contribution. The x-bit lower bound and the final `x+y`
estimate therefore share the same LUT quantization error. The RaBitQ error term
is still controlled by `rabitq_error_rate`; more aggressive pruning can trade
recall for fewer supplement reads.

## Search pipeline

The split search path has four stages:

1. The query is transformed and normalized once. Non-residual scans and
   residual IVF `candidate_reorder` build one byte lookup table per query.
   Residual IVF `heap` builds bucket-specific `q-c` lookup tables to preserve
   its lower-bound formula.
2. Graph traversal or IVF bucket scanning reads only the filter record. It
   computes an x-bit distance estimate; lower-bound paths also compute a
   conservative lower bound.
3. HGraph can discard candidates whose lower bound cannot enter the result set.
   IVF either retains `factor * topk` candidates by filter distance (the default)
   or compares each lower bound with an `x+y`-estimate heap (the opt-in `heap`
   strategy).
4. The final distance combines the filter contribution and supplement
   contribution into one `x+y`-bit RaBitQ estimate. Both IVF strategies carry
   the byte-LUT-quantized x-bit inner product `S_x` from the filter scan, so
   normal final-distance calculation reads only the y-bit record and does not
   reconstruct the inner product from the filter distance.

Both IVF strategies use the bucket-local 32-vector FastScan layout. A
candidate scan emits the x-bit estimate and the reusable inner product; heap
mode additionally emits the lower bound. The final distance uses
`2^y * S_x + S_y`, where only `S_y` requires reading the y-bit supplement.
There is no companion float-LUT pass. Packed x bits are reread only by the
correctness fallback for an invalid or stale candidate.

The traversal or bucket-scan heap is therefore not populated with an `x+y`
distance for every
visited vector. The inexpensive x-bit distance drives traversal; the more
accurate distance is evaluated only during candidate reorder.

## Encoding and bit planes

Let:

```text
d       = transformed dimension
x       = filter bits per dimension
y       = supplement bits per dimension
B       = x + y
P       = ceil(d / 8), bytes in one bit plane
q_i     = transformed and normalized query coordinate
u_i     = unsigned B-bit base code, 0 <= u_i < 2^B
```

The centered full code is:

```text
c_B = (2^B - 1) / 2
z_i = u_i - c_B
N_B = sqrt(sum_i z_i^2)
```

`PackIntoPlanes` stores each logical bit of `u_i` in a separate bit plane.
The split is defined by:

```text
f_i = floor(u_i / 2^y)    # top x bits
s_i = u_i mod 2^y         # low y bits
u_i = 2^y * f_i + s_i
```

The physical order keeps the most significant filter planes contiguous:

```text
filter record:     logical B-1, B-2, ..., B-x
supplement record: logical 0, 1, ..., y-1
```

This order lets traversal scan exactly `x * P` plane bytes and lets reorder
fetch exactly `y * P` additional plane bytes, excluding metadata and alignment.

## Datacell layout

`RaBitQSplitDataCell` owns two `RaBitQSplitCodeStorage` instances.

### Filter record

The filter record in `x_bit_cell_` contains:

```text
x high bit planes
base norm
filter-code norm when x > 1
optional MRQ residual norm
optional raw norm for IP/cosine
lower-bound error
filter approximation error
```

For one vector, its plane payload is:

```text
FilterPlanesSize = x * ceil(d / 8)
```

The filter record is the hot traversal record. Graph search and prefetch do
not need the supplement record while the x-bit estimate is valid.

### Supplement record

The supplement record in `supplement_cell_` contains:

```text
y low bit planes
full-code norm
full-code approximation error
remaining metadata required by the selected metric and transforms
```

Its plane payload is:

```text
SupplementPlanesSize = y * ceil(d / 8)
```

The complete code payload is approximately `(x+y) * d / 8` bytes per vector,
plus aligned norms, errors, and optional transform metadata.

## X-bit filter estimate and lower bound

The filter code for coordinate `i` is `f_i` in `[0, 2^x - 1]`. Define:

```text
c_x   = (2^x - 1) / 2
N_x   = sqrt(sum_i (f_i - c_x)^2)
S_x   = sum_i q_i * f_i
Q_sum = sum_i q_i
rho_x = (S_x - c_x * Q_sum) / N_x
```

During index construction, RaBitQ stores the absolute filter approximation
error `E_x` and the geometric error scale:

```text
E_safe    = clamp(abs(E_x), 1e-5, 1)
epsilon_x = sqrt(max(0, 1 - E_safe^2) / max(1, d - 1))
```

The corrected filter inner-product estimate is:

```text
rho_hat_x = rho_x / abs(E_x)
```

For L2, with base norm `N_o` and query norm `N_q`, the x-bit distance and
lower bound are:

```text
D_x = N_o^2 + N_q^2 - 2 * N_o * N_q * rho_hat_x

LB = D_x
     - 2 * N_o * N_q * rabitq_error_rate * epsilon_x / abs(E_x)
```

The implementation subtracts a small floating-point guard from `LB`. IP and
cosine apply the corresponding metric conversion to the error term.

The lower bound is used only to reject candidates safely. `D_x` remains the
traversal estimate, while the final ranking uses the `x+y` distance.

## IVF 1-bit, 2-bit, and 3-bit 32-vector FastScan layout

When IVF split storage uses `x` from 1 through 3, `RaBitQSplitBucketDataCell`
packages each bucket in groups of 32 candidates. These bucket-local packed
blocks are the only persistent filter-code layout and contain the SIMD-oriented
filter planes plus compact metadata. Single-lane x bits are unpacked from the
block only for correctness fallback or full-code access by ID. Supplement
reranking normally reads only the separate y-bit record.

Let `G = ceil(d / 8) * 2` be the number of four-dimensional groups in one
bitplane. Each group stores one four-bit mask per candidate. For every group,
the 32 masks are transposed into 16 bytes. The low nibbles represent candidates
`0..15`, the high nibbles represent candidates `16..31`, and both halves use
the lane order `0, 8, 1, 9, ..., 7, 15`. Multiple bitplanes remain separate
and contiguous in plane-major order.

For `x = 1`, each group has the 16-entry subset-sum table:

```text
LUT_g[m] = sum(q_(4g+j) for j in [0, 3] when bit_j(m) is set)
```

For `x = 2` or `x = 3`, let bitplane zero be the most significant plane. The
centered lookup table is:

```text
weight(x, p) = 2^(x - p - 2)
LUT_(p,g)[m] = weight(x, p) * sum((2 * bit_j(m) - 1) * q_(4g+j))
```

The plane weights are `{1, 0.5}` for two bits and `{2, 1, 0.5}` for three
bits. Coordinates beyond `d` are zero.

One-bit and two-bit scans use one query-wide unsigned-byte quantizer. Three-bit
scans quantize each plane independently, preventing the low-weight plane from
losing effective precision to the four-times-wider range of the high plane:

```text
delta[p] = (v_max[p] - v_min[p]) / 255
plane_sum[p] ~= G * v_min[p] + delta[p] * sum LUT_u8[p][code]
centered_ip ~= plane_sum[0] + plane_sum[1] + plane_sum[2]
```

The three-bit scanner invokes `PQFastScanLookUp32` once for each contiguous
plane. This keeps the same `3 * G` total shuffle groups and packed-code size as
a single combined scan, while allowing separate dequantization scales.

For one bit, the lookup sum is converted to the normalized binary inner
product with `sum(q)` and `sqrt(d)`. For two and three bits, it is already the
centered inner product and is divided by the stored filter-code norm. All paths
then use the stored RaBitQ norm and error metadata to recover the filter-stage
distance.

The packed filter planes are followed by the metadata for all 32 candidates.
A final bucket block with fewer than 32 vectors is zero-padded, while the
scanner returns only valid candidates. Runtime dispatch selects generic, SSE,
AVX2, or AVX-512 code. In heap search, the dequantized byte-LUT result is saved
as `S_x` and reused in the final `x+y` estimate, so filtering and reranking share
the same LUT quantization error. Wider filter widths continue to use the
bit-plane batch paths.

Example configurations are:

```json
{
    "rabitq_bits_per_dim_base": 1,
    "rabitq_bits_per_dim_precise": 7,
    "rabitq_bits_per_dim_query": 32
}
```

```json
{
    "rabitq_bits_per_dim_base": 2,
    "rabitq_bits_per_dim_precise": 6,
    "rabitq_bits_per_dim_query": 32
}
```

```json
{
    "rabitq_bits_per_dim_base": 3,
    "rabitq_bits_per_dim_precise": 5,
    "rabitq_bits_per_dim_query": 32
}
```

### GIST1M comparison

The following results compare traditional 8-bit IVF RaBitQ with the split
layouts on `gist-960-euclidean.hdf5` (1,000,000 base vectors, 960 dimensions,
L2). The index used 1,024 buckets, 100,000 training samples, 16 build threads,
and `rabitq_bits_per_dim_query = 32`. Search used one thread, 32 scanned
buckets, `factor = 10`, top-10, and 5,000 measured queries. The split search
results include the 32-vector FastScan layout described above.

| Layout | Build time (s) | Build TPS | Index memory (bytes) | Search QPS | Avg latency (ms) | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Traditional RaBitQ 8-bit | 150.742 | 6,633.840 | 1,284,595,208 | 17.767 | 56.280 | 0.9197 |
| Split RaBitQ 1+7 | 551.701 | 1,812.576 | 1,450,355,304 | 118.333 | 8.447 | 0.9019 |
| Split RaBitQ 2+6 | 509.431 | 1,962.974 | 1,582,237,512 | 74.786 | 13.368 | 0.9128 |
| Split RaBitQ 3+5 | 496.873 | 2,012.589 | 1,702,580,248 | 67.892 | 14.726 | 0.9094 |

These numbers are a single-machine comparison rather than a universal
performance guarantee. Split storage spends more time building two code
streams and keeps a bucket-local FastScan copy in memory. In return, its
filter scan avoids reading the full 8-bit code for every candidate. On this
workload, 2+6 provides the closest recall to traditional 8-bit RaBitQ while
improving search throughput by about 4.2 times; 1+7 maximizes throughput with
a larger recall trade-off.

### Search strategy comparison

The same in-memory split indexes were also searched with the default
`candidate_reorder` strategy (`factor = 10`) and the opt-in `heap` strategy.
The table reports 1,000 measured single-thread queries with the other settings
unchanged. "Supplement probes" is the average number of y-bit records read per
query.

| Layout | Strategy | Search QPS | Avg latency (ms) | Recall | Supplement probes/query |
| --- | --- | ---: | ---: | ---: | ---: |
| 1+7 | Candidate reorder | 101.009 | 9.896 | 0.8997 | 100.0 |
| 1+7 | Shared-quantized-IP heap | 66.826 | 14.961 | 0.9047 | 791.3 |
| 2+6 | Candidate reorder | 75.641 | 13.217 | 0.9128 | 100.0 |
| 2+6 | Shared-quantized-IP heap | 57.108 | 17.508 | 0.9145 | 421.2 |
| 3+5 | Candidate reorder | 67.481 | 14.816 | 0.9094 | 100.0 |
| 3+5 | Shared-quantized-IP heap | 53.983 | 18.521 | 0.9099 | 199.4 |

All heap supplement probes used the saved quantized x-bit inner product
directly; none fell back to reconstructing it from a distance hint. Removing
the former float-LUT pass over every scanned candidate improved heap QPS by
24.7%, 45.9%, and 63.9% for 1+7, 2+6, and 3+5 respectively. The shared LUT
quantization error changes heap ordering and reduced recall from the former
float-LUT heap results of 0.9160, 0.9231, and 0.9151. On this in-memory workload,
heap still reads more supplements than the fixed `factor * topk` pool and is
slower than candidate reorder, but the throughput gap is substantially smaller.

## Query lookup table and SIMD

For `x = 2` and `x = 3`, the query computer builds a FastScan-style byte
lookup table. Each table row corresponds to eight query coordinates and has
256 entries:

```text
LUT[block][byte_value]
    = sum of q_i for the set bits in byte_value within that 8-D block
```

Each filter plane then contributes one lookup per byte instead of decoding
eight coordinates separately. Binary weights combine the x planes into
`S_x`.

The AVX2 and AVX512 kernels gather multiple lookup entries at once and also
provide a batch-of-four path. The scalar implementation is kept as the
portable fallback. The relevant entry points are:

- `RaBitQFloatMultiBitIPByLookup`
- `RaBitQFloatMultiBitIPBatch4ByLookup`
- `RaBitQFloatBuildByteIPLookupTable`

An x-bit width outside the specialized set remains supported through the
generic bit-plane computation path.

## Reorder scans only y supplement bits

The full unsigned code satisfies:

```text
sum_i q_i * u_i
    = 2^y * sum_i q_i * f_i
      + sum_i q_i * s_i
```

For L2 with an x-bit lookup filter, HGraph, IVF, and Pyramid pass the previously computed
filter distance to reorder as a hint. `ComputeDistWithSplitCodeAndFilterDist`
recovers the first term from that hint and computes only the second term from
the y supplement planes:

```text
full contribution = shifted filter contribution + supplement contribution
```

Thus a `3+5` index reuses the 3-bit filter result and scans only 5 new bit
planes for each reordered candidate. If the hint is unavailable or cannot be
used, the code falls back to `ComputeDistWithSplitCode`, which computes the
same final distance directly from both split records.

## Memory, disk, and hybrid IO

Both records use the base IO type unless a separate supplement IO type is
configured.

### Both records in memory

```json
{
    "base_io_type": "block_memory_io"
}
```

### Both records on disk

```json
{
    "base_io_type": "async_io",
    "base_file_path": "/data/hgraph_rabitq_split"
}
```

VSAG creates separate backing paths for the filter and supplement records.

### Filter in memory, supplement on disk

```json
{
    "base_io_type": "block_memory_io",
    "base_supplement_io_type": "async_io",
    "base_file_path": "/data/hgraph_rabitq_split"
}
```

The supported mixed-IO combination keeps `x_bit_cell_` in block memory and
places `supplement_cell_` in async IO. During batched reorder, the filter
record is read by direct pointer while `MultiRead` fetches only supplement
records. `base_supplement_file_path` may be set explicitly; otherwise VSAG
derives a supplement path from `base_file_path`.

## Serialization and loading

Use the normal index-level serialization API. Applications do not need to
persist the two records independently.

```cpp
std::ofstream out("/path/to/index.bin", std::ios::binary);
auto serialized = index->Serialize(out);

auto loaded = vsag::Factory::CreateIndex("hgraph", index_params).value();
std::ifstream in("/path/to/index.bin", std::ios::binary);
auto deserialized = loaded->Deserialize(in);
```

The split datacell serializes, in order:

1. Base datacell state and supplement IO type.
2. Filter storage.
3. Supplement storage.
4. RaBitQ quantizer state.

Create the destination index with parameters compatible with the serialized
index, especially `dim`, `metric_type`, x/y bit widths, and query bits.
Changing an encoded parameter requires rebuilding the index. Tuning only the
search-time `hgraph.rabitq_error_rate` does not.

## Implementation map

| Area | File / entry point |
| --- | --- |
| External x/y parameter mapping | `src/algorithm/hgraph/hgraph_param_mapping.cpp` |
| Split record ownership and IO | `src/datacell/rabitq_split_datacell.h` |
| Plane layout and code splitting | `RaBitQuantizer::StoredPlaneIndex`, `SplitCode` |
| Filter estimate and lower bound | `ComputeDistWithOneBitLowerBound` |
| Direct split distance | `ComputeDistWithSplitCode` |
| Reorder using the filter-distance hint | `ComputeDistWithSplitCodeAndFilterDist` |
| Heap using saved quantized x-bit IP | `ComputeDistWithSplitCodeAndFilterInnerProduct` |
| SIMD dispatch | `src/simd/rabitq_simd.cpp` |
| AVX2 / AVX512 lookup kernels | `src/simd/avx2.cpp`, `src/simd/avx512.cpp` |
| Runnable memory/disk/hybrid example | `examples/cpp/323_index_hgraph_rabitq_split.cpp` |

## Operational notes

- Split storage is currently available on HGraph, IVF, and Pyramid and requires
  fp32 query codes. Pyramid enables the one-bit split search path by default for
  split indexes; pass `rabitq_one_bit_search: false` under `pyramid` to force the
  standard search path.
- `l2`, `ip`, and `cosine` are supported. IVF heap search uses the saved
  quantized x-bit inner product for all three metrics.
- Keep `use_reorder: true` unless x-bit traversal accuracy alone has been
  validated for the dataset.
- Changing x, y, metric, or transform parameters requires rebuilding the
  index. A search-time `hgraph.rabitq_error_rate` override does not.
- Use [RaBitQ](rabitq.md) for the general quantizer description and
  [HGraph](../indexes/hgraph.md) and [Pyramid](../indexes/pyramid.md) for the complete index parameter tables.
