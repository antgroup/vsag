# RaBitQ x+y Split

RaBitQ x+y split is an HGraph storage and search mode for low-bit base codes.
Each vector is divided into two records:

- `x` filter bits are read during graph traversal and lower-bound filtering.
- `y` supplement bits are fetched only for candidates that reach reorder.
- The final reorder distance uses all `x+y` bits.

This layout keeps the traversal record small while retaining a higher-precision
RaBitQ distance for final ranking. It also allows the filter record to stay in
memory while the colder supplement record is stored on disk.

## Enable split mode

HGraph selects split mode when both quantization types are `rabitq` and
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

If `rabitq_bits_per_dim_precise` is omitted, HGraph uses the standard RaBitQ
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

## Search pipeline

The split search path has four stages:

1. The query is transformed and normalized once. For supported filter widths,
   a byte lookup table is also built once per query.
2. Graph traversal reads only the filter record. It computes an x-bit distance
   estimate and a conservative lower bound for each visited vector.
3. Reorder discards candidates whose lower bound cannot enter the result set.
   It fetches the y-bit supplement record only for the remaining candidates.
4. The final distance combines the filter contribution and supplement
   contribution into one `x+y`-bit RaBitQ estimate.

The HGraph heap is therefore not populated with an `x+y` distance for every
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

For L2 with an x-bit lookup filter, HGraph passes the previously computed
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
| Reorder using the filter hint | `ComputeDistWithSplitCodeAndFilterDist` |
| SIMD dispatch | `src/simd/rabitq_simd.cpp` |
| AVX2 / AVX512 lookup kernels | `src/simd/avx2.cpp`, `src/simd/avx512.cpp` |
| Runnable memory/disk/hybrid example | `examples/cpp/323_index_hgraph_rabitq_split.cpp` |

## Operational notes

- Split storage is currently an HGraph feature and requires fp32 query codes.
- `l2`, `ip`, and `cosine` are supported. The filter-hint reorder shortcut is
  currently specialized for L2.
- Keep `use_reorder: true` unless x-bit traversal accuracy alone has been
  validated for the dataset.
- Changing x, y, metric, or transform parameters requires rebuilding the
  index. A search-time `hgraph.rabitq_error_rate` override does not.
- Use [RaBitQ](rabitq.md) for the general quantizer description and
  [HGraph](../indexes/hgraph.md) for the complete index parameter table.
