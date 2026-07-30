# DMQ (Distribution Maintenance Quantization for Sparse Vectors)

DMQ (Distribution Maintenance Quantization) is the sparse-vector quantization method implemented
by VSAG for the [SINDI](../indexes/sindi.md) rerank forward store. The current implementation is
named `dmq8`: every non-zero value uses an 8-bit code, while each document stores an additional
mean and scale factor. Term IDs are mapped to compact IDs and then encoded with either fixed-width
bit packing or Elias–Fano coding.

This page describes the algorithm implemented by `SparseDmqQuantizer` in VSAG. DMQ is not
currently a general quantizer selectable through `base_quantization_type` for dense indexes.

## Background

SINDI uses inverted lists to retrieve candidates. When `use_reorder` is enabled, it also stores a
complete sparse vector for each document so that candidate inner products can be recomputed. If a
document contains `L` non-zero entries, storing `uint32` term IDs and `float32` values directly
requires about `8L` bytes for the two arrays alone. At corpus scale, this forward copy can become
a significant memory cost.

Global uniform quantization does not exploit two properties of sparse retrieval data:

- value distributions may differ substantially across terms;
- values within one document usually have a document-specific center and scale.

DMQ therefore combines three levels of compression:

1. **Per-document centering:** store a document mean and convert values to residuals;
2. **Per-term modeling:** use an independent codebook for frequent terms and optionally share one
   among infrequent terms;
3. **Per-document calibration:** store a scale factor that adapts codebook representatives to the
   document.

DMQ also stores compact IDs only for terms observed during training and compresses the sorted ID
sequence within each document.

## Notation

Let the training corpus be `D`. The sparse vector for document `d` is

```text
x_d = {(t_{d,i}, x_{d,i}) | i = 0, ..., L_d - 1}
```

where:

- `L_d` is the number of non-zero entries;
- `t_{d,i}` is a term ID;
- `x_{d,i}` is its value;
- `K = 2^8 = 256` is the number of DMQ8 codewords;
- `τ` is the shared-codebook threshold for infrequent terms, 1024 by default.

## Training

### 1. Center each document

DMQ first computes the mean of the non-zero values in every training document:

```text
μ_d = (1 / L_d) Σ_i x_{d,i}
```

It then computes residuals:

```text
r_{d,i} = x_{d,i} - μ_d
```

The mean of an empty vector is defined as zero. Only non-zero entries participate; absent sparse
dimensions are not treated as zeros in this calculation.

### 2. Assign term buckets and share codebooks

Let `f_t` be the number of occurrences of term `t` in the training corpus. VSAG assigns
training buckets as follows:

```text
f_t <= τ    -> all infrequent terms use one shared bucket
f_t >  τ    -> term t uses its own bucket
```

Each bucket collects all residuals for its terms and trains one 256-entry codebook. Setting
`dmq_shared_codebook_threshold: 0` gives every observed term an independent codebook.

Sharing avoids storing a roughly 2 KiB codebook for every term with very few samples and combines
their training data. The trade-off is that distinct infrequent terms use the same residual model.

### 3. Build a distribution-weighted codebook

Suppose a bucket contains `N` residuals, sorted as
`z_0 ≤ z_1 ≤ ⋯ ≤ z_{N-1}`. The implementation assigns each sample the weight:

```text
w_i = Σ_j (z_i - z_j)²
    = N z_i² + Σ_j z_j² - 2 z_i Σ_j z_j
```

The total weight is:

```text
W = Σ_i w_i
```

`w_i` measures the squared distance from residual `z_i` to the bucket distribution. Samples
near the distribution body receive less weight, while tail samples receive more. The codebook
therefore preserves more resolution in the distribution tails than an ordinary equal-frequency
partition.

Define the cumulative weight `C_i = Σ_{j=0}^{i} w_j`. For
`k=0,…,K-1`, the representative value is:

```text
c_k = z_min{i : C_i >= ((2k + 1) / (2K)) W}
```

For `k=0,…,K-2`, the encoding threshold is:

```text
θ_k = z_min{i : C_i >= ((k + 1) / K) W}
```

The 256 representatives lie at weighted midpoints between the 255 thresholds, which map residuals
to `[0, 255]`. If all residuals in a bucket are equal and `W` is effectively zero, every
representative and threshold is set to that residual.

## Encoding

### 1. Compact term IDs

After training, all observed terms are sorted by original ID and mapped to compact IDs in
`[0, V - 1]`, where `V` is the number of distinct terms. Fixed-width coding requires:

```text
b = max(1, ceil(log₂ V))
```

A length-`L` ID sequence then occupies:

```text
B_packed = ceil(Lb / 8) bytes
```

The current serialization format also computes the Elias–Fano size for the same sorted sequence
and selects the smaller representation per vector. Let:

```text
l = floor(log₂(V / L)), with l = 0 when V / L <= 1
```

The current Elias–Fano layout occupies:

```text
B_low  = ceil(Ll / 8)
B_high = ceil((((V - 1) >> l) + L + 1) / 8)
B_EF   = B_low + B_high
```

Elias–Fano is selected only when `B_{EF} < B_{packed}`; ties use bit packing.
The Elias–Fano formula above assumes `L > 0`; an empty vector has a zero-byte ID payload.
See [Elias–Fano Ordered Integer Compression](elias-fano.md) for the complete byte layout,
worked example, streaming reader state, and complexity analysis.

#### Elias–Fano data layout

Let a document contain the ordered compact IDs
`x_0 ≤ x_1 ≤ ... ≤ x_{L-1} < V`. Each ID is split at bit `l`:

```text
low_i  = x_i & (2^l - 1)
high_i = x_i >> l
```

The encoded result contains two consecutive regions:

```text
+-----------------------------+----------------------------------+
| low bits                    | high bits                        |
| L tightly packed l-bit ints | set bit at position high_i + i   |
+-----------------------------+----------------------------------+
```

The low region stores each `low_i` in ID order. The high region does not store `high_i`
directly. Instead, its `i`-th one bit is placed at:

```text
p_i = high_i + i
```

Because the IDs are nondecreasing, `p_i` is strictly increasing. Duplicate IDs therefore still
occupy distinct positions in the high bitmap. Its length is:

```text
((V - 1) >> l) + L + 1 bits
```

For example, with `V=16` and IDs `[3, 5, 8, 13]`, `L=4` and `l=2`:

```text
low  = [3, 1, 0, 1]
high = [0, 1, 2, 3]
p    = [0, 2, 4, 6]
```

Bits 0, 2, 4, and 6 are consequently set in the high bitmap.

The current format does not store a per-document packed/EF tag. The decoder obtains `L` from the
header and `V` from the DMQ model. Applying the same size comparison as the encoder determines
both the ID representation and the start offset of the value codes.

#### Elias–Fano encoding

The encoder first calls `GetLayout(L, V)` to compute `l` and both region sizes, then clears the
output:

1. verify that every ID is less than `V` and that the sequence is nondecreasing;
2. write `low_i` at bit offset `i·l` in the low-bit stream;
3. set bit `high_i+i` in the high bitmap.

The current `store_packed` implementation writes low bits one at a time, so its implementation
cost is `O(L·l)`. With the fixed 32-bit ID bound, this can be treated as `O(L)`. The output space
is:

```text
O(L log₂(V/L) + L) bits
```

Elias–Fano is most useful when `L` is small relative to `V` and the IDs are ordered. The hybrid
strategy still compares actual byte counts for every document, avoiding EF whenever it does not
save space.

#### Elias–Fano streaming decode

The reader maintains separate cursors for the low and high regions:

1. extract the next `low_i` from the low-bit stream;
2. find the next set-bit position `p_i` in the high bitmap;
3. compute `high_i = p_i - i`;
4. reconstruct `x_i = (high_i << l) | low_i`.

`EliasFanoStreamReader` loads up to eight bytes of the high bitmap into a 64-bit buffer. It uses
a trailing-zero count to find the lowest set bit, then clears it with `buffer &= buffer - 1`.
When the buffer is zero, it skips that block and loads subsequent bytes. A separate 64-bit buffer
refills the low-bit stream on demand.

Sequentially decoding all `L` IDs costs:

```text
O(L + high_bits_count / machine_word_bits)
```

This is amortized `O(1)` per ID with `O(1)` reader state. It does not imply constant-time random
access: the current reader has no select index, so reaching ID `i` requires continuing from the
current stream position.

The reader also exposes `ReadBatch`, which returns at most eight IDs per call. It extracts a group
of high parts followed by their low parts and can be mixed with scalar `Read` calls. SINDI DMQ
query reranking now uses `EliasFanoSeekReader` to locate query-term ordinals through high-bitmap
buckets; `ReadBatch` remains an independent sequential-decode capability.

#### Decode performance versus fixed-width packing

A fixed-width packed decoder mainly performs loads, shifts, and masks. For every ID, EF must also
maintain two bit streams, locate the next high one bit, handle refills across 64-bit blocks, and
reconstruct `p_i-i`. Long zero runs in the high bitmap add refill work and branches.

A per-ID scan reconstructs every candidate's complete forward ID sequence. With
`n_candidate=1000`, that work repeats roughly 1000 times per query. The current seek path instead
scans ordered query terms over 64-bit high-bitmap words, skips complete words with `popcount`,
searches low bits only in target high buckets, and returns matching value-code ordinals directly.
The hybrid layout is computed once when a candidate enters distance computation and is reused to
select the ID reader and locate the value codes; it is no longer recomputed four times for the
same candidate.

### 2. Quantize values

For a document `d` being encoded, DMQ recomputes its non-zero value mean `μ_d` and residuals
`r_{d,i}`. Using the thresholds from the codebook associated with each term, it encodes:

```text
q_{d,i} = min{k | r_{d,i} <= θ_k}, or q_{d,i} = K - 1 if no such k exists
```

This matches the implementation's `lower_bound` over 255 thresholds. Let the representative for
the resulting code be:

```text
ĉ_{d,i} = c_{q_{d,i}}
```

DMQ then computes one scale factor for the entire document:

```text
             Σ_i r_{d,i}²
α_d = ---------------------------
             Σ_i ĉ_{d,i} r_{d,i}
```

If the absolute denominator is at most `10^{-12}`, `α_d` is set to zero. This calibration
ensures:

```text
Σ_i r_{d,i}(α_d ĉ_{d,i}) = Σ_i r_{d,i}²
```

The scaled quantized residual therefore preserves the original residual's projection along its
own direction. This differs from an ordinary least-squares scale that minimizes mean squared
reconstruction error.

### 3. Per-vector layout

A document with `L` non-zero entries is encoded as:

```text
+----------------------+----------------------+------------------+
| header (12 bytes)    | compact term IDs     | value codes      |
| len, μ_d, α_d        | packed or Elias–Fano | L bytes          |
+----------------------+----------------------+------------------+
```

Its encoded size is:

```text
B_DMQ = 12 + min(B_packed, B_EF) + L bytes
```

An empty vector stores only the 12-byte header.

Actual memory also includes one offset per document, global term mappings, and codebooks. A
codebook contains 255 `float32` thresholds and 256 `float32` representatives, or 2044 bytes
excluding container overhead.

## Decoding and rerank scoring

### Decoding

Using the term's codebook, DMQ reconstructs an approximate value as:

```text
x̂_{d,i} = μ_d + α_d c_{q_{d,i}}
```

The compact ID is also mapped back to the original term ID.

### Query scoring

SINDI supports inner product only. For a query `q=\{(t,q_t)\}`, query preparation builds a
256-entry lookup table for every known query term:

```text
LUT[t, k] = q_t c_{t,k}
```

The approximate inner product for candidate document `d` is:

```text
IP(q, x̂_d)
  = Σ_{t ∈ supp(q) ∩ supp(d)} q_t (μ_d + α_d c_{t,q_{d,t}})
  = μ_d Σ_t q_t + α_d Σ_t LUT[t, q_{d,t}]
```

VSAG returns:

```text
distance(q, d) = 1 - IP(q, x̂_d)
```

The implementation accumulates the mean and LUT terms directly from compressed data, without
fully decoding candidate vectors. Query terms absent from the trained DMQ vocabulary do not
contribute. An empty support intersection has inner product zero and distance one.

## Algorithm flow

```text
Build
raw sparse vectors
  -> compute per-document means and residuals
  -> assign independent/shared training buckets by term frequency
  -> train a 256-entry distribution-weighted codebook per bucket
  -> map term IDs to compact IDs
  -> encode IDs, 8-bit value codes, mean, and scale per document

Search
SINDI inverted-list candidate retrieval
  -> obtain n_candidate candidates
  -> build the query's DMQ code LUT
  -> scan compressed forward codes and compute approximate inner products
  -> sort by 1 - inner_product and return top-k
```

## Configuration

DMQ8 can only be used as SINDI's rerank forward format:

```json
{
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "term_id_limit": 30000,
        "window_size": 50000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "rerank_type": "dmq8",
        "dmq_shared_codebook_threshold": 1024
    }
}
```

`use_quantization` controls the inverted posting value format, while `rerank_type` controls the
rerank forward copy. They are independent. A common combination uses SQ8 postings and a DMQ8
forward store.

## Trade-offs and limitations

- **Accuracy:** DMQ is lossy. Recall changes depend on the data distribution, candidate count, and
  pruning parameters and should be evaluated on the target dataset.
- **Memory:** values shrink from four bytes to one and term IDs are compressed. Savings must be
  measured after including the 12-byte per-vector header, offsets, term mappings, and codebooks.
- **Codebook sharing:** increasing `dmq_shared_codebook_threshold` reduces codebook memory but may
  fit term-specific distributions less accurately.
- **Inner product only:** `SparseDmqQuantizer` currently fixes its metric to inner product.
- **Offline model only:** SINDI fixes the DMQ codebooks during the initial build and does not
  support incremental `Add` or `UpdateVector`.
- **Variable-length codes:** documents differ in non-zero count and ID representation, so DMQ does
  not support the generic fixed-length batch encode/decode interface.

## Related implementation

- `src/quantization/sparse_quantization/sparse_dmq_quantizer.{h,cpp}`: training, encoding,
  decoding, and distance computation;
- `src/datacell/sparse_dmq_datacell.{h,cpp}`: variable-length code storage, offsets, and
  serialization;
- `src/impl/elias_fano_stream.{h,cpp}`: Elias–Fano coding for ordered compact IDs;
- [SINDI](../indexes/sindi.md): index flow, build parameters, and search parameters.
