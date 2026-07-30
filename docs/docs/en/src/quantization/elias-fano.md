# Elias–Fano Ordered Integer Compression

This page documents the exact storage format, encoding and decoding algorithms implemented by
VSAG's `EliasFanoStream`, and how SINDI DMQ uses it to compress forward-store term IDs.

Elias–Fano (EF) is designed for monotone integer sequences with a known value range. The VSAG
implementation accepts **nondecreasing** sequences, including duplicates. It provides sequential
streaming decode but does not build a select index for random access.

## Preconditions and notation

The input sequence is:

```text
0 <= x_0 <= x_1 <= ... <= x_(N-1) < U
```

where:

- `N` is the number of elements, or the number of non-zero terms in one document;
- `U` is the exclusive universe bound, equal to the compact-term vocabulary size in DMQ;
- `x_i` is compact term ID `i`.

An empty sequence has a zero-byte payload. When `N>0`, `U` must be positive.

## 1. Computing the layout

EF splits every integer into high and low parts. The low width is:

```text
l = floor(log2(U / N)), with l = 0 when U / N <= 1
```

VSAG computes `U/N` with integer division and then takes its floor base-2 logarithm.

The low region contains `N` tightly packed `l`-bit integers:

```text
low_bits_bytes = ceil(N * l / 8)
```

The valid bit count and stored byte count of the high bitmap are:

```text
high_bits_count = ((U - 1) >> l) + N + 1
high_bits_bytes = ceil(high_bits_count / 8)
```

The complete payload size is:

```text
EF_bytes = low_bits_bytes + high_bits_bytes
```

`EliasFanoStreamLayout` stores these four results. A caller can compute the layout once and reuse
it for size calculation, encoding, and reader construction.

## 2. Encoding format

### 2.1 Split high and low parts

For every `x_i`:

```text
low_i  = x_i & (2^l - 1)
high_i = x_i >> l
```

When `l=0`, there is no low region, `low_i=0`, and all information is in the high bitmap.

### 2.2 Low region

Each `low_i` is tightly packed in sequence order, starting at the least-significant bit of each
byte:

```text
bit_offset(low_i) = i * l
```

An integer may cross a byte boundary. Unused bits at the end of the region remain zero.

### 2.3 High region

The high parts use a unary/select representation. The set-bit position for element `i` is:

```text
p_i = high_i + i
```

Because `high_i` is nondecreasing, `p_i` is strictly increasing and every element has a unique
one bit. The `+i` term distinguishes elements even when `x_i == x_(i-1)`.

The final in-memory layout is:

```text
+-----------------------------+----------------------------------+
| low bits                    | high bitmap                      |
| N tightly packed l-bit ints | set the i-th one at high_i + i   |
+-----------------------------+----------------------------------+
```

The payload does not embed `N`, `U`, `l`, or an encoding tag. The caller must supply `N` and `U`,
or a previously computed layout.

## 3. Complete encoding example

Let:

```text
U = 16
N = 4
x = [3, 5, 8, 13]
```

The low width is:

```text
l = floor(log2(16 / 4)) = 2
```

The split is:

| `i` | `x_i` | `low_i` | `high_i` | `p_i = high_i + i` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 3 | 0 | 0 |
| 1 | 5 | 1 | 1 | 2 |
| 2 | 8 | 0 | 2 | 4 |
| 3 | 13 | 1 | 3 | 6 |

Packing the low parts LSB-first produces `0x47`. Setting high-bitmap positions 0, 2, 4, and 6
produces `0x55`:

```text
payload = [0x47, 0x55]
           low    high
```

The total is two bytes. Fixed-width coding with `ceil(log2 U)=4` bits also needs two bytes, so
the DMQ hybrid strategy selects packed coding for this example because ties do not select EF.

## 4. Encoding algorithm

`EliasFanoStream::Encode` performs:

```text
layout = GetLayout(N, U)
clear layout.SizeInBytes() bytes

for i in [0, N):
    verify x_i < U
    verify i == 0 or x_(i-1) <= x_i
    write low_i
    high_bitmap[high_i + i] = 1
```

The current low-part writer processes one bit at a time, giving an encoding cost of `O(N*l)`.
Because inputs are `uint32_t` and `l <= 31`, this can also be viewed as `O(N)` in the number of
elements.

The encoder clears the entire output first, adding initialization work proportional to payload
bytes. It uses `O(1)` auxiliary space beyond the output buffer.

## 5. Scalar streaming decode

`EliasFanoStreamReader` maintains separate low and high state:

- `low_cursor_`, `low_buffer_`, and `low_available_bits_`;
- `high_cursor_`, `high_buffer_`, and `high_available_bits_`;
- `high_base_position_`, the current 64-bit high block's position in the full bitmap;
- `index_`, the next element index.

A `Read()` call follows this order:

### 5.1 Find the next high part

If `high_buffer_` is zero, the reader loads up to eight bytes from the high region. It assembles
the `uint64_t` with explicit byte shifts, making the operation independent of host endianness.

If a loaded 64-bit block is still zero, the reader advances `high_base_position_` and continues
until it finds a block containing a set bit. It then performs:

```text
bit_offset = count_trailing_zeros(high_buffer)
high_buffer &= high_buffer - 1
p_i = high_base_position + bit_offset
high_i = p_i - i
```

Under GCC and Clang, `count_trailing_zeros` maps to `__builtin_ctzll`, which normally generates a
hardware bit-scan/tzcnt instruction. `buffer &= buffer-1` clears the lowest consumed one bit.

### 5.2 Read the low part

When the low buffer contains fewer than `l` bits, the reader appends bytes until a complete
`low_i` is available:

```text
low_i = low_buffer & low_mask
low_buffer >>= l
```

For `l=0`, it returns zero without accessing the low region.

### 5.3 Reconstruct the integer

```text
x_i = (high_i << l) | low_i
```

The reader then increments `index_`. Calling `Read()` after reaching `N` is an error.

## 6. Batch decode

`ReadBatch(values, max_count)` returns at most
`EliasFanoStreamReader::MAX_BATCH_SIZE`, currently eight. It uses two passes:

1. extract all `high_i` values for the batch;
2. restore the batch's starting `index_`, read all `low_i` values, and reconstruct the results.

This organization reduces alternation between high and low state and provides an entry point for
future unrolling or SIMD work. Scalar and batch calls can be mixed, and the final partial batch
returns its actual size.

`ReadBatch` has correctness tests, but SINDI DMQ query reranking now uses
`EliasFanoSeekReader` to locate query-term ordinals directly instead of using this batch API.
The sequential reader remains in use for complete vector decode and vector-to-vector merge.

## 7. Complexity

### Space

The implementation's space bound is:

```text
N*l + O(N) bits
```

For a strictly increasing sequence, where `N<=U`, this is the classic
`N*floor(log2(U/N))+O(N)` Elias–Fano bound. The `l=0` form also covers the `N>U` case
made possible by duplicate values.

The exact byte count in this implementation is:

```text
ceil(N*l/8) + ceil((((U-1)>>l) + N + 1)/8)
```

### Time

| Operation | Time | Auxiliary space |
| --- | --- | --- |
| Layout calculation | `O(log U)`, bounded by the 32-bit integer width | `O(1)` |
| Encoding | current implementation `O(N*l)` | `O(1)` |
| Decode all IDs sequentially | `O(N + high_bits_count/word_bits)` | `O(1)` |
| Amortized sequential decode per ID | `O(1)` | `O(1)` |
| Seek ordered query terms | `O(H_s/w + sum_q log(k_q+1) + M)` | `O(1)` |
| Random access to ID `i` | not currently provided | — |

Every high-bitmap machine word is loaded at most once and every one bit is consumed once, making
a complete sequential scan linear. A large gap between adjacent high parts can make the reader
skip several all-zero words; that work is included in the `high_bits_count/word_bits` term.

## 8. SINDI DMQ hybrid strategy

DMQ supports both fixed-width packed and EF coding. For compact-term vocabulary size `V`, the
fixed width is:

```text
b = max(1, ceil(log2 V))
packed_bytes = ceil(N*b/8)
```

Each document uses:

```text
EF_bytes < packed_bytes  -> Elias–Fano
otherwise                -> packed
```

There is no extra format tag. During decode, the header's `N` and the model's `V` reproduce the
same deterministic comparison, yielding the representation and the following value-code offset.
An empty vector has no ID payload.

The hybrid layout is computed once when a candidate enters distance computation and reused to:

- select `EliasFanoStreamReader` or `PackedReader`;
- obtain the ID payload byte count;
- locate the DMQ value codes.

This removes the earlier implementation's four layout calculations for the same candidate.

Here, `H_s` is the number of high-bitmap bits scanned through the greatest query high part,
`k_q` is the size of the query term's high bucket, and `M` is the number of matches.

## 9. Query-driven seek and performance

A fixed-width packed reader mainly performs loads, shifts, and masks. For every ID, EF must also:

- maintain two independent bit streams;
- locate the next one bit in the high bitmap;
- clear the consumed bit;
- refill across 64-bit boundaries and skip all-zero blocks;
- compute `p_i-i` and combine the high and low parts;
- handle additional data dependencies and branches.

Scanning every candidate's complete ID sequence repeats these operations across roughly 1000
documents when `n_candidate=1000`. Query scoring therefore now uses
`EliasFanoSeekReader`:

1. process query compact IDs in ascending order;
2. treat zero bits in the high bitmap as high-bucket delimiters;
3. use 64-bit `popcount` to skip a complete bitmap word at once;
4. derive the bucket's ordinal range from the number of consumed one bits;
5. binary-search low bits within that range;
6. access the DMQ value code at the matching ordinal directly.

This path no longer reconstructs every candidate ID and does not change the EF payload or index
format. It is a scalar word-level skipping optimization, not an enabled SIMD implementation.
Further SIMD work must still handle variable-length high bitmaps, block boundaries, and
dependencies in the query-matching loop.

## 10. Validation and edge cases

The implementation validates:

- `U > 0` for a non-empty sequence;
- non-null input, output, and batch pointers when they are used;
- every value satisfies `x_i < U`;
- the input is nondecreasing;
- the high bitmap does not end before `N` one bits are read;
- `Read()` does not pass the sequence end;
- a `ReadBatch` request does not exceed eight elements.

Duplicate IDs are valid. Unordered IDs and IDs equal to or greater than `U` are rejected.

## 11. Implementation and tests

- `src/impl/elias_fano_stream.h`: layout, encoder, and reader interfaces;
- `src/impl/elias_fano_stream.cpp`: layout, encoding, scalar decode, and batch decode;
- `src/impl/elias_fano_stream_test.cpp`: ordered round trips, duplicates, batch reads, empty high
  blocks, mixed scalar/batch calls, and invalid-input tests;
- `src/quantization/sparse_quantization/sparse_dmq_quantizer.cpp`: hybrid packed/EF selection and
  SINDI DMQ distance computation;
- [DMQ](dmq.md): value quantization, forward layout, and rerank scoring.
