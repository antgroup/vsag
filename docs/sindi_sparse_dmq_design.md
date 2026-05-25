# SINDI Direct 8-Bit DMQ Design

This note defines the VSAG SINDI `rerank_type="dmq"` implementation. The rerank backend now uses a
direct 8-bit DMQ code per stored nonzero. Sparse term id is treated as the DMQ dimension, so
quantization points are trained per term, not per vector.

## Sparse DMQ Estimator

For a stored sparse vector `x_i`, let `S_i` be its nonzero term set. SINDI stores a per-vector mean

```text
mu_i = mean(x_it), t in S_i
r_it = x_it - mu_i
```

For every sparse term `t`, the backend trains a 256-level codebook `C_t` from residual samples
`r_it` collected from the vectors that contain term `t`. Encoding chooses the closest codebook
level:

```text
z_it = argmin_k |r_it - C_t[k]|,  k in [0, 255]
u_it = C_t[z_it]
```

The per-vector DMQ rescale is

```text
alpha_i = ||r_i||^2 / <u_i, r_i>
```

with a zero fallback when the denominator is numerically zero. The decoded value used by rerank is

```text
xhat_it = mu_i + alpha_i * C_t[z_it]
```

For a sparse query `q`, only the intersection `I = S_i & S_q` contributes:

```text
approx_ip(x_i, q) = sum_{t in I} q_t * xhat_it
distance = 1 - approx_ip(x_i, q)
```

This is the direct multi-bit analogue of the DMQ residual estimator: the codebook is dimension
scoped, while `mu_i` and `alpha_i` remain vector scoped.

## Codebook Training

Initial `Build()` calls the DMQ rerank backend once with the accepted build batch. This is required
for term-level training because a term codebook needs residuals from all vectors in the batch, not a
single vector at a time.

The current scalar trainer follows the DMQ weighted quantile rule over sorted residuals for each
term. For a residual sample `v_i`, its weight is the sum of squared distances to all residual
samples of the same term. The 512 weighted partitions produce 255 separator thresholds and 256
representative decode values. Very short or constant term lists reuse the nearest available sample,
keeping every term codebook monotonic and fixed-size.

Incremental `Add()` keeps existing term codebooks fixed and trains codebooks only for newly observed
terms. This preserves existing encoded vectors and avoids silently changing earlier rerank scores.

## Storage Layout

The backend stores:

- sorted term ids for exact sparse intersection, bit-packed by `term_id_limit`;
- one 8-bit direct DMQ value code per stored nonzero;
- one `{mean, alpha}` factor pair per vector;
- one 256-float codebook per observed sparse term;
- a term-id to codebook-index map rebuilt after deserialization;
- a guarded direct term-id to codebook-index lookup table for common bounded term-id ranges.

The public option remains `rerank_type="dmq"`, but the DMQ backend now accepts only `dmq_bits=8`.
The old 1+x split stage-0/residual format was an intermediate experiment and is not the production
rerank path.

## Search Path

Queries are sorted once in the rerank context. If the query maximum term id is small enough, the
backend builds a direct lookup table from term id to query value. This avoids a sorted merge for the
common Wholenet path. Large term ids fall back to the sorted sparse merge path to avoid allocating a
huge lookup table.

For direct8 search, candidate scoring processes stored sparse terms in small blocks. Value codes are
loaded as bytes, term ids are unpacked sequentially from the packed bitstream, and codebook indices
are resolved through the direct lookup table when possible. These are decode-path optimizations only;
they do not change the DMQ estimator or the final rerank score.

## Serialization

Direct8 DMQ writes an internal magic value and version before encoded vectors, codes, and codebooks.
The version was bumped when replacing the old split 1+x storage, so older bytes are not silently
interpreted as direct8 DMQ bytes.