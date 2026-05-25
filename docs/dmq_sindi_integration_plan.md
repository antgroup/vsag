# DMQ Integration Plan for SINDI

## Goal

Integrate DMQ into VSAG so SINDI can keep high recall while reducing memory.

The immediate target is SINDI rerank storage. Today `use_quantization` only
compresses values in SINDI posting lists. When `use_reorder` is enabled, SINDI
also stores a full fp32 sparse copy in `SparseIndex` for precise reranking.
That fp32 copy keeps recall high, but it dominates memory on large sparse data.

For the current Wholenet Sparse 10M fp32 rerank baseline:

- Dataset: `/root/data/wholenet-sparse-10m-ip.hdf5`
- Build params: `use_quantization=true`, `use_reorder=true`
- Index memory: about `4.87 GB`
- Index file: about `5.20 GB`
- Build peak RSS: about `10.51 GB`
- Search peak RSS: about `10.29 GB`
- Recall avg after the recall fix: about `0.9482`
- QPS: about `797`

The integration must reduce the rerank memory without giving up the high-recall
behavior that `use_reorder` currently provides.

Detailed follow-up research on the full DMQ estimator, low-bit FastScan kernels,
and high-bit low-rank SIMD acceleration is recorded in
[`docs/dmq_math_simd_research.md`](dmq_math_simd_research.md).

## Current SINDI Rerank Path

Current build path:

1. `SINDI::Add` writes each sparse vector into windowed `SparseTermDataCell`
   posting lists.
2. If `use_reorder` is true, it also writes the original sparse vector into
   `rerank_flat_index_`.
3. `rerank_flat_index_` is a `SparseIndex`.
4. `SparseIndex` stores `[len][uint32 term ids][float values]` for every vector.

Current search path:

1. SINDI posting lists collect approximate candidates.
2. If `use_reorder` is true, `SINDI::search_impl` sorts the query once.
3. It calls `SparseIndex::CalDistanceByIdUnsafe` for each candidate.
4. The final heap uses those high-precision distances.

This is a good integration seam: keep the coarse SINDI inverted-list search
unchanged and replace only the rerank storage/computation backend.

## DMQ Source State

`/root/vsag/DMQ-main` contains an Apache-2.0 prototype library named
`rabitqlib`. It has useful DMQ code, but it is not ready to import wholesale:

- It is mostly header-only.
- It includes its own Eigen copy.
- It uses OpenMP in several paths.
- Its CMake defaults to `-march=native` and `-Ofast`.
- Much of the code is IVF/sample oriented.
- Several files are experimental copies or benchmark entrypoints.

The VSAG integration should extract the algorithmic core into VSAG-style code
instead of adding all of `DMQ-main` as a dependency.

## Proposed First Architecture

SINDI now has a rerank backend boundary with two implementations:

- `fp32`: existing behavior, backed by `SparseIndex`.
- `dmq`: compressed sparse value-code backend.

Suggested SINDI parameter shape:

```json
{
  "index_param": {
    "use_reorder": true,
    "rerank_type": "dmq",
      "dmq_bits": 8
  }
}
```

Default behavior remains `fp32` for backward compatibility. The `dmq` rerank backend now accepts
only direct 8-bit mode.

The new backend should provide the small surface SINDI actually needs:

- Add one or more sparse vectors.
- Compute distance for a query and an inner id.
- Serialize and deserialize.
- Report memory usage.
- Optionally decode/get sparse vector for compatibility APIs.

## Sparse DMQ Backend Scope

For SINDI, vectors are sparse and variable length. The first DMQ backend should
therefore be sparse-native rather than the dense IVF implementation from
`DMQ-main`.

The current `dmq` backend now uses direct 8-bit sparse-DMQ storage:

- Keep sorted term ids for exact sparse intersection.
- Store term ids as bit-packed codes derived from `term_id_limit`.
- Treat sparse term id as the DMQ dimension.
- Train a 255-threshold / 256-representative codebook per observed sparse term from residuals
   `x_t - mean(x)`.
- Use the DMQ weighted quantile rule for those term-level quantization points.
- Store one 8-bit direct code per nonzero and per-vector `{mean, alpha}` factors.
- Compute the sparse IP estimator `sum(q_t * (mean + alpha * C_t[code_t]))` during candidate
   rerank.
- Use a guarded direct query lookup for query max term id up to 1,000,000;
   larger term ids fall back to the sorted sparse merge path.

The backend writes a sparse-DMQ magic and version before encoded vectors, so old
simple-DMQ or split 1+x bytes are not silently interpreted as the direct8 format.

This remains sparse-native and does not import `DMQ-main` wholesale. Dense IVF
FastScan layouts remain reference material only; SINDI still needs sparse
intersection, packed-id unpack, and query-lookup acceleration first.

If term-id memory remains too high after value compression, add a second step
for sorted term-id delta compression. That should be separate from the first
DMQ milestone so recall and memory changes are easier to attribute.

## Milestones

1. Refactor SINDI rerank access behind a small backend interface.
   Keep the fp32 backend behavior identical. Done.
2. Add benchmark fields to report SINDI posting memory and rerank backend memory
   separately. Done.
3. Implement direct 8-bit sparse-DMQ value-code backend for IP rerank. Done with exact bit-packed
   term ids, term-level weighted quantile codebooks, 8-bit value codes, and per-vector factors.
4. Add unit tests for encode/decode, distance estimation, serialization, and
   SINDI search equivalence at small scale. Done for direct8 validation, rejection of non-8-bit
   DMQ modes, direct query lookup, and large-term-id fallback.
5. Run Wholenet Sparse 10M with the direct benchmark script and compare:
   memory, index file size, peak RSS, recall, QPS, and latency. Done for
   `dmq_bits=8`.
6. Tune `n_candidate`, query lookup, and SIMD/block decode to keep recall near the fp32 rerank
   baseline while recovering QPS. Done for the first production hot path: direct code loads,
   direct codebook-index lookup, 32-term scorer blocks, and sequential packed-id block unpack.

## Wholenet Sparse 10M Result

Latest direct benchmark runs use `query_prune_ratio=0.2`, `topk=10`, and 10,000 searched queries:

- fp32 result directory: `sindi_wholenet_sparse_10m_20260519_123839`
- direct8 weighted DMQ result directory:
   `sindi_wholenet_direct8dmq_weighted_20260520_090022`

| Metric | fp32 rerank | direct8 DMQ, n=100 | direct8 DMQ, n=200 optimized |
| --- | ---: | ---: | ---: |
| Build time | 55.2489 s | 202.0537 s | same index |
| Index memory | 4.87 GB | 3.38 GB | 3.18 GB loaded |
| Rerank backend memory | 3.29 GB | 1.79 GB build / 1.60 GB loaded | 1.60 GB loaded |
| Index file size | 5.20 GB | 3.70 GB | same index |
| Build peak RSS | 10.51 GB | 8.54 GB | same index |
| Search peak RSS | 10.29 GB | 8.00 GB | 8.00 GB |
| Recall avg | 0.9482 | 0.9319 | 0.9499 |
| QPS | 796.5264 | 562.8659 | 659.2839 |
| Latency avg | 1.2555 ms | 1.7766 ms | 1.5168 ms |
| Latency p95 | 2.5420 ms | 3.0497 ms | 2.7826 ms |
| Latency p99 | 3.5248 ms | 4.0635 ms | 3.7859 ms |

The direct8 backend now follows the intended term-level DMQ quantization semantics. The codebooks
add about 119 MB loaded memory for 61,150 observed terms, but the rerank backend still drops from
about 3.29 GB fp32 to about 1.60 GB loaded. With `n_candidate=200`, recall slightly exceeds the fp32
baseline while loaded index memory remains about 35% lower. The optimized hot path recovers most of
the earlier scalar decode and lookup loss at the wider candidate settings.

Search-only sweep on the same direct8 index:

| n_candidate | Recall avg | QPS | Latency avg | Latency p95 | Latency p99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.9319 | 562.8659 | 1.7766 ms | 3.0497 ms | 4.0635 ms |
| 150 | 0.9431 | 482.91 | 2.0708 ms | 3.3581 ms | 4.3784 ms |
| 175 | 0.9469 | 455.72 | 2.19 ms | 3.50 ms | 4.51 ms |
| 200 | 0.9499 | 424.34 | 2.3566 ms | 3.6860 ms | 4.7299 ms |
| 500 | 0.9660 | 245.54 | 4.0727 ms | 5.4767 ms | 6.6914 ms |

Hot-path optimized search-only runs on the same direct8 index:

| n_candidate | Recall avg | QPS before | QPS optimized | Latency before | Latency optimized |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 175 | 0.9469 | 455.72 | 684.61 | 2.19 ms | 1.4607 ms |
| 200 | 0.9499 | 424.34 | 659.28 | 2.3566 ms | 1.5168 ms |

The compute-side implementation keeps exact direct8 scoring semantics: it only changes how packed
ids, value codes, query values, and codebook indices are loaded in the rerank loop.

Stage-level profiling is recorded in [`sindi_search_stage_profile.md`](sindi_search_stage_profile.md).
At `n_candidate=200`, about 79% of profiled time is still in inverted-list candidate generation,
about 15% is direct8 DMQ forward scoring, and about 3.4% is final reorder heap work. This makes the
next optimization target the coarse inverted path, especially duplicated posting-list scans between
accumulation and candidate heap insertion.

## Acceptance Target

Use the current direct benchmark as the baseline:

```bash
bash benchs/run_sindi_wholenet_direct.sh
```

Run the compressed rerank variant with:

```bash
RERANK_TYPE=dmq DMQ_BITS=8 bash benchs/run_sindi_wholenet_direct.sh
```

The first useful DMQ result should satisfy:

- Recall avg remains close to the fp32 rerank baseline, currently about `0.9482`.
- Index memory and index file size drop measurably.
- Search peak RSS drops measurably.
- QPS and latency do not regress enough to erase the memory win.
