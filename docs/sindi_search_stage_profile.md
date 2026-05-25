# SINDI Direct8 DMQ Search Stage Profile

This note profiles the SINDI direct8 DMQ search path on Wholenet Sparse 10M. It focuses on the
search-time path from inverted-list candidate generation, to forward DMQ rerank scoring, to final
reorder heap maintenance.

## Run Context

- Dataset: `/root/data/wholenet-sparse-10m-ip.hdf5`
- Index: `benchs/results/sindi_wholenet_direct8dmq_weighted_20260520_090022/sindi.index`
- Build params: `use_quantization=true`, `use_reorder=true`, `rerank_type="dmq"`,
  `dmq_bits=8`, `term_id_limit=300000`, `window_size=50000`, `doc_prune_ratio=0.2`
- Search params: `query_prune_ratio=0.2`, `term_prune_ratio=0`, `topk=10`,
  `use_term_lists_heap_insert=true`
- Profile query count: 10,000, cycling over the 1,000 dataset queries

The profile mode is analyzer-only. It repeats the same internal stages as `KnnSearch`, but adds
per-stage timers and counters. Use regular search metrics for QPS and latency; use this profile for
stage ratios and bottleneck attribution.

Profile outputs:

- `profile_n100_metrics.json`
- `profile_n175_metrics.json`
- `profile_n200_metrics.json`

under `benchs/results/sindi_wholenet_direct8dmq_weighted_20260520_090022/`.

## Old SINDI vs DMQ + Compact ID

All rows below use Wholenet Sparse 10M, `topk=10`, `query_prune_ratio=0.2`,
`term_prune_ratio=0`, `use_term_lists_heap_insert=true`, and 10,000 searched queries. The old SINDI
baseline uses fp32 reorder. The DMQ8 rows replace the fp32 forward rerank store with compressed DMQ
rerank storage. The compact-id row additionally bit-packs stored term ids by `term_id_limit`.

| Variant | n_candidate | Recall | QPS | Latency avg | Latency p95 | Loaded index | Rerank backend | Search peak RSS | Index file | Build time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old SINDI fp32 reorder | 100 | 0.9482 | 796.53 | 1.2555 ms | 2.5420 ms | 4.87 GB | 3.29 GB | 10.29 GB | 5.20 GB | 55.25 s |
| DMQ8, before compact id | 100 | 0.9425 | 771.97 | 1.2954 ms | 2.5908 ms | 3.93 GB | 2.35 GB | 9.20 GB | 4.34 GB | 79.79 s |
| DMQ8 + compact id | 100 | 0.9425 | 603.10 | 1.6581 ms | 2.9246 ms | 3.14 GB | 1.56 GB | 7.97 GB | 3.66 GB | 119.16 s |
| Direct8 weighted DMQ + compact id, optimized | 200 | 0.9499 | 659.28 | 1.5168 ms | 2.7826 ms | 3.18 GB | 1.60 GB | 8.00 GB | 3.70 GB | 202.05 s |

Same-parameter comparison at `n_candidate=100`:

- DMQ8 before compact id reduces loaded index memory by about 19.3% vs old SINDI, and keeps QPS
   close to fp32 reorder: 771.97 vs 796.53, about 3.1% lower. Recall drops from 0.9482 to 0.9425.
- DMQ8 + compact id reduces loaded index memory by about 35.5% and rerank backend memory by about
   52.6% vs old SINDI. Search peak RSS drops by about 22.5%, and index file size drops by about
   29.6%.
- The compact-id step is where most of the QPS regression appears: QPS drops from 771.97 to 603.10
   after term ids become bit-packed. This is because rerank must unpack packed term ids in the hot
   candidate scoring loop.
- The current direct8 weighted DMQ keeps the compact-id memory profile but uses a wider candidate
   pool for recall. With the optimized scorer at `n_candidate=200`, recall reaches 0.9499 and QPS is
   659.28, still about 17.2% below old SINDI fp32 QPS but with about 34.7% lower loaded index memory.

## Search Pipeline

1. Query preparation
   - Optional term-id remap.
   - Build `SparseTermComputer`, sort query terms, apply `query_prune_ratio`.
   - Allocate the per-query window distance buffer.

2. Inverted-list accumulation
   - `SparseTermDataCell::Query()` walks active query terms in every window.
   - For each posting entry, it adds `-q_t * value_t` into the window-local `dists` buffer.
   - With quantization enabled, posting values are stored as one byte and converted during scoring.

3. Coarse candidate heap
   - `InsertHeapByTermLists()` revisits active term posting ids and reads accumulated `dists[id]`.
   - It keeps the global coarse heap at `n_candidate` and resets touched distance slots.
   - This pass is part of inverted search, but is reported separately because it is large.

4. Forward DMQ query preparation
   - `SINDIDmqRerankBackend::PrepareQuery()` sorts the query for rerank.
   - For bounded term ids, it builds a direct term-id to query-value lookup table.

5. Forward DMQ scoring
   - For each coarse candidate, `CalDistanceByInnerId()` loads the candidate's encoded vector:
     bit-packed term ids, 8-bit value codes, and `{mean, alpha}`.
   - The optimized path decodes term ids in 32-term blocks, uses direct byte value-code loads, and
     resolves codebooks through a direct term-id lookup.
   - Score formula remains:

```text
approx_ip = mean * sum(q_t over intersection) + alpha * sum(q_t * C_t[code_t])
distance  = 1 - approx_ip
```

6. Reorder heap and materialization
   - The reranked distance is inserted into a final top-k heap.
   - The heap is popped into output order. This is small compared with inverted search and DMQ score.

## Persistent Memory

Loaded index memory is dominated by the inverted postings and the forward DMQ store.

| Component | Bytes | Rendered size | Notes |
| --- | ---: | ---: | --- |
| Posting data cells | 1,698,099,271 | 1.58 GB | Windowed inverted lists: `uint16` local ids plus quantized values |
| DMQ rerank backend | 1,720,721,800 | 1.60 GB | Forward direct8 DMQ storage |
| Label table | 200,000,360 | 190.74 MB | Label to inner-id mapping |
| Window pointer array | 3,200 | 3.12 KB | 200 windows |
| Quantization params | 12 | 12 B | Posting-list value min/max/diff |
| Accounted total | 3,418,824,787 | 3.18 GB | Index memory without common label/extra accounting |
| Total with common | 3,618,825,147 | 3.37 GB | Includes label table and extra infos |

Forward DMQ backend detail:

| Component | Bytes | Rendered size | Notes |
| --- | ---: | ---: | --- |
| Packed term ids | 952,812,746 | 908.67 MB | 19-bit term ids for 401,184,314 stored nonzeros |
| Value codes | 401,184,314 | 382.60 MB | One direct8 code per stored nonzero |
| Encoded vector metadata | 240,000,000 | 228.88 MB | Offset, length, mean, alpha per vector |
| Codebooks | 124,990,600 | 119.20 MB | 61,150 terms, 256 float representatives plus thresholds |
| Codebook lookup | 1,000,004 | 976.57 KB | Direct term-id to codebook-index table |
| Codebook map estimate | 489,200 | 477.73 KB | Hash map fallback estimate |
| Codebook term ids | 244,600 | 238.87 KB | Term id list for codebook serialization |

The codebook lookup table costs about 1 MB and is not a meaningful memory regression. The largest
remaining forward-memory target is packed term ids, not value codes or codebooks.

## Per-query Transient Memory

| Item | n=100 | n=175 | n=200 | Notes |
| --- | ---: | ---: | ---: | --- |
| Window `dists` buffer | 200,000 B | 200,000 B | 200,000 B | `window_size * sizeof(float)` |
| DMQ query lookup mean | 604,017 B | 604,017 B | 604,017 B | Direct query value table, mean over queries |
| DMQ query lookup max | 999,772 B | 999,772 B | 999,772 B | Guarded by the 1,000,000 lookup limit |
| Coarse heap pair estimate | 1,600 B | 2,800 B | 3,200 B | `n_candidate * sizeof(pair<float,int64_t>)` |
| Reorder heap pair estimate | 160 B | 160 B | 160 B | `topk=10` |

Transient memory is small relative to persistent index memory. The direct query lookup is the only
per-query allocation above a few hundred KB, and it is capped.

## Stage Time

Average per-query profile time:

| n_candidate | Query prep | Inverted accumulate | Coarse heap | Inverted total | DMQ prepare | DMQ score | Reorder heap | Profiled total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.0048 ms | 0.7464 ms | 0.4056 ms | 1.1520 ms | 0.0259 ms | 0.1221 ms | 0.0291 ms | 1.3343 ms |
| 175 | 0.0052 ms | 0.7394 ms | 0.4421 ms | 1.1815 ms | 0.0272 ms | 0.2033 ms | 0.0456 ms | 1.4632 ms |
| 200 | 0.0054 ms | 0.7357 ms | 0.4601 ms | 1.1958 ms | 0.0276 ms | 0.2266 ms | 0.0509 ms | 1.5066 ms |

Stage ratio:

| n_candidate | Inverted accumulate | Coarse heap | Inverted total | DMQ prepare | DMQ score | Reorder heap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 55.9% | 30.4% | 86.3% | 1.9% | 9.2% | 2.2% |
| 175 | 50.5% | 30.2% | 80.8% | 1.9% | 13.9% | 3.1% |
| 200 | 48.8% | 30.5% | 79.4% | 1.8% | 15.0% | 3.4% |

Search work counters are stable across candidate widths:

| Counter | Value |
| --- | ---: |
| Active query terms after prune | 4.293 mean |
| Windows searched | 200 max |
| Term-window posting hits | 847.149 mean/query |
| Posting entries scanned | 278,785.4 mean/query |
| Coarse candidates scored at n=100 | 99.781 mean/query |
| Coarse candidates scored at n=175 | 174.515 mean/query |
| Coarse candidates scored at n=200 | 199.415 mean/query |

## Interpretation

The main bottleneck is now inverted search, not forward DMQ. At `n_candidate=200`, about 79% of
profiled time is still spent before DMQ rerank: roughly 49% in posting accumulation and 31% in the
coarse heap pass. Forward DMQ scoring is about 15%, and final reorder heap work is only about 3.4%.

Increasing `n_candidate` mostly scales the DMQ and reorder stages. From `n=100` to `n=200`, DMQ score
increases from 0.1221 ms to 0.2266 ms, while posting accumulation stays around 0.74 ms. The wider
candidate pool is therefore a recall/QPS tradeoff in the forward scorer, but the absolute floor is
set by inverted-list scanning.

The current direct8 hot-path work was still worthwhile: it reduced the candidate-dependent rerank
cost enough for `n=175/200` to recover most QPS. The remaining large time bucket is the coarse path.

## Optimization Priorities

1. Reduce duplicated inverted-list work. `Query()` scans posting entries to accumulate `dists`, then
   `InsertHeapByTermLists()` walks active posting ids again to select candidates. A per-window touched
   id list or mark array could turn the second pass from repeated posting-list scans into iteration
   over unique touched docs.
2. Vectorize or specialize posting accumulation. The hot loop is `uint16 id + uint8 value -> float
   dists[id] += q * value`. AVX512 scatter or block-local accumulation may help, but the random
   scatter pattern is the hard part.
3. Tune pruning before widening candidates. `query_prune_ratio` and `term_prune_ratio` directly
   control the 278k posting entries scanned per query. Any recall-preserving reduction here buys more
   than another DMQ micro-optimization.
4. Keep DMQ lookup optimizations. The direct codebook lookup costs about 1 MB and removes hash-map
   work from the scoring loop; it should stay.
5. Treat final reorder heap as low priority. It is below 4% even at `n=200`.
