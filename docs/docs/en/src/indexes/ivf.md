# IVF

![IVF: Voronoi partition over k-means centroids; only the scan_buckets_count buckets closest to the query are scanned, with an optional precise rerank](../figures/indexes/ivf-overview.svg)

IVF (Inverted File) is VSAG's **partition-based** index. It clusters the corpus into
buckets at build time, and at query time only scans the buckets whose centroids are
closest to the query. This turns an O(N) linear scan into O(N · `scan_buckets_count`
/ `buckets_count`) with tunable recall/latency.

IVF trades a little recall (compared to graph indexes) for lower memory overhead,
higher throughput on batch workloads, and simpler sharding — which makes it a good
default when the corpus is large (hundreds of millions or more), when memory is
tight, or when queries are naturally parallelizable.

- Source: `src/algorithm/ivf.{h,cpp}`, `src/algorithm/ivf_parameter.{h,cpp}`
- Example: [`examples/cpp/106_index_ivf.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/106_index_ivf.cpp)

## How it works

1. **Clustering.** A sample of the dataset is clustered with k-means (or sampled
   randomly, `ivf_train_type: "random"`, or clustered into equal-size clusters,
   `ivf_train_type: "balanced_kmeans"`) to produce `buckets_count` centroids.
2. **Assignment.** Every vector is written to the inverted list of its nearest
   centroid (with `balanced_kmeans`, the first `Build` fills every bucket equally;
   see [Balanced k-means training](#balanced-k-means-training)), stored in the
   configured coarse quantization (`base_quantization_type`).
   Optionally, a second high-precision copy is kept (`use_reorder: true`) for
   post-filter reordering.
3. **Search.** For each query, the `scan_buckets_count` nearest centroids are
   computed first, then the vectors in those buckets are scored. When reordering is
   enabled, `factor` controls how many extra candidates are fetched from the coarse
   stage before being re-scored with the precise quantizer.

A second partition strategy, **GNO-IMI** (`partition_strategy_type: "gno_imi"`),
splits the space into two orthogonal sets of centroids
(`first_order_buckets_count` × `second_order_buckets_count`) for even finer
partitioning on very large corpora.

## Quick start

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "buckets_count": 256,
        "base_quantization_type": "sq8",
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans"
    }
})";
auto index = vsag::Factory::CreateIndex("ivf", params).value();

// Build.
auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Build(base);

// Search.
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10,
    R"({"ivf": {"scan_buckets_count": 16}})").value();
```

## Input data type

The public `Build`, `Add`, and search paths currently accept FP32 vectors supplied with `Dataset::Float32Vectors`; set `dtype` to `"float32"`. `base_quantization_type` selects internal encoding and storage and does not enable FP16/BF16 input. `dtype: "int8"` is not supported when creating an IVF index.

## Build parameters

Build-time parameters live under `index_param`. See
[Index Parameters](../resources/index_parameters.md) for the exhaustive list.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `partition_strategy_type` | string | `"ivf"` | `ivf` (single-level) or `gno_imi` (two-level orthogonal) |
| `buckets_count` | int | `10` | Number of inverted lists (effective for `ivf`) |
| `first_order_buckets_count` | int | `10` | First-level count (effective for `gno_imi`) |
| `second_order_buckets_count` | int | `10` | Second-level count (effective for `gno_imi`) |
| `ivf_train_type` | string | `"kmeans"` | Centroid training: `kmeans`, `random` or `balanced_kmeans` (equal-size buckets, effective for `ivf`; see [Balanced k-means training](#balanced-k-means-training)). Other names are rejected with `INVALID_ARGUMENT`. |
| `kmeans_iter_count` | int | `25` (`ivf`), `30` (`gno_imi`) | Training iterations (effective with `ivf_train_type: "kmeans"` or `"balanced_kmeans"`). Higher → better bucket quality, longer build. |
| `route_max_degree` | int | `64` | Routing HGraph maximum degree (effective for `ivf`) |
| `route_ef_construction` | int | `300` | Routing HGraph construction search breadth (effective for `ivf`) |
| `enable_gpu_build` | bool | `false` | Use the CUDA backend while building (effective for `ivf`); see [GPU-accelerated build](#gpu-accelerated-build) |
| `gpu_device_id` | int | `0` | CUDA device ordinal (effective for `ivf`); an ordinal the machine does not have keeps the build on the CPU |
| `gpu_memory_budget` | int | `0` | Ceiling in bytes on the device working set (effective for `ivf`); `0` derives it from what the device has free |
| `gpu_min_work_threshold` | int | `0` | Smallest `count * buckets_count * dim` worth offloading (effective for `ivf`); `0` uses the calibrated default |
| `base_quantization_type` | string | `"fp32"` | `fp32`, `fp16`, `bf16`, `sq8`, `sq4`, `sq8_uniform`, `sq4_uniform`, `pq`, `pqfs`, `rabitq` — see the [Quantization chapter](../quantization/) for per-quantizer details |
| `base_pq_dim` | int | `1` | PQ subspaces (required with `pq` / `pqfs`) |
| `rabitq_pca_dim` | int | `0` | Optional PCA preprocessing dimension for `base_quantization_type: "rabitq"` |
| `rabitq_bits_per_dim_query` | int | `32` | Query bits for `rabitq`; allowed values are `4` or `32` |
| `rabitq_bits_per_dim_base` | int | `1` | Stored-code bits for `rabitq`; allowed range is `[1, 8]` |
| `rabitq_bits_per_dim_precise` | int | unset | Enables RaBitQ split storage and sets the supplement (`y`) bits; `rabitq_bits_per_dim_base` is the filter (`x`) bits and `x + y <= 8` |
| `rabitq_version` | string | `"standard"` | `rabitq` layout: `"standard"` or `"split_1bit_7bit"` |
| `rabitq_error_rate` | float | `1.9` | Positive error-budget parameter for `rabitq` encoding |
| `rabitq_use_fht` | bool | `false` | Enable FHT rotation before `rabitq` binarization |
| `fast_encode_rabitq` | bool | `true` | Use CAQ fast construction for multi-bit `rabitq`; set to `false` for exact encoding |
| `fast_encode_rabitq_rounds` | int | `6` | CAQ adjustment rounds; allowed range is `[1, 32]` |
| `use_residual` | bool | `false` | Encode vectors relative to their IVF bucket centroid (`x-c`). |
| `use_reorder` | bool | `false` | Keep a high-precision copy and re-rank after the coarse scan |
| `precise_quantization_type` | string | `"fp32"` | Quantizer used for reordering (with `use_reorder: true`) |
| `precise_codes_layout` | string | `"flat"` | Storage layout for precise codes: `"flat"` keeps the legacy one-code-per-vector layout; `"bucket"` stores the precise code in the same bucket and offset as its basic posting |
| `base_io_type` | string | `"memory_io"` | Storage backend for coarse codes; supports `uring_io` when built with liburing |
| `base_supplement_io_type` | string | unset | Optional IO backend for RaBitQ split supplement codes |
| `base_supplement_file_path` | string | derived | Optional file path for disk-backed RaBitQ supplement codes |
| `precise_io_type` | string | `"block_memory_io"` | Storage backend for precise codes (`memory_io`, `block_memory_io`, `mmap_io`, `buffer_io`, `async_io`, `uring_io`, `reader_io`) |
| `precise_file_path` | string | `""` | File path when the precise IO type is disk-backed |

`precise_codes_layout: "bucket"` requires `use_reorder: true`. It supports
`memory_io`, `block_memory_io`, `buffer_io`, `async_io`, and `uring_io`
(when io_uring is available).
`mmap_io` and `pqfs` precise quantization are not supported. The bucket layout currently requires
`buckets_per_data: 1`; configurations that assign one vector to multiple buckets are rejected.

For both `flat` and `bucket` layouts, serialized precise codes can be loaded read-only from an
external `Reader`. Pass `precise_io_type: "reader_io"` and `precise_reader` to `Index::Load`.
The reader must expose exactly the `high_precision_codes` block payload for `flat`, or the
`ivf_precise_bucket` block payload for `bucket`. `reader_io` uses the normal read cache when
`precise_enable_read_cache` is enabled.

```cpp
vsag::LoadParameters load_parameters;
load_parameters.Set("precise_io_type", "reader_io")
    .Set("precise_enable_read_cache", true)
    .Set("precise_cache_total_size", 256ULL * 1024 * 1024)
    .SetReader("precise_reader", precise_codes_reader);
auto loaded = vsag::Index::Load(stream, load_parameters).value();
```

`reader_io` is a load-and-query placement policy. Build the index with a writable precise IO,
serialize it, and then use the external reader when loading it for search.

A rule of thumb for `buckets_count` is `sqrt(N)` to `4 * sqrt(N)` where `N` is the
corpus size.

### GPU-accelerated build

Building an IVF index is dominated by two things: clustering the training sample
into centroids, and then assigning every base vector to its bucket. Clustering
grows with `buckets_count`, because the sample does too (`train_sample_count`
defaults to `max(65536, 64 * buckets_count)`), and with
`ivf_train_type: "kmeans"` the seeding pass alone is
`O(buckets_count * train_sample_count * dim)` and runs single-threaded.

When VSAG is built with `ENABLE_CUDA=ON`, both move to a CUDA device. Set
`enable_gpu_build` to turn it on:

```json
{
    "buckets_count": 4096,
    "base_quantization_type": "fp32",
    "partition_strategy_type": "ivf",
    "ivf_train_type": "kmeans",
    "enable_gpu_build": true,
    "gpu_device_id": 0
}
```

Four passes run on the device: k-means++ seeding, nearest-centroid assignment
and centroid accumulation inside clustering, and the bucket assignment of the
base vectors afterwards. `ivf_train_type: "random"` does not cluster, so only
the last one applies to it. `ivf_train_type: "balanced_kmeans"` clusters and
fills the buckets of a `Build` on the host, so of these passes only the
nearest-bucket assignment applies to it: for vectors added later and, with codes
relative to a bucket, for the training vectors.

The four settings reach the `ivf` partition strategy only. With
`partition_strategy_type: "gno_imi"` they are accepted and ignored, as the other
strategy-specific parameters in the table are, and that build stays on the host.

**Scope.** Only the build changes. Search is untouched: queries still route through the
graph, which is the right shape for one query at a time. The four settings are
recorded in the index footer as every other build parameter is, so two otherwise
identical builds are not byte-identical, but the compatibility check on load
ignores them: an index built with the backend on loads into a reader configured
any other way, including one on a machine with no GPU, and then behaves as that
reader is configured. The centroids can differ slightly from a host build's: the
random choices come from the caller's generator either way, but the device sums
each cluster with atomics in no fixed order.

**Fallback.** A pass that declines is taken by the host, silently, and returns
what a build with the backend off would have returned. This happens when VSAG
was built without CUDA, no device is present, `gpu_device_id` names an ordinal
the machine does not have, the problem is below `gpu_min_work_threshold`, the
budget cannot hold what a pass needs, or any CUDA call fails. A build never
fails because of the backend.

Two cases stay on the host by design. Inner-product metrics order buckets by the
dot product alone, which is not what the device pass minimises, so they keep the
graph routing. A build that asked for a reproducible reduction keeps centroid
accumulation on the host as well, since the device sums with atomics whose
order is not fixed.

**Memory.** Vectors are streamed in chunks, so the corpus does not have to fit
in device memory; only the centroids and one chunk do. The working set is
bounded by `gpu_memory_budget`, by default derived from what the device reports
free, so the same build behaves on a small card and makes use of a large one.
Seeding differs only in shape: it sweeps the whole training sample once per
centroid, so the sample has to be resident rather than chunked, and a sample the
budget cannot hold sends seeding back to the host while the other passes stay on
it. When no budget is named the chunked passes take a default ceiling, since a
chunk larger than what saturates the device only costs more; seeding takes what
the device has free, because a ceiling there would only push large samples to
the host.

Measured on SIFT1M (1M base vectors, 128 dimensions), against a 48-thread host
build as the baseline:

| `buckets_count` | CPU build | GPU build | Speedup | Recall@10 change |
|-----------------|-----------|-----------|---------|------------------|
| 1000 | 59.45 s | 6.30 s | 9.43x | +0.058 pt |
| 4096 | 167.66 s | 8.02 s | 20.90x | +0.075 pt |
| 16384 | 729.04 s | 27.21 s | 26.79x | +0.079 pt |

The speedup grows with `buckets_count` because clustering takes a larger share
of the build. How large it is depends on the balance between the host's CPU and
its GPU, which is also what `gpu_min_work_threshold` exists to retune.

Recall moves because the device compares against every centroid where the host
does not, in two places.

Assigning base vectors to buckets, on the default `use_route_graph: true`
layout, searches the routing graph, which is approximate. On 20000 random
vectors of 64 dimensions the two paths agreed exactly at `buckets_count` 16; at
256 and 1000 they placed roughly 10% and 12% of the vectors in different
buckets, the device naming the nearer centroid in every one of those cases, and
the mean distance to the assigned centroid improved by under 1% and by about 2%.
The figures shift a little from run to run, because training is not seeded and
each run clusters differently; they held across unit-norm and raw random data.
The scan layout assigns exactly on either path and does not move.

Clustering does the same above 10000 centroids, where the host switches to a
graph search over the centroids to keep training affordable.

So a build with the backend on assigns somewhat better, not only faster. The
evidence for that is the distance comparison above, not the recall column: the
table is one run per configuration, and recall between two separately trained
indexes moves by more than the differences it shows. Four builds with identical
settings on 50000 random vectors of 128 dimensions spanned a full point of
recall, against the hundredths of a point in the table. Read that column as
showing no regression, not as measuring the gain. Below those sizes the two agree
except where two centroids are equidistant from a point to within floating-point
rounding.

## Balanced k-means training

Plain k-means lets the data decide how large every bucket is. On skewed or
duplicated data a few buckets end up holding a large share of the corpus, every
query routed to them scans all of it, and the work per query becomes uneven.
`ivf_train_type: "balanced_kmeans"` trains the centroids with balanced k-means
and fills the buckets of the first `Build` the same way: every bucket receives
`floor(n / buckets_count)` or `ceil(n / buckets_count)` vectors, so no bucket is
empty and no two differ by more than one vector.

```json
{
    "buckets_count": 4096,
    "base_quantization_type": "fp32",
    "partition_strategy_type": "ivf",
    "ivf_train_type": "balanced_kmeans"
}
```

**What is balanced.** The clusters of the training sample, and the buckets that a
`Build` into an empty index writes. The build assigns the base vectors to the
trained centroids under the same exact sizes, scored with the routing key of the
metric (`1 - <x, c>` for `ip`, one minus the cosine for `cosine`), so a vector goes
to the bucket a query near it is routed to whenever the sizes allow. Codes kept
relative to a bucket centroid (`use_residual`, RaBitQ split storage) are trained on
residuals from the same balanced assignment of the training vectors and, where it
differs, from their nearest buckets, so they cover both the vectors a `Build` places
away from their nearest bucket and the vectors `Add` routes to it. A `Build` on a
model trained or exported beforehand can bring fewer vectors than buckets; every
vector then gets a bucket of its own. Vectors added later with `Add` go to their
nearest bucket, as with the other trainers, so the sizes drift as the index grows.
With `buckets_per_data` above 1 the first copy of every vector is balanced and the
other copies go to the nearest other buckets.

**What stays the same.** Training uses the same sample, objective and iterations
as `kmeans` (squared L2, centroids are means, `kmeans_iter_count` iterations), and
the same post-processing: for `cosine` the centroids are normalized afterwards.
`kmeans` stays the default and its results do not change. The trainer is not part
of the compatibility check, so an index built with one trainer loads with a reader
configured with another. `partition_strategy_type: "gno_imi"` always trains with
plain k-means and rejects `balanced_kmeans`.

### How it works

Every iteration assigns the vectors under the size constraint and then moves each
centroid to the mean of its cluster, as k-means does. The assignment is a
transportation problem, solved by an auction (Bertsekas) with one price per
cluster: a vector bids for the cluster with the lowest `distance + price`, a full
cluster gives the seat to the higher bid and its lowest bidder bids again, and
prices only rise. With epsilon-scaling every assignment is optimal for its
centroids and its cluster sizes within `n * epsilon` (for the distances as computed
in float), epsilon being 1e-4 of the typical distance gap between a vector's closest
clusters; as the mean is optimal for a fixed cluster, the loss does not grow between
iterations. When `n` is not divisible by `buckets_count`, the buckets with one vector
more start as the ones most vectors prefer, and a vector that saves cost by moving to
a bucket without that extra seat takes the seat along. This choice of buckets is a
good one, not a jointly optimal one.
Every iteration starts from the previous assignment and prices, raised by how far
each centroid moved, so most vectors keep their seats and only the others bid.

The working set never holds an `n x buckets_count` matrix. Each vector keeps its
16 cheapest clusters and a lower bound for all the others, and distances are
computed in GEMM tiles of at most 64 vectors by 1024 centroids per task. When the
centroids lie far from the origin compared with their spread, the tiles are
computed around the centroids' mean, so that small distances keep their precision.
A vector whose remembered clusters all become dearer than its bound is scored again
against every centroid, so the bound is never trusted beyond what it proves.

| | `kmeans` | `balanced_kmeans` |
|---|---|---|
| Distances per iteration | `O(n · k · d)` | `O(n · k · d)`, plus `O(r · k · d)` for `r` re-scored vectors |
| Assignment | nearest centroid | auction, `O(b · 16)` for `b` bids |
| Working memory | a `min(n, 65536) x k` distance block (`k` < 10000) | `O(n · 16 + k · d)`, plus one tile per thread |
| Cluster sizes | unconstrained | `floor(n / k)` or `ceil(n / k)` |

On real data `b` and `r` stay at a few per vector per iteration. Dense blobs that
each need many clusters (high-dimensional Gaussian mixtures, for example) make the
auction fight over nearly equivalent clusters, so the work of every assignment is
counted and bounded. Bids and re-scored vectors are counted rather than timed, so
the outcome does not depend on threads. When the first assignment exceeds its
bound, training switches to a two-level decomposition: about `sqrt(k)`
super-clusters whose sizes are whole numbers of leaves, each split into its leaves
in parallel, with the same exact sizes. When a later assignment exceeds it, the
result is whichever of the iterations so far and the two-level run has the lower
loss.

### Trade-offs

Measured with [`cluster_benchmark`](../resources/cluster_benchmark.md) on 32
threads of one NUMA node, 25 iterations, loss as the mean squared distance to the
assigned centroid and peak memory as what the trainer allocates beyond its input.
Every row is the run with the median time of three:

| Data (`n x d`, `k`) | Trainer | Sizes | CV | Loss | Time | Peak memory |
|---|---|---|---|---|---|---|
| Uniform (100,000 x 64, 256) | `kmeans` | 334–452 | 0.052 | 4.5991 | 1.06 s | 64.4 MiB |
| | `balanced_kmeans` | 390–391 | 0.001 | 4.6004 | 0.82 s | 19.0 MiB |
| Not divisible (99,968 x 64, 256) | `kmeans` | 342–448 | 0.050 | 4.5963 | 1.03 s | 64.4 MiB |
| | `balanced_kmeans` | 390–391 | 0.001 | 4.5969 | 0.82 s | 19.0 MiB |
| Skewed, overlapping (100,000 x 64, 256) | `kmeans` | 5–1327 | 0.393 | 3.5829 | 1.05 s | 64.4 MiB |
| | `balanced_kmeans` | 390–391 | 0.001 | 3.7469 | 4.01 s | 19.0 MiB |
| Skewed, separated + duplicates (100,000 x 64, 256) | `kmeans` | 2–809 | 0.526 | 3.5802 | 1.09 s | 64.4 MiB |
| | `balanced_kmeans` (two-level) | 390–391 | 0.001 | 19.2411 | 4.24 s | 41.8 MiB |
| SIFT1M, first 262,144 (128, 4096) | `kmeans` | 1–355 | 0.525 | 43694 | 54.2 s | 1027 MiB |
| | `balanced_kmeans` | 64–64 | 0 | 43818 | 13.6 s | 60.4 MiB |
| GloVe-100, first 262,144 (100, 4096) | `kmeans` | 1–734 | 1.297 | 21.371 | 39.7 s | 1027 MiB |
| | `balanced_kmeans` | 64–64 | 0 | 21.142 | 21.9 s | 59.9 MiB |
| Skewed, overlapping (262,144 x 128, 4096) | `kmeans` | 1–1698 | 1.305 | 6.9020 | 53.6 s | 1027 MiB |
| | `balanced_kmeans` (two-level) | 64–64 | 0 | 6.9823 | 37.2 s | 95.8 MiB |

The skewed sets are Zipf-weighted Gaussian mixtures; "separated" places 64
components far apart, so many small ones have to share clusters, and repeats eight
vectors in 5% of the rows.

- **Balance is exact** on every set. Plain k-means leaves its largest cluster at 1.1
  to 26.5 times the mean size, and at 4096 clusters some clusters hold one vector.
- **Loss** is within 0.3% of plain k-means on uniform, non-divisible and SIFT data,
  and lower on GloVe, where plain k-means keeps many nearly empty clusters. Where
  the data is clustered unevenly, equal sizes cost loss by construction: 4.6% and
  1.2% on the two overlapping mixtures, and on the separated one small far-apart
  components have to share clusters.
- **Time and memory:** at IVF scale the trainer is faster than plain k-means
  (whose seeding is single-threaded and whose distance block grows with
  `65536 x k`) and needs 11 to 17 times less memory. Its own seeding does not
  decide that: with k-means++ (`--init kmeans++`) instead of a random sample it
  takes 12.8 s on SIFT, at a 0.05% higher loss.

The same holds for whole indexes. Built from the full SIFT1M (`l2`) and GloVe-100
(`cosine`) files with 4096 buckets and `fp32` codes, then searched by 32 threads on
one NUMA node at `scan_buckets_count` 16 / 64 / 128 (`cluster_benchmark --ivf`,
recall@10 against the files' ground truth; the run with the median build time of
three, with the median QPS of the three at each setting):

| Data | Trainer | Build | Bucket sizes | Recall@10 | Distances per query | QPS |
|---|---|---|---|---|---|---|
| SIFT1M | `kmeans` | 66.0 s | 1–1897 | 0.834 / 0.971 / 0.992 | 5212 / 19415 / 37259 | 126k / 37.8k / 18.6k |
| | `balanced_kmeans` | 28.1 s | 244–245 | 0.826 / 0.971 / 0.992 | 4148 / 16173 / 32059 | 132k / 43.0k / 21.7k |
| GloVe-100 | `kmeans` | 55.7 s | 0–2792 | 0.776 / 0.902 / 0.942 | 12359 / 44815 / 84130 | 78.3k / 23.6k / 12.8k |
| | `balanced_kmeans` | 38.7 s | 288–289 | 0.697 / 0.852 / 0.907 | 4980 / 19431 / 38489 | 148k / 45.0k / 23.2k |

Queries follow the data, so large buckets are probed more often than small ones,
and with plain k-means the probed buckets are larger than the mean (a bucket probed
in proportion to its size is on average `1 + CV²` times the mean: 1.3 on SIFT, 2.7
on GloVe). At the same `scan_buckets_count` plain k-means therefore evaluates 16% to
26% more distances on SIFT and 2.2 to 2.5 times as many on GloVe, which buys it
recall on GloVe. At about the same throughput the balanced index has the same or
higher recall (GloVe: 0.907 at 23.2k QPS against 0.902 at 23.6k; SIFT: 0.992 at
21.7k against 0.992 at 18.6k), and no query scans more than
`scan_buckets_count x ceil(n / buckets_count)` vectors until `Add` grows the
buckets.

Use `balanced_kmeans` when the cost of the largest bucket matters: skewed or
duplicated data, buckets that live on disk or are moved between shards, or
latency that has to stay predictable per query. On uniformly spread data it gives
about the same index as `kmeans`.

## RaBitQ split storage

Set both quantizers to `rabitq` and divide the stored precision into filter
(`x`) and supplement (`y`) bits:

```json
{
    "base_quantization_type": "rabitq",
    "precise_quantization_type": "rabitq",
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 3,
    "rabitq_bits_per_dim_precise": 5,
    "use_reorder": true
}
```

IVF stores the `x`-bit filter planes in bucket-local 32-vector packed blocks and
stores the `y`-bit supplement separately. The default `candidate_reorder`
strategy scans only the packed x bits, retains `factor * topk` candidates, and
carries the byte-LUT-quantized x-bit inner product together with each
candidate's source bucket, offset, and version. Normal reranking combines this
saved inner product with the y-bit supplement contribution. It does not reread
or unpack x-bit planes and does not reconstruct the inner product from a
distance hint.

For residual L2 candidate search, all routed buckets share one LUT built from
the original transformed query. Per-lane factors recover the raw filter-code
contribution and combine it with the supplement under the original query, so
normal reranking does not build a `q-c` computer. A correctness fallback may
unpack x bits only when the saved value or factors are invalid, source
provenance is stale, or a FastScan lane could not produce a valid result.

The split configuration requires `x >= 1`, `y >= 1`, `x + y <= 8`,
`rabitq_bits_per_dim_query: 32`, `use_reorder: true`, and
`buckets_per_data: 1`. Bucket-local graphs (`graph_build_threshold > 0`) are
not supported with split storage. Supported homogeneous IO backends are
`memory_io`, `block_memory_io`, `buffer_io`, `async_io`, `mmap_io`,
and `reader_io`; the supported hybrid layout is filter
`block_memory_io` plus supplement `async_io`.

## Search parameters

Search-time parameters live under the `ivf` sub-object:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scan_buckets_count` | int | — (required) | Number of buckets probed per query. Must be ≤ `buckets_count` (except when `disable_bucket_scan` is true, where larger values are allowed and unavailable slots are padded with `-1`). |
| `disable_bucket_scan` | bool | `false` | Return bucket IDs and distances. Supports batch queries. |
| `factor` | float | `2.0` | With reordering enabled, pulls `factor * topk` coarse candidates before the precise rescore. |
| `enable_reorder` | bool | `true` | Set to `false` to skip the final reorder stage for this request even when the index was built with reorder enabled. |
| `parallelism` | int | `1` | Threads used to scan buckets in parallel for a single query. |
| `timeout_ms` | double | `+∞` | Hard cap in milliseconds; partial results are returned once exceeded. |

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"ivf": {"scan_buckets_count": 32, "factor": 2.0, "parallelism": 4}})").value();
```

```cpp
auto fast_result = index->KnnSearch(
    query, topk,
    R"({"ivf": {"scan_buckets_count": 32, "factor": 2.0, "enable_reorder": false}})").value();
```

## When to use IVF

- Large corpora (hundreds of millions of vectors and above), especially when the
  working set does not fit comfortably in RAM.
- Batch or high-throughput workloads where per-query latency is less critical than
  queries-per-second.
- Memory-tight deployments that benefit from aggressive quantization (`sq8`,
  `sq4_uniform`, `pq`, `pqfs`) combined with `use_reorder` to recover recall.
- Shard-friendly setups: buckets map naturally onto shards or disk blocks.

For latency-sensitive, high-recall workloads on dense embeddings, compare against
[HGraph](hgraph.md) first.

## See also

- [Creating an Index](../guide/create_index.md)
- [Index Parameters](../resources/index_parameters.md)
- [Serialization](../advanced/serialization.md)
