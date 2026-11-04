# Clustering Benchmark Tool (`cluster_benchmark`)

`cluster_benchmark` compares the two IVF trainers, `kmeans` and `balanced_kmeans`, on the same
vectors. It reports how even the clusters are, the clustering loss, the training time and the
peak memory the trainer allocates, and with `--ivf` it builds IVF indexes with both trainers and
searches them.

The clustering comparison is seeded, and `balanced_kmeans` repeats its result exactly. `kmeans`
runs as an index build runs it, merging its per-block sums in completion order, so its numbers
vary slightly between runs. The `--ivf` builds go through the public API, which takes no seed.

See [IVF: Balanced k-means training](../indexes/ivf.md#balanced-k-means-training) for what the
balanced trainer does and for results measured with this tool.

## Build

```bash
VSAG_ENABLE_TOOLS=ON make release
# Output: ./build-release/tools/cluster_benchmark/cluster_benchmark
```

## Clustering comparison

```bash
# Uniform, skewed and not-divisible synthetic data, both trainers
./build-release/tools/cluster_benchmark/cluster_benchmark --dataset suite \
    --count 100000 --dim 64 -k 256 --threads 32

# A skewed mixture whose components overlap
./build-release/tools/cluster_benchmark/cluster_benchmark --dataset skewed --spread 0.5 \
    --count 100000 --dim 64 -k 256 --threads 32

# The first 262144 vectors of an ann-benchmarks file, at IVF scale
./build-release/tools/cluster_benchmark/cluster_benchmark \
    --dataset sift-128-euclidean.hdf5 --count 262144 -k 4096 --threads 32
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | `suite` | `uniform`, `skewed`, `indivisible`, `suite` (all three), or a `.hdf5` (ann-benchmarks `train` split) or `.fvecs` file |
| `--count` | `100000` | Vectors (synthetic data) or the most to read from a file |
| `--dim` | `64` | Dimension of synthetic data |
| `-k`, `--clusters` | `256` | Clusters (IVF buckets) |
| `--iterations` | `25` | Lloyd iterations for both trainers |
| `--threads` | `16` | Threads of the shared thread pool |
| `--seed` | `2870` | Seed of the data and of both trainers |
| `--algorithms` | `kmeans,balanced_kmeans` | Trainers to run |
| `--candidates` | `16` | `balanced_kmeans`: clusters remembered per vector |
| `--init` | `random` | `balanced_kmeans`: seeding, `random` or `kmeans++` |
| `--epsilon` | `1e-4` | `balanced_kmeans`: auction tolerance relative to the cost gap |
| `--work-budget` | `1` | `balanced_kmeans`: scale of the per-assignment work bound; `inf` never switches to the two-level path, `0` always does |
| `--spread` | `4` | Skewed data: half-width of the box holding the mixture centers (the noise is 0.25) |
| `--duplicates` | `0.05` | Skewed data: fraction of rows that repeat one of eight hot vectors |
| `--json` | none | Also write the results to this file |

The synthetic sets are: `uniform`, uniform in `[0, 1)^dim`; `skewed`, a mixture of 64 Gaussian
components with Zipf weights (exponent 1.2) whose centers are uniform in `[-spread, spread]^dim`,
plus repeated hot vectors; and `indivisible`, uniform data with `k / 2` vectors more than a
multiple of `k`.

Every row of the output reports, per trainer:

| Column | Meaning |
|---|---|
| `min`, `max`, `mean`, `std`, `CV` | Cluster sizes and their coefficient of variation (`std / mean`) |
| `loss (MSE)` | Mean squared distance of every vector to its centroid |
| `time (s)` | Training time |
| `peak MiB` | Peak of the memory the trainer allocates through the VSAG allocator; the input vectors are excluded |
| `bids/n`, `rescored` | `balanced_kmeans`: auction bids per vector and vectors scored again |
| `path` | `balanced_kmeans`: `flat`, or `two-level` when the work bound switched to the two-level decomposition |

## IVF comparison

With `--ivf` the tool reads an ann-benchmarks file (`train`, `test` and `neighbors`), builds one
IVF index per trainer with the public API, and searches it at several `scan_buckets_count`
values:

```bash
./build-release/tools/cluster_benchmark/cluster_benchmark --ivf \
    --dataset sift-128-euclidean.hdf5 --metric l2 -k 4096 --scan 8,16,32,64 --threads 32
```

It reports the build time, the bucket sizes from `GetStats()`, recall@10 against the file's
ground truth, queries per second, and the distance evaluations of an average query, which grow
with the number of vectors in the probed buckets.

On a machine with several NUMA nodes, run it on one node
(`numactl --cpunodebind=0 --membind=0 ./build-release/...`). Otherwise the two indexes can
end up with their memory on different nodes, and their QPS then differ by more than their
distance evaluations explain.
