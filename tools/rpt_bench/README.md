# rpt_bench

Benchmarks `vsag::rpt_partition` against baseline partitioning strategies on an
ann-benchmarks style hdf5 dataset, reporting partition uniformity, locality preservation
and build cost.

See the API documentation for the partitioner itself:
[`docs/docs/en/src/advanced/rpt_partition.md`](../../docs/docs/en/src/advanced/rpt_partition.md) /
[`docs/docs/zh/src/advanced/rpt_partition.md`](../../docs/docs/zh/src/advanced/rpt_partition.md).

## Build

```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build --target rpt_bench --parallel
```

## Datasets

```bash
bash scripts/download_annbench_datasets.sh   # downloads into /tmp/data
```

That script provides `sift-128-euclidean.hdf5` (SIFT1M), `gist-960-euclidean.hdf5`
(GIST1M) and `deep-image-96-angular.hdf5`, which are the datasets the project targets.

## Usage

```bash
./build/tools/rpt_bench/rpt_bench \
    --dataset /tmp/data/sift-128-euclidean.hdf5 \
    --bucket_sizes 100,1000,10000 \
    --strategies rpt,random,single_dim \
    --topk 10 \
    --output sift1m.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset`, `-d` | required | Path to the hdf5 dataset |
| `--bucket_sizes`, `-L` | `100,1000,10000` | Comma separated target bucket sizes |
| `--strategies`, `-s` | `rpt,random,single_dim` | Comma separated strategies |
| `--seed` | `0` | Random seed handed to the strategies |
| `--topk`, `-k` | `10` | Ground-truth neighbours used for the locality metrics |
| `--max_queries` | `0` (all) | Limit the queries used for the locality metrics |
| `--output`, `-o` | none | Write the full results as JSON |

## Strategies

| Name | Description |
| --- | --- |
| `rpt` | `vsag::rpt_partition`, the random projection tree partitioner |
| `random` | Shuffle the ids, then cut into equal-sized chunks |
| `single_dim` | Sort the ids by the highest-variance dimension, then cut into equal-sized chunks |
| `kmeans` | `vsag::KMeansCluster` with `ceil(n / L)` clusters; sizes are unbounded |

## Metrics

### Uniformity

`parts`, `max`, `min`, `max/min`, the standard deviation of the partition sizes, and the
Gini coefficient (0 means every partition is the same size). `bounds` reports whether the
partitioning satisfies all of the size guarantees: exactly `ceil(n / L)` partitions, no
partition above `L`, and — unless `n < L` — no partition below `floor(L / 2)` with a
max/min ratio of at most 2.

Note that `rpt`, `random` and `single_dim` are all perfectly balanced by construction, so
they tie on every uniformity metric. The uniformity columns verify compliance; they do not
discriminate between these three strategies.

### Locality

Computed from the dataset's own ground truth, so no brute-force distance matrix is needed.
For each query, its ground-truth top-k neighbours are located in the partitioning:

| Column | Meaning | Better |
| --- | --- | --- |
| `touched` | Mean number of distinct partitions the top-k neighbours fall into | lower |
| `same1` | Fraction of the top-k sharing the partition of the nearest neighbour | higher |
| `scan@90` | Mean number of partitions to scan to cover 90% of the top-k, visiting partitions in the best possible order | lower |

`scan@90` is the metric closest to how a partitioning behaves in practice: it is the
number of shards a query would have to touch to reach 90% recall under an oracle ordering.

### Cost

`build(ms)` is the wall time of the partitioning call. `peak(MiB)` is the growth of the
process high-water RSS across that call. Because the kernel only tracks a high-water mark,
that number is only meaningful for the first strategy that reaches a given peak — run one
strategy per invocation when exact per-strategy peaks matter:

```bash
for s in rpt random single_dim; do
    ./build/tools/rpt_bench/rpt_bench -d /tmp/data/sift-128-euclidean.hdf5 \
        -L 1000 -s "$s" -o "sift1m-$s.json"
done
```

## Interpreting the output

```
strategy           L     parts     max     min  max/min     gini  bounds   build(ms)  peak(MiB)   touched     same1    scan@90
rpt              500        20     500     500     1.00   0.0000     yes         4.4        0.6     1.445     0.861       1.27
random           500        20     500     500     1.00   0.0000     yes         0.3        0.0     8.070     0.141       7.07
single_dim       500        20     500     500     1.00   0.0000     yes         3.1        0.0     2.235     0.680       1.89
kmeans           500        20     500     500     1.00   0.0000     yes        82.7        4.0     1.000     1.000       1.00
```

The three balanced strategies tie on uniformity and separate on locality. KMeans preserves
locality best but gives no size guarantee at all, and `bounds` turns to `no` as soon as the
data is not uniformly distributed over the requested number of clusters.
