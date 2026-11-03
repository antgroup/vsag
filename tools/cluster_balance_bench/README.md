# cluster_balance_bench

Reproducible comparison of the plain k-means and balanced k-means IVF
trainers (issue #2870).

`cluster_balance_bench` trains `KMeansCluster` and `BalancedKMeansCluster` on
seeded synthetic datasets - uniform, deliberately skewed, a skewed set whose
size is not divisible by `k`, and a duplicated-vector set - and reports per
trainer and dataset:

- minimum/maximum/mean cluster size, size standard deviation and coefficient
  of variation;
- clustering loss (mean squared distance from each vector to its assigned
  centroid, recomputed from the returned labels with the same yardstick for
  both trainers);
- training time;
- peak memory allocated by the trainer beyond the dataset (through a
  recording allocator), with the size of a dense `n x k` float distance
  matrix printed for reference - the balanced trainer must stay far below it.

## Build

The binary is part of the tools build (`-DENABLE_TOOLS=ON`, requires
`-DENABLE_CXX11_ABI=ON`, Linux):

```bash
make release EXTRA_DEFINED="-DENABLE_TOOLS=ON"
# or
cmake --build <build-dir> --target cluster_balance_bench
```

## Usage

```text
cluster_balance_bench [--dataset all|uniform|skewed|indivisible|duplicates]
                      [--n 50000] [--dim 64] [--k 256] [--iter 25]
                      [--seed 42] [--threads N]
```

Every run is fully seeded; repeated invocations with the same arguments
reproduce the reported clustering numbers up to timing noise.
