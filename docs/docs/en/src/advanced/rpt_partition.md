# Random Projection Tree Partitioning

`vsag::rpt_partition` splits a set of dense vectors into **near-uniform partitions** using a
random projection tree (RPT). It is a standalone utility: it does not build an index and does not
retain the input data. Typical uses are sharding a large collection before building per-shard
indexes, balancing parallel build jobs, or producing locality-preserving buckets for downstream
algorithms.

Compared with random bucketing or sorting on a single dimension, an RPT keeps nearby vectors in
the same partition far more often while still producing partitions of almost identical size.

## API Overview

```cpp
struct RPTPartitionParams {
    uint64_t bucket_size{1000};  // target bucket size L
    uint64_t max_depth{64};      // upper bound on the tree depth
    uint64_t seed{0};            // random seed
};

tl::expected<std::vector<std::vector<int64_t>>, Error>
rpt_partition(uint64_t d, uint64_t n, const float* x, const RPTPartitionParams& params);
```

Declarations live in
[`include/vsag/utils.h`](https://github.com/antgroup/vsag/blob/main/include/vsag/utils.h).

| Parameter | Description |
| --- | --- |
| `d` | Dimensionality of the vectors. Must be positive. |
| `n` | Number of vectors. `n == 0` returns an empty result. |
| `x` | Row-major array of `n * d` floats. Owned by the caller; neither modified nor retained. |
| `params.bucket_size` | Target bucket size `L`. Every partition holds at most `L` vectors. Must be positive. |
| `params.max_depth` | Safety bound on recursion depth. When reached, the remaining vectors are split into equal chunks by id order so the size guarantees still hold. |
| `params.seed` | Random seed. The same input and seed always yield the same partitions. |

The result is one `std::vector<int64_t>` per partition, holding the row indices into `x`.

## Size Guarantees

Given `n` vectors and bucket size `L`, let `K = ceil(n / L)`. The output always satisfies:

- exactly `K` partitions are produced;
- every partition holds at most `L` vectors;
- when `n >= L`, every partition holds at least `floor(L / 2)` vectors, and the largest partition
  is at most twice the size of the smallest one.

These guarantees hold by construction and do not depend on the data distribution or on the
seed. Every partition holds either `floor(n / K)` or `ceil(n / K)` vectors.

## How It Works

A plain RPT cuts every node at its median until the node fits into a bucket. That yields a power
of two leaves, overshooting `ceil(n / L)` by up to 45% and leaving buckets only ~64% full.

Instead, every node is assigned the number of leaves `k = ceil(n_node / L)` it must produce.
With `n_node = q * k + r`, the node is cut so that the left child receives `ceil(k / 2)` leaves
and `q * ceil(k / 2) + min(r, ceil(k / 2))` vectors. The cut position is found with
`std::nth_element` on the projection values; ties are broken by vector id, so duplicated vectors
and constant projections are handled without special cases. Vectors whose projection is `NaN`
are ranked last.

Each node draws its projection direction from a generator seeded by a hash of the user seed and
the node position, so the output is independent of the traversal order.

Memory overhead beyond the input is one id permutation (`n * 4` bytes), one projection buffer
(`n * 4` bytes) and one direction vector (`d * 4` bytes). No vector data is copied.

## Example

```cpp
#include <vsag/vsag.h>

vsag::RPTPartitionParams params;
params.bucket_size = 1000;
params.seed = 42;

auto result = vsag::rpt_partition(dim, num_vectors, datas.data(), params);
if (not result.has_value()) {
    std::cerr << result.error().message << std::endl;
    return -1;
}
for (const auto& partition : result.value()) {
    // partition holds the row indices of one bucket
}
```

A complete program is available at
[`examples/cpp/601_utils_rpt_partition.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/601_utils_rpt_partition.cpp).

## Errors

| Condition | Result |
| --- | --- |
| `d == 0` | `INVALID_ARGUMENT` |
| `params.bucket_size == 0` | `INVALID_ARGUMENT` |
| `x == nullptr` and `n > 0` | `INVALID_ARGUMENT` |
| `n` larger than `2^32 - 1` | `INVALID_ARGUMENT` |
