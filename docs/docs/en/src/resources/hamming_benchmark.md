# Hamming Kernel Benchmark (`hamming_benchmark`)

`tools/hamming_benchmark` holds two benchmarks for [binary vectors](../guide/binary_vectors.md):

- `hamming_benchmark` (C++) verifies and times every Hamming distance kernel the CPU can run.
- `bench_index.py` (Python) measures the recall and single-thread throughput of binary `hgraph`
  and `brute_force` indexes on the ann-benchmarks Hamming datasets, with the FAISS binary indexes
  as a reference when the `faiss` package is installed.

## Build

Using the project Makefile, enable tools with `VSAG_ENABLE_TOOLS=ON`; `bench_index.py` also needs
the Python bindings (`VSAG_ENABLE_PYBINDS=ON`, or an installed `pyvsag`):

```bash
VSAG_ENABLE_TOOLS=ON make release
# Output: ./build-release/tools/hamming_benchmark/hamming_benchmark
```

When invoking CMake directly, pass `-DENABLE_TOOLS=ON`.

## Kernel Benchmark

```bash
./build-release/tools/hamming_benchmark/hamming_benchmark \
    --dims 64,256,1024 --vectors 2048 --min-time 0.3 --json kernels.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--dims` | `64,128,256,512,1024,2048,4096` | Comma-separated dimensions, in bits |
| `--vectors` | `2048` | Codes per pass (rounded down to a multiple of 4) |
| `--min-time` | `0.3` | Minimum seconds per measurement |
| `--json` | none | Also write the results to this JSON file |

For every dimension the tool first checks that each kernel reproduces the generic distance of
every code, one code at a time and in a scan; a mismatch is printed and the exit status is 1. It
then reports, per kernel, the nanoseconds per distance of three entry points:

| Column | Entry point | Used by |
| --- | --- | --- |
| `single` | `HammingDistance`, one code per call | single distances |
| `batch4` | `HammingDistanceBatch4`, four codes at arbitrary addresses per call | graph search |
| `scan` | `HammingDistanceScan`, one call over all codes stored back to back | `brute_force` scans |

`scan GB/s` is the code bytes read per second by the scan, and the speedup compares the fastest
entry point of a kernel with the generic single-code kernel. A pass reads at most 1 MiB of codes,
so the figures measure computation rather than memory bandwidth. Run the tool on an idle core,
for example `taskset -c 3 ./hamming_benchmark`, and keep the other hyper-thread of that core idle.

## Index Benchmark

```bash
pip install numpy h5py faiss-cpu   # faiss-cpu is optional
wget http://ann-benchmarks.com/sift-256-hamming.hdf5
taskset -c 0-15 python3 tools/hamming_benchmark/bench_index.py \
    --data sift-256-hamming.hdf5 --threads 16 --json sift256.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--data` | required | ann-benchmarks Hamming HDF5 file (`train`, `test`, optional `distances`) |
| `--base` | all | Use at most this many base vectors |
| `--queries` | `1000` | Number of queries |
| `--k` | `10` | Neighbors per query |
| `--ef` | `10 20 40 80 160 320` | `ef_search` values of `hgraph` (and `efSearch` of FAISS HNSW) |
| `--max-degree` | `32` | `max_degree` of `hgraph`; FAISS HNSW uses `M = max_degree / 2` |
| `--ef-construction` | `200` | Construction beam width of both graphs |
| `--threads` | `16` | Build threads; searches always use one thread |
| `--exact-limit` | `2000000` | Skip the exact indexes above this many base vectors |
| `--exact-queries` | `200` | Queries used for the exact indexes |
| `--repeat` | `3` | Passes over the queries per setting; the best pass is reported |
| `--json` | none | Also write the rows to this JSON file |

The script packs the 0/1 elements of the dataset MSB-first with `numpy.packbits`. Recall@k is
tie-aware: a returned neighbor counts when its distance is not larger than the true k-th
distance, taken from the `distances` dataset (ann-benchmarks stores it divided by the number of
bits; the script scales it back) or computed by brute force when the file has none. FAISS
`IndexBinaryHNSW(M)` keeps `2 * M` neighbors on its base layer, the counterpart of `max_degree`.

Pin the process to idle physical cores of one NUMA node so that the build threads and the
memory they allocate stay on that node.

## Reference Results

The figures below were measured on 2026-11-01 on an AMD EPYC 9A75 server (Zen 5 with AVX-512
VPOPCNTDQ, two NUMA nodes, 1.5 TB of memory) running Linux 5.4, with a `make release` build by
GCC 13.4, Python 3.12, NumPy 2.5 and `faiss-cpu` 1.15.1. Searches ran on one idle core; build
threads ran on 16 idle cores of the same NUMA node.

### Kernels

Nanoseconds per distance; the scalar columns give the fastest of the three entry points.

| bits | generic | popcnt | avx2 | avx512bw | vpopcntdq single | vpopcntdq batch4 | vpopcntdq scan | scan GB/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.29 | 0.66 | 0.74 | 0.40 | 1.81 | 0.67 | 0.05 | 153 | 43.9x |
| 100 | 7.73 | 1.96 | 2.05 | 1.25 | 2.01 | 0.97 | 0.51 | 25 | 15.1x |
| 128 | 3.94 | 0.85 | 0.86 | 0.60 | 2.01 | 0.85 | 0.09 | 173 | 42.5x |
| 256 | 5.80 | 1.22 | 1.10 | 1.27 | 2.21 | 1.03 | 0.19 | 170 | 30.9x |
| 512 | 9.98 | 2.06 | 1.40 | 1.03 | 2.21 | 0.84 | 0.44 | 145 | 22.6x |
| 768 | 14.00 | 2.90 | 1.74 | 1.83 | 1.96 | 1.35 | 0.70 | 138 | 20.1x |
| 1000 | 22.26 | 4.98 | 4.24 | 1.81 | 2.09 | 1.19 | 0.77 | 162 | 28.9x |
| 1024 | 17.73 | 3.74 | 2.23 | 1.49 | 1.82 | 1.37 | 0.74 | 173 | 24.0x |
| 2048 | 33.95 | 7.09 | 4.20 | 2.45 | 2.55 | 2.06 | 1.64 | 156 | 20.7x |
| 4096 | 72.99 | 14.21 | 8.16 | 5.21 | 5.28 | 4.60 | 3.53 | 145 | 20.7x |

With AVX-512 VPOPCNTDQ the scan handles eight codes per step with a single horizontal reduction,
and packs two to eight codes into each 64-byte register when a code takes 8, 16 or 32 bytes,
which is why 64-, 128- and 256-bit codes scan fastest. The NEON and SVE kernels were verified bit
for bit under QEMU (NEON only, and SVE with 128- to 2048-bit vectors); no ARM timings are given.

### Exact Search

Random codes, top 10, one query at a time; nanoseconds per scanned vector (lower is better) and
queries per second.

| vectors | bits | `brute_force` ns/vector | `brute_force` QPS | `IndexBinaryFlat` ns/vector | `IndexBinaryFlat` QPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 256 | 1.02 | 49,070 | 0.74 | 67,121 |
| 100,000 | 256 | 0.56 | 17,750 | 0.66 | 15,236 |
| 400,000 | 256 | 0.46 | 5,452 | 0.61 | 4,083 |
| 988,000 | 256 | 0.75 | 1,344 | 0.69 | 1,469 |
| 20,000 | 1024 | 1.90 | 26,321 | 2.51 | 19,917 |
| 100,000 | 1024 | 1.39 | 7,196 | 2.32 | 4,309 |
| 400,000 | 1024 | 2.49 | 1,003 | 2.78 | 898 |
| 988,000 | 1024 | 2.63 | 385 | 2.78 | 364 |

Small sets are dominated by the fixed cost of a query, and the largest ones by memory bandwidth.

### sift-256-hamming

988,258 vectors of 256 bits, 1,000 queries, k = 10, `max_degree` 32, `ef_construction` 200.

| index | ef_search | recall@10 | QPS (1 thread) | build s |
| --- | ---: | ---: | ---: | ---: |
| vsag hgraph | 10 | 0.8838 | 33,118 | 16.2 |
| vsag hgraph | 20 | 0.9619 | 25,224 | 16.2 |
| vsag hgraph | 40 | 0.9914 | 15,935 | 16.2 |
| vsag hgraph | 80 | 0.9977 | 9,294 | 16.2 |
| vsag hgraph | 160 | 0.9993 | 5,227 | 16.2 |
| vsag hgraph | 320 | 0.9998 | 2,875 | 16.2 |
| vsag brute_force | - | 1.0000 | 1,533 | 0.2 |
| faiss IndexBinaryHNSW | 10 | 0.8735 | 57,808 | 10.2 |
| faiss IndexBinaryHNSW | 20 | 0.9501 | 41,033 | 10.2 |
| faiss IndexBinaryHNSW | 40 | 0.9888 | 26,269 | 10.2 |
| faiss IndexBinaryHNSW | 80 | 0.9974 | 14,835 | 10.2 |
| faiss IndexBinaryHNSW | 160 | 0.9995 | 7,898 | 10.2 |
| faiss IndexBinaryHNSW | 320 | 1.0000 | 4,167 | 10.2 |
| faiss IndexBinaryFlat | - | 1.0000 | 1,491 | 0.0 |

### word2bits-800-hamming

399,000 vectors of 800 bits, 1,000 queries, k = 10, `max_degree` 32, `ef_construction` 200.

| index | ef_search | recall@10 | QPS (1 thread) | build s |
| --- | ---: | ---: | ---: | ---: |
| vsag hgraph | 10 | 0.7885 | 28,275 | 6.8 |
| vsag hgraph | 20 | 0.8830 | 22,718 | 6.8 |
| vsag hgraph | 40 | 0.9404 | 14,932 | 6.8 |
| vsag hgraph | 80 | 0.9717 | 9,075 | 6.8 |
| vsag hgraph | 160 | 0.9856 | 5,308 | 6.8 |
| vsag hgraph | 320 | 0.9928 | 2,965 | 6.8 |
| vsag brute_force | - | 1.0000 | 1,134 | 0.1 |
| faiss IndexBinaryHNSW | 10 | 0.7775 | 43,734 | 5.4 |
| faiss IndexBinaryHNSW | 20 | 0.8720 | 27,714 | 5.4 |
| faiss IndexBinaryHNSW | 40 | 0.9332 | 17,036 | 5.4 |
| faiss IndexBinaryHNSW | 80 | 0.9649 | 9,911 | 5.4 |
| faiss IndexBinaryHNSW | 160 | 0.9829 | 5,510 | 5.4 |
| faiss IndexBinaryHNSW | 320 | 0.9893 | 2,965 | 5.4 |
| faiss IndexBinaryFlat | - | 1.0000 | 979 | 0.0 |

At the same `ef_search`, `hgraph` returns more of the true neighbors than FAISS HNSW at every
setting on word2bits-800-hamming and up to `ef_search` 80 on sift-256-hamming, while FAISS
answers a query with the same `ef_search` faster. Compare the two graphs at equal recall, and
measure on the target hardware before drawing conclusions.
