# Lite persistent Add benchmark

This benchmark compares an in-memory Add, an Add followed by synchronous full
snapshot replacement, and the opt-in synchronous Add log. Each run starts from
the same saved index for its storage mode. All three paths use the same input
vectors, external IDs and search settings. The memory-only control exports a
snapshot after timing so a fresh reader can verify its final state. The
snapshot control syncs a temporary file, replaces the old file in the same
directory and syncs the directory.

## Reproduce

Use the prepared SIFT and GIST inputs from the
[Lite benchmark inputs](https://github.com/antgroup/vsag/blob/ec110b60433377493c91d304f05fc1023812a4ee/lite/benchmark/ACCEPTANCE_REPORT.md).
Each input directory contains little-endian `base.fvecs` and `queries.fvecs`.
The preparation tools are [prepare_sift.py](https://github.com/antgroup/vsag/blob/ec110b60433377493c91d304f05fc1023812a4ee/lite/benchmark/prepare_sift.py)
and [prepare_gist.cpp](https://github.com/antgroup/vsag/blob/ec110b60433377493c91d304f05fc1023812a4ee/lite/benchmark/prepare_gist.cpp).
Compare the prepared input hashes with [results.json](results.json).

```bash
cmake -S lite -B build-lite -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_RABITQ_LITE_BACKEND=ON -DENABLE_LITE_PERSISTENT_ADD=ON
cmake --build build-lite --parallel 2
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -I include \
  lite/benchmark/persistent_add/benchmark.cpp -L build-lite -lvsag-lite \
  -Wl,-rpath,"$PWD/build-lite" -o build-lite/persistence-benchmark
python3 lite/benchmark/persistent_add/run.py \
  --benchmark build-lite/persistence-benchmark --library build-lite/libvsag-lite.so \
  --sift "$SIFT_PREPARED" --gist "$GIST_PREPARED" --output persistent-add-results
```

On Linux, pass `--cpu N` to pin the benchmark to an available CPU. The output
directory must be new and on the filesystem to measure. The runner verifies
the dynamic-library path, compares exported snapshot and ordered neighbor
bytes after each reopen, and preserves the raw samples and result hashes.

## Protocol

- SIFT 1,000 and 10,000 rows, GIST 10,000 rows, with BruteForce, FP32 graph,
  FP16 graph and RaBitQ8. Graph degree 16, build and query budget 128, Top-10.
- Each combination runs 32 adds with three fresh processes per variant and a
  rotating variant order. Add IDs are consecutive after the seed IDs. Input
  comes from the first rows of the base file with a fixed offset on coordinate
  zero. Each fresh reader checks 100 query rows. BruteForce gives exact Top-10
  on the same live IDs for the recall calculation.
- Replay scaling uses SIFT 1,000 with FP32 graph and RaBitQ8, for 0, 128 and
  1,024 log entries. Each point has three journal and three memory-control
  runs.
- Wall-clock Add percentiles and throughput exclude index initialization and
  recovery. Reported P50/P99 are computed per process and then medianed across
  repeats. Write bytes are logical bytes attributable to the measured Adds.
  Open time uses a fresh process and the host's existing page cache. Peak RSS
  covers the recovery process including the subsequent queries.
- The recorded run used Linux x86-64, GCC 11.4, an ext4 filesystem and CPU 7.
  Other unit and functional test processes were active. Treat timings as
  observations from that host. Raw metadata records the observed workload.

## Results

| Seed | Storage | Full snapshot write bytes | Add log write bytes | Add P50 snapshot / log (ms) | Open snapshot / log (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| SIFT 1,000 | BruteForce | 16,916,096 | 16,768 | 15.17 / 7.15 | 0.31 / 1.63 |
| SIFT 1,000 | FP32 | 21,340,416 | 16,768 | 15.75 / 7.26 | 0.56 / 3.89 |
| SIFT 1,000 | FP16 | 13,013,248 | 16,768 | 13.90 / 6.09 | 0.43 / 3.25 |
| SIFT 1,000 | RaBitQ8 | 9,649,792 | 16,768 | 13.44 / 7.02 | 0.57 / 4.09 |
| SIFT 10,000 | BruteForce | 166,676,096 | 16,768 | 27.71 / 1.74 | 2.33 / 14.45 |
| SIFT 10,000 | FP32 | 210,268,416 | 16,768 | 35.89 / 1.68 | 4.46 / 23.01 |
| SIFT 10,000 | FP16 | 128,213,248 | 16,768 | 23.33 / 1.75 | 3.64 / 15.54 |
| SIFT 10,000 | RaBitQ8 | 94,897,792 | 16,768 | 17.94 / 1.78 | 4.71 / 15.76 |
| GIST 10,000 | BruteForce | 1,233,393,280 | 123,264 | 180.16 / 1.76 | 14.47 / 100.19 |
| GIST 10,000 | FP32 | 1,276,985,600 | 123,264 | 189.11 / 2.03 | 16.79 / 114.09 |
| GIST 10,000 | FP16 | 661,571,840 | 123,264 | 104.69 / 2.06 | 11.12 / 65.29 |
| GIST 10,000 | RaBitQ8 | 361,696,896 | 123,264 | 57.34 / 2.18 | 8.05 / 46.23 |

| GIST 10,000 storage | Snapshot peak RSS (KiB) | Log before replay fix (KiB) | Log after replay fix (KiB) |
| --- | ---: | ---: | ---: |
| BruteForce | 41,796 | 78,968 | 41,824 |
| FP32 | 45,040 | 82,576 | 45,564 |
| FP16 | 26,212 | 45,088 | 26,532 |
| RaBitQ8 | 21,388 | 35,512 | 21,400 |

| SIFT 1,000 storage | Open with 0 records (ms) | 128 records (ms) | 1,024 records (ms) |
| --- | ---: | ---: | ---: |
| FP32 | 1.99 | 8.33 | 54.19 |
| RaBitQ8 | 1.21 | 11.77 | 85.31 |

[results.json](results.json) contains all metrics including Add throughput,
query P50/P99, Recall@10, recovery RSS and input/library identities.
[raw.json.gz](raw.json.gz) contains the per-call samples, file and result hashes,
and ordered Top-10 IDs. The pre-fix replay result
[summary](before-replay-reserve.json) and [raw samples](before-replay-reserve-raw.json.gz)
come from code revision `c078950` and use the same benchmark executable and
inputs. The outputs before and after the replay fix were byte-identical in all
18 compared cells.

The first replayed Add could previously grow a full vector buffer.
RaBitQ also used a full graph copy for its first Add with cold incoming counts.
The final replay reserves capacity before loading the checkpoint and prepares
RaBitQ's incoming counts before replay. Those changes preserve the ordinary
snapshot format and public Add semantics.

The log stores the original FP32 input even for compressed storage. Its size
and replay work grow with the number of Adds. Use ordinary snapshot export to
start a new persistent directory when an application wants a fresh checkpoint.
The recovery contract covers one writer, serialized calls, local filesystems
and process exit.

[library-size.json](library-size.json) compares unstripped standalone shared
libraries built with GCC 11.4 Release and RaBitQ off. The upstream baseline
`774fd84` is 194,016 B, the feature branch with persistence off is 194,240 B,
and the feature branch with persistence on is 237,000 B. The gzip sizes
measure those same libraries, excluding an application or dependency bundle.
