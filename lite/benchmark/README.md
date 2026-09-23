# VSAG Lite baseline experiment

This experimental tool records a deterministic standalone Lite BruteForce baseline. It does not claim a performance improvement.

See [FINAL_REPORT.md](FINAL_REPORT.md) for the consolidated current-head and historical evidence.
See [RABITQ_LITE_FEASIBILITY.md](RABITQ_LITE_FEASIBILITY.md) for the source-based boundary and next experimental gate.\

## Directory layout

- `lite/benchmark/main.cpp`, `dataset_main.cpp`, and `run_*.sh`:
  deterministic Lite baseline, CRUD, load, and SIFT runners.
- `lite/benchmark/full/`, `full_main.cpp`, and `full_dataset_main.cpp`:
  separate Full VSAG comparison consumers.
- `lite/benchmark/prepare_sift.py`: prepares the documented SIFT subsets.
- `lite/benchmark/quantization_probe.cpp`: opt-in SQ8/FP16 scan experiment.
- `lite/benchmark/rabitq_lite_layout_probe.cpp`: opt-in RaBitQ 3+5 bit-plane differential probe.
- `lite/benchmark/rabitq_lite_codec_probe.cpp`: deterministic FHT training, 3-bit lower-bound filtering, 5-bit supplement reranking, and filter-first graph traversal probe.
- `lite/benchmark/prepare_gist.cpp`: bounded GIST fbin-prefix converter that recomputes exact squared-L2 Top-10 ground truth for each selected scale.
- `src/lite/fp16_codec.h` and the internal FP16 factory/tests in
  `src/lite/graph_backend.cpp` and `src/lite/graph_backend_test.cpp`:
  candidate graph experiment, not a public `Index` backend.
- `lite/CMakeLists.txt`: `ENABLE_BENCHMARKS` and
  `ENABLE_QUANTIZATION_PROBE` opt-in build switches.

The public FP32 graph, filtering, CRUD, and snapshot implementation comes from
the Lite feature branch; this experiment branch adds measurements and internal
quantization candidates.

Build it with `-DENABLE_BENCHMARKS=ON`, then run:

```bash
lite/benchmark/run_baseline.sh build-lite-baseline/lite_benchmark baseline-results
```

The runner executes fixed 10k x 128 and 100k x 128 cases. Each case writes one CSV row, a Lite snapshot, and `/usr/bin/time -v` process measurements. It also records the commit, CPU, compiler, CMake version, OS, and benchmark binary checksum in `environment.txt`. The CSV includes build, CRUD, search, save/warm-load, snapshot-size, process peak-RSS, top-1 self-query recall, and a result checksum. The fixed seed makes generated vectors reproducible. In the baseline and Full/Lite comparison CSVs, `peak_rss_kib` is the process-lifetime peak reported by `getrusage(RUSAGE_SELF)`; it covers build, CRUD, search, save, and warm load, not just the load phase.

The same runner also executes a 20-round CRUD stability case. Every round performs 500 Update-Remove-Add cycles while preserving 10,000 live vectors, verifies exact self-query results before and after Save-Load, and records per-round latency, resident/peak RSS, snapshot size, recall, and checksum in `crud-stability.csv`. The snapshot is deliberately overwritten inside a newly created output directory so the experiment retains one verifiable final snapshot instead of twenty identical-size files.

The stability command can also be run directly:

```bash
lite_benchmark stability COUNT DIM ROUNDS CRUD_OPS QUERIES K SEED SNAPSHOT_DIRECTORY
lite_benchmark graph-stability COUNT DIM ROUNDS CRUD_OPS QUERIES K SEED MAX_DEGREE EF_SEARCH SNAPSHOT_DIRECTORY
```

The graph stability mode applies the same update-remove-readd cycle after an explicit
`BuildGraph`. It records the active count, v2 snapshot size, current and peak RSS, recall,
and round-trip checksum together with the graph parameters. The default runner uses a
2,000 x 32, 10-round configuration so this diagnostic remains practical despite physical
graph repair on every removal. Stable RSS and snapshot size show bounded behavior for that
recorded workload; they do not prove that the allocator returns memory to the operating
system or that all workloads are fragmentation-free.

At commit `66f2a18`, the Release runner completed the default graph case with 2,000
live 32-dimensional vectors, degree 12, ef_search 64, and 100 update-remove-readd
cycles per round. All 10 rounds retained 2,000 live IDs and passed Save/Load result
checks. The v2 snapshot stayed within 442,544–472,456 bytes rather than growing
monotonically. Current RSS was 4,300 KiB after round 1 and 5,044 KiB after round 10;
the process-lifetime peak remained 5,044 KiB. Top-1 self-query recall ranged from
0.875 to 1.000, so this run is storage/churn evidence rather than a graph-quality
target. The raw CSV, `/usr/bin/time -v` output, environment record, summary, and
SHA-256 hashes are retained outside Git with the experiment artifacts.

After generating the two baseline snapshots, the runner starts seven separate loader processes for each scale. Every loader reports snapshot size, `load_ms`, first-query latency, follow-up query P50/P99, current/peak RSS, result checksum, and final index size. The known ID 0 query uses variant 1 because the fixed baseline CRUD sequence updates that ID before saving. The loader repeats this same query for every follow-up search, so its `search_p50_us` and `search_p99_us` measure repeated-query latency rather than a distribution across distinct queries. Its `peak_rss_kib` covers the entire loader process.

The loader can also be run directly for a snapshot generated by this benchmark:

```bash
lite_benchmark load SNAPSHOT DIM QUERY_ID QUERY_VARIANT QUERIES K SEED
```

The loader is a fresh process, but the runner does not evict the snapshot from the operating-system page cache. Its CSV therefore records `page_cache_control=uncontrolled`, and the results must be described as fresh-process load measurements rather than strict cold-load measurements.

`warm_load_ms` is measured in the same process immediately after saving and must not be reported as cold-start latency. A strict cold-load experiment must additionally control and document the operating-system page cache.

### Batched snapshot-load comparison

Commit `fb36355` replaces per-value stream calls with bulk reads directly into the final
owned ID, FP32 vector, and adjacency containers. It keeps the v1/v2 little-endian formats
and converts payload values in place on non-little-endian hosts. This remains an
owned-memory load rather than mmap or zero-copy.

On 2026-09-20, the same Release executables alternated only the loaded shared library
between parent `64bbed5` and `fb36355`. The v1 generated 100k x 128 snapshot used seven
runs per version. The v2 graph case used five runs per version on the 100k SIFT snapshot
at degree 16 / ef 128; every run executed the existing `[lite-sift-load]` assertions and
100 queries. Page cache state was uncontrolled, so these are fresh-process comparisons,
not strict cold-load measurements.

| Snapshot | Before median Load (ms) | Batched median Load (ms) | Speedup | Result check |
| --- | ---: | ---: | ---: | --- |
| v1 BruteForce, 52,000,048 bytes | 192.956 | 22.436 | 8.60x | identical checksum and 100,000 IDs |
| v2 graph, 65,600,064 bytes | 250.835 | 34.652 | 7.24x | Recall@10 0.946 and all 611 assertions passed |

Median process RSS was effectively unchanged: 58,448 versus 58,428 KiB for v1 and
76,172 versus 76,140 KiB peak RSS for v2. The raw stdout, `/usr/bin/time -v` files,
commands, and exact-commit artifact directories are retained outside Git. The comparison
does not measure mmap, zero-copy loading, native big-endian execution, or strict cold I/O.

The runners check that their benchmark executables and library inputs exist before creating an output directory, and refuse to overwrite an existing output directory or snapshot. Run them on an otherwise idle machine and retain the compiler, commit SHA, CPU, and raw output with any report.

## Full/Lite comparison

`full_main.cpp` exercises the public Full VSAG BruteForce API with the same generated FP32 vectors, query sequence, dimensions, seed, and operation counts as the Lite baseline. Full removal explicitly uses `RemoveMode::FORCE_REMOVE`, matching Lite's immediate last-record-to-hole removal semantics; the default Full `MARK_REMOVE` behavior is not equivalent.

Build and install Full VSAG from the same commit, build `lite/benchmark/full` against that installation, and build the Lite benchmark with `ENABLE_BENCHMARKS=ON`. Then run:

```bash
lite/benchmark/run_comparison.sh \
  FULL_BENCHMARK LITE_BENCHMARK FULL_LIBVSAG_SO LITE_LIBVSAG_LITE_SO OUTPUT_DIRECTORY
```

The runner alternates Full/Lite execution order over seven repetitions for both 10k x 128 and 100k x 128 cases. It records raw and stripped shared-library sizes, binary and library checksums, per-process `/usr/bin/time -v` output, CSV measurements, and snapshots. It extracts the single CSV header/data pair from captured stdout and fails if the pair is missing, duplicated, or malformed; Full VSAG diagnostic logs remain in the raw stdout file. Results compare two exact FP32 squared-L2 BruteForce implementations; they do not establish graph-index performance or standard-dataset Recall@K.

## SIFT-128 subset Recall@10

The optional `lite_dataset_benchmark` evaluates independent queries from the public
[ANN-Benchmarks SIFT-128 Euclidean dataset](https://github.com/erikbern/ann-benchmarks).
Install Python `numpy` and `h5py` in an isolated environment, download the HDF5
file to a directory outside the repository, and prepare the two prefixes:

```bash
curl -fL -o /data/sift-128-euclidean.hdf5 https://ann-benchmarks.com/sift-128-euclidean.hdf5
python3 lite/benchmark/prepare_sift.py /data/sift-128-euclidean.hdf5 /data/sift-prepared
cmake -S lite -B build-lite-baseline -DCMAKE_BUILD_TYPE=Release -DENABLE_BENCHMARKS=ON
cmake --build build-lite-baseline --target lite_dataset_benchmark
lite/benchmark/run_sift.sh build-lite-baseline/lite_dataset_benchmark /data/sift-prepared /data/sift-results
```

The preparation script takes the first 10k and 100k training vectors and the
first 100 independent test queries. It recomputes squared-L2 exact Top-10
ground truth separately on each selected base prefix (ties by ascending ID).
The HDF5 file's original 1M-base neighbors must not be reused for a subset.
Each `manifest.json` records the source and converted-file SHA-256 hashes;
the output directory must be new. The runner records seven raw CSVs, process
measurements, environment and per-scale manifests. The executable verifies
each query's search result against the selected-prefix ground truth, then
checks exact result and distance equality after Save/Load. These measurements
cover standalone Lite exact BruteForce only; they do not compare Full on this
dataset, imply an ANN quality improvement, or measure strict cold loading.
The scripts require a little-endian host for the fvecs/ivecs interchange files.

To compare Full and Lite BruteForce on the same prepared SIFT subsets, build
the independent Full consumer against a Full VSAG installation from this
commit and run the alternating seven-round comparison:

```bash
cmake -S lite/benchmark/full -B build-full-dataset -DCMAKE_PREFIX_PATH=/path/to/full-install
cmake --build build-full-dataset --target full_dataset_benchmark
lite/benchmark/run_sift_comparison.sh build-full-dataset/full_dataset_benchmark \
  build-lite-baseline/lite_dataset_benchmark /path/to/full-install/lib/libvsag.so \
  build-lite-baseline/libvsag-lite.so /data/sift-prepared /data/sift-comparison
```

Each implementation gets the same base/query/ground-truth files and Top-10
definition. Full uses FP32 squared-L2 BruteForce and its own snapshot format;
it copies query values into an existing Dataset outside the timed search.
The runner retains raw process stdout, stderr, CSV, snapshots, environment,
and seven executions per scale per implementation. The Full library and Lite
library build identities should be recorded alongside the executable hashes
when reporting this comparison.

# SQ8 and FP16 candidate probe (not an index backend)

This standalone executable evaluates two possible Lite quantization formats. It
does **not** change Index, graph traversal, the v1/v2 snapshot, or process
resident memory of the current Lite index. The measured `sq8_code_bytes`
and `fp16_code_bytes` count only encoded vectors; the probe also keeps the
FP32 base dataset and training model in memory to calculate exact ground truth and reconstruction
error. Its process RSS is therefore not the memory use of a quantized index.

The probe reuses the repository's generic scalar SQ8 squared-L2 kernel and
`ClampScalarQuantizationDelta`. It trains **classic per-dimension minimum and
maximum bounds** on the selected base vectors, encodes one unsigned byte per
dimension, and searches all encoded vectors with FP32 queries. Full VSAG's
`ScalarQuantizer` normally uses a sampled/truncated-bound trainer and brings
additional allocator, parameter and SIMD-dispatch dependencies; this probe is
not bit-identical to that full training configuration.

FP16 uses a standalone portable IEEE binary16 encoder with round-to-nearest-even
and the repository's generic half-precision squared-L2 kernel. It rejects
values outside the finite FP16 range. The encoder is not Full VSAG's
`FloatToFP16` implementation; this probe does not test production format
compatibility. Both the base and query vectors are encoded for FP16 search.
The self-test includes positive and negative zero, a subnormal, a halfway
rounding case, the maximum finite value, and overflow rejection.

Configure the independent Lite build and run a self-test:

    cmake -S lite -B /path/to/build-quantization -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_QUANTIZATION_PROBE=ON
    cmake --build /path/to/build-quantization --target lite_quantization_probe -j2
    /path/to/build-quantization/lite_quantization_probe --self-test

Run the prepared 10k or 100k SIFT-128 subset from the experiment workflow:

    /usr/bin/time -v -o /path/to/results/quantization-100k.time.txt /path/to/build-quantization/lite_quantization_probe /path/to/scale-100000 > /path/to/results/quantization-100k.csv 2> /path/to/results/quantization-100k.stderr.txt

The program checks that the FP32 exhaustive top-10 matches the supplied exact
ground truth on 100 independent queries, then reports SQ8 and FP16
Recall@10, RMSE, query P50/P99 and encoded byte counts. Search times are
exploratory, rotate FP32/SQ8/FP16 order per query, and use generic scalar
kernels. They are neither graph-search latency nor an optimized SIMD
comparison. The tool assumes Linux x86_64 little-endian fvecs/ivecs input. It rejects missing,
truncated, wrong-dimension, non-finite and inconsistent datasets.

On 2026-09-14, three serial 100k SIFT-128 runs measured the following
candidate outcomes (median of each run's query P50):

| Scan | Recall@10 | P50 | Encoded vector bytes |
| --- | ---: | ---: | ---: |
| FP32 exact | 1.000 | 8.855 ms | 51,200,000 |
| SQ8 | 0.988 | 34.145 ms | 12,800,000 plus 1,024-byte model |
| FP16 | 1.000 | 47.474 ms | 25,600,000 |

These timings are from generic scalar exhaustive scans, not the Lite graph.
The FP16 result is lossless on original integer-valued SIFT coordinates:
its reconstruction RMSE was zero. As an additional **transformed input**
check, multiplying the 10k SIFT base and queries by 0.001 produced nonzero
FP16 RMSE (0.000009) and Recall@10 1.000 against FP32 exact truth. This
transformed case is not a standard dataset result and does not establish
quality for general floating-point embeddings. The probe keeps all three
representations in memory; encoded bytes are not index RSS or snapshot size.

The exhaustive scan remains an offline algorithm comparison. FP16 graph storage
is now exposed by the Lite API and uses the official SIMD-dispatched half
kernel plus the separately versioned v3 snapshot; SQ8 remains an offline
candidate.

## Public FP16 graph resident memory

The hidden `[lite-sift-rss]` probe builds either the default FP32 graph or
`VectorStorage::FP16` through the public Index API in a fresh process. It
releases the input base vectors, calls `malloc_trim(0)` on glibc, executes the
same 100 independent queries, and reads current `VmRSS` from
`/proc/self/status`. Queries and ground truth remain resident, so the result
is a same-process-layout comparison rather than an exact container byte count.

    VSAG_SIFT_DIR=/path/to/scale-100000 VSAG_SIFT_RSS_BACKEND=fp32 /path/to/lite_graph_tests '[lite-sift-rss]'
    VSAG_SIFT_DIR=/path/to/scale-100000 VSAG_SIFT_RSS_BACKEND=fp16 /path/to/lite_graph_tests '[lite-sift-rss]'

On 2026-09-20, one fresh process per backend produced:

| SIFT-128 subset | Storage | Recall@10 | Current RSS |
| --- | --- | ---: | ---: |
| 10k | FP32 | 0.973 | 12,036 KiB |
| 10k | FP16 | 0.973 | 9,492 KiB |
| 100k | FP32 | 0.946 | 77,424 KiB |
| 100k | FP16 | 0.946 | 52,236 KiB |

FP16 reduced current RSS by 21.1% at 10k and 32.5% at 100k in these single
runs while preserving measured Recall@10. The 100k FP16 graph also stores
25,600,000 vector-code bytes instead of 51,200,000 FP32 bytes; IDs, links,
containers, queries, ground truth, allocator behavior, and the process image
explain why total RSS does not fall by 50%. These are single-host, single-run
measurements and not a variance or cross-platform study.

## FP16 v3 snapshot size and fresh-process load

The hidden `[lite-sift]` probe accepts `VSAG_GRAPH_STORAGE=fp32|fp16`; omitting the variable preserves the FP32 default. It builds through the public API, writes a v2 FP32 or v3 FP16 snapshot, reloads it, verifies the active storage, and checks exact result and distance equality across the round trip. The `[lite-sift-load]` probe discovers the stored representation after loading and reports first-query latency separately from follow-up-query P50/P99, together with current and process-lifetime peak RSS.

At commit `bcd2388`, one snapshot per storage and scale was generated and each snapshot was loaded in seven fresh processes with alternating FP32/FP16 execution order. The table reports the single build/save observation and the median of seven load processes. Page cache state was uncontrolled, so Load is a fresh-process metric rather than strict cold I/O.

| SIFT-128 subset | Storage | Snapshot bytes | Recall@10 | Build (ms) | Save (ms) | Load median (ms) | First query median (us) | Follow-up P50/P99 median (us) | Steady RSS median (KiB) | Peak RSS median (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | FP32 v2 | 6,560,064 | 0.973 | 864.663 | 26.526 | 3.373 | 129.207 | 100.097 / 128.808 | 11,724 | 11,764 |
| 10k | FP16 v3 | 4,000,064 | 0.973 | 843.392 | 24.027 | 27.149 | 175.937 | 108.498 / 156.537 | 9,260 | 14,064 |
| 100k | FP32 v2 | 65,600,064 | 0.946 | 20,178.9 | 276.240 | 34.874 | 459.440 | 337.664 / 462.920 | 75,808 | 75,908 |
| 100k | FP16 v3 | 40,000,064 | 0.946 | 17,230.1 | 243.660 | 270.980 | 455.701 | 308.394 / 432.701 | 50,752 | 100,600 |

The v3 snapshot is 39.0% smaller at both scales. Its median steady RSS is 21.0% lower at 10k and 33.1% lower at 100k, with unchanged measured Recall@10. The current v3 Load path is 8.05x slower at 10k and 7.77x slower at 100k, and its process peak RSS is higher. `Index::Load` currently reads every binary16 value through a per-value stream operation into a temporary FP32 vector; `GraphBackend::Restore` then encodes that FP32 vector back into final FP16 storage. This double conversion explains the observed load-time and transient-memory bottleneck and identifies a targeted follow-up optimization. Query latency differences are mixed and should not be generalized from one snapshot generation.

Raw stdout, stderr, `/usr/bin/time -v`, environment metadata, snapshot SHA-256 values, and the generated summary are retained in `/home/ubuntu/project/vsag-lite-fp16-v3-load-validation-20260920-bcd2388` on the measured host. The experiment does not measure mmap, zero-copy loading, strict cold I/O, native big-endian execution, or cross-host variance.

### Direct v3 restore follow-up

Feature commit `5ac36ef` removes the FP32 staging copy. `Index::Load` bulk-reads portable binary16 values into a `uint16_t` container, byte-swaps in place when required, rejects non-finite exponent patterns without decoding to float, and moves the container directly into the FP16 graph backend. Versions 1 and 2 and the v3 bytes remain unchanged.

The comparison reused the exact `[lite-sift-load]` executable from probe commit `bcd2388` (rebased as `73f78a3`), the same v3 snapshot per scale, and switched only `LD_LIBRARY_PATH` between feature baseline `73f9c4c` and `5ac36ef`. Each version ran in seven fresh processes with alternating order. Values below are medians; page cache state remained uncontrolled.

| SIFT-128 subset | Library | Load (ms) | First query (us) | Follow-up P50/P99 (us) | Steady RSS (KiB) | Peak RSS (KiB) | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | Before | 27.089 | 169.066 | 107.697 / 152.277 | 9,204 | 14,012 | 0.973 |
| 10k | Direct restore | 2.580 | 124.417 | 95.097 / 123.678 | 9,200 | 9,240 | 0.973 |
| 100k | Before | 270.248 | 441.151 | 297.024 / 406.631 | 50,752 | 100,600 | 0.946 |
| 100k | Direct restore | 26.168 | 397.111 | 283.013 / 381.693 | 50,748 | 50,848 | 0.946 |

Direct restore improved median v3 Load by 10.50x at 10k and 10.33x at 100k. Median process peak RSS fell by 34.1% and 49.5%; steady RSS, snapshot bytes, snapshot SHA-256, and Recall@10 remained unchanged. Query timings were not the optimization target and show no regression in these runs. The result demonstrates removal of the measured conversion bottleneck, while retaining the owned-memory, non-mmap load design and the same strict-cold-I/O limitation.

All 28 loader processes passed their assertions with empty stderr. Raw stdout, stderr, `/usr/bin/time -v`, environment and binary hashes, snapshot hashes, and summaries are retained in `/home/ubuntu/project/vsag-lite-fp16-v3-load-compare-20260920-5ac36ef`.

## Full HGraph RaBitQ reference probe

`full_rabitq_dataset_benchmark` is an opt-in Full VSAG consumer used to evaluate
official RaBitQ behavior before proposing a Lite storage format. It compares the
same public HGraph implementation at degree 16 and `ef_search=128` in three modes:

- `fp32`: FP32 base storage.
- `rabitq1`: one-bit RaBitQ traversal with FP32 reorder, the documented safe default.
- `rabitq3x5`: three filter bits plus five supplement bits in split storage.

The probe uses public `Factory`, `Index::Build`, search, Serialize, and Deserialize
APIs. It reports Recall@10, build and query latency, snapshot size, and process RSS,
and requires identical results after round-trip loading. It does not copy or expose
the internal RaBitQ quantizer in Lite.

Build the independent Full consumer and run seven alternating repetitions:

```bash
cmake -S lite/benchmark/full -B build-full-rabitq \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/full-vsag-install
cmake --build build-full-rabitq --target full_rabitq_dataset_benchmark
lite/benchmark/run_rabitq_reference.sh \
  build-full-rabitq/full_rabitq_dataset_benchmark \
  /path/to/full-vsag-install/lib/libvsag.so \
  /path/to/prepared-sift /new/output-directory
```

This is a Full HGraph reference, not a Lite graph benchmark. It is suitable for
checking achievable quality, storage, and dependency cost, but algorithm-level
differences prevent attributing Full-versus-Lite differences solely to quantization.

### Measured RaBitQ reference results

At experiment commit `844790629e57bcd89809dbba58b260f8a74d74e1`,
the runner executed seven fresh processes per scale and mode, rotating the mode
order on each repetition. All 42 processes passed the serialize/deserialize
identity checks and produced empty stderr. Values below are medians; brackets
show the observed Recall@10 or P99 range where it materially affects
interpretation.

| SIFT-128 subset | Mode | Recall@10 median [range] | Build (ms) | Query P50/P99 median (us) | Snapshot bytes | Final RSS (KiB) | Peak RSS (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | FP32 | 0.997 [0.971, 1.000] | 258.890 | 90.148 / 141.856 | 5,765,755 | 169,320 | 177,052 |
| 10k | RaBitQ 1-bit + FP32 reorder | 0.987 [0.921, 0.990] | 304.969 | 93.278 / 124.655 | 6,013,051 | 300,676 | 308,660 |
| 10k | RaBitQ 3+5 split | 0.983 [0.919, 0.995] | 328.196 | 121.757 / 153.896 | 2,202,806 | 301,480 | 308,896 |
| 100k | FP32 | 0.997 [0.996, 0.998] | 3,227.986 | 190.635 / 265.864 | 57,157,546 | 190,456 | 315,288 |
| 100k | RaBitQ 1-bit + FP32 reorder | 0.915 [0.901, 0.928] | 3,389.983 | 181.305 / 238.453 | 59,567,366 | 321,404 | 446,720 |
| 100k | RaBitQ 3+5 split | 0.986 [0.984, 0.989] | 3,605.873 | 221.003 / 326.071 [290.072, 2,891.121] | 22,235,021 | 322,544 | 461,840 |

At 100k, the 3+5 split snapshot is 61.1% smaller than the FP32 HGraph
snapshot, with an absolute Recall@10 median change of -0.011. Its median query
P50 is 15.9% slower, build is 11.7% slower, and load is 20.3% slower
(79.451 ms versus 66.044 ms). One 3+5 run produced a 2.891 ms P99 outlier, so
the tail result is not yet stable. The 1-bit mode is not a storage candidate
for the Lite objective in this configuration: its FP32 reorder payload makes
the snapshot 4.2% larger than FP32, while Recall@10 is lower and resident
memory is substantially higher.

The Full process retains transform models and Full VSAG dependencies, so its
RSS is evidence about integration cost rather than a projection of a future
Lite backend. These measurements also compare three Full HGraph
configurations; they must not be combined with the separate Lite graph
measurements to claim a quantization-only speed or memory change. The next
bounded step is therefore a Lite-specific 3+5 storage-layout and dependency
audit. No RaBitQ code or API should be added to the feature PR until that
design identifies an independently testable minimal subset.

Raw CSV, stdout, stderr, `/usr/bin/time -v`, snapshots, manifests, environment
metadata, and binary/library hashes are retained outside Git at:

```text
/home/ubuntu/project/vsag-lite-rabitq-reference-20260921-8447906
```

The measured benchmark executable SHA-256 is
`b50d34d3cc4e46a9b38564d930c3ae05fb38565a23407ac94d49b3bd7d7feba6`;
the Full VSAG shared library SHA-256 is
`4d245bfcde3969ecd18b338b68d7cd8b3a8f4c47d39eeba71045b9aafadb7505`.

## Lite RaBitQ 3+5 search probe

The opt-in codec probe follows the Full RaBitQ L2 split-search equations without changing the public Lite API or snapshots. It trains a deterministic four-round FHT model, encodes an 8-bit CAQ scalar code, stores the high 3 bits in filter planes and the low 5 bits in supplement planes, and records the norm/error metadata needed by the official lower-bound formula. Search scans the filter payload first and reads the supplement payload only when the lower bound can still enter the current Top-K heap.

Run its deterministic self-test or the prepared SIFT-10k smoke with:

```bash
build-lite-rabitq-codec-run/lite_rabitq_codec_probe --self-test
build-lite-rabitq-codec-run/lite_rabitq_codec_probe /path/to/scale-10000
```

The 2026-09-21 SIFT-10k smoke produced 0.994 Recall@10 for both the full 8-bit split scan and lower-bound-filtered search. The filtered result matched the full-code Top-10 exactly and reordered a mean 136.43 of 10,000 candidates (1.3643%). Its 480,000 filter bytes, 800,000 supplement bytes, and 240,000 metadata bytes describe encoded payloads only; the standalone process also retains source vectors, queries, model state, and C++ container overhead. The single run is a functional smoke, not a stable latency benchmark or evidence for a public Lite backend.

On 2026-09-22, seven fresh processes repeated the prepared SIFT-100k case at commit `a99b0e6`. Every run produced 0.985 Recall@10 for both the full 8-bit split scan and lower-bound-filtered search, with exact Top-10 agreement between those two paths. A mean 227.26 of 100,000 candidates (0.2273%) read the 5-bit supplement. Median build-and-encode time was 406.992 ms, filtered-search query P50 was 14,216.053 us, and process peak RSS was 78,456 KiB. The process retains the input vectors and per-record C++ allocations, so RSS is not the encoded payload size. Dataset page-cache state was uncontrolled, and this probe uses scalar plane decoding; the timing is experimental evidence rather than an optimized search claim. Raw CSV, `/usr/bin/time -v`, environment, hashes, and summary are retained outside Git at `/home/ubuntu/project/vsag-lite-rabitq-search-100k-20260922-a99b0e6`.

The next probe revision adds an independent little-endian `VSLRBQ01` v1 snapshot for the FHT model, split payloads, and distance metadata. Its self-test verifies byte round-trip search equality and rejects truncation, a bad magic value, non-finite or invalid metadata, excessive dimensions/counts, and trailing bytes. This remains an experiment-only format and does not alter public Lite snapshot versions v1/v2/v3.

### Contiguous RaBitQ record storage

The next probe revision replaces one `filter` and one `supplement` allocation per record with three owned contiguous arrays: all filter planes, all supplement planes, and fixed-size metadata. The scalar 8-bit code remains a temporary encoding buffer and is not retained. The self-test verifies record strides and byte-identical snapshot output after load/save. Single fresh-process checks retained the same 10k/100k Recall@10, filtered/full agreement, and reorder counts. The 100k process peak was 69,048 KiB versus the earlier seven-run median of 78,456 KiB (12.0% lower), but that comparison is directional because the new value is a single run and both processes retain source vectors. Raw output and `/usr/bin/time -v` evidence are in `/home/ubuntu/project/vsag-lite-rabitq-contiguous-validation-20260922`.

### RaBitQ filter-first graph traversal

The graph adapter reuses the existing Lite FP32 graph builder to produce a controlled topology, exports that topology into contiguous CSR offsets and neighbors, and then searches it using only the RaBitQ 3-bit filter distance. After traversal, it reads the 5-bit supplement for at most `ef_search=128` candidates and returns the full-code Top-10. This isolates traversal and reorder behavior without introducing another graph-construction algorithm. The probe now links `vsag::lite` only to build the reference topology; its RaBitQ model, records, traversal, and distance path remain experiment-local.

Single fresh-process SIFT results with degree 16 and `ef_search=128` were:

| Scale | Full RaBitQ Recall@10 | Graph Recall@10 | Mean visited | Supplement reorder | Full-scan P50 (us) | Graph P50 (us) | CSR bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 0.994 | 0.966 | 829.15 | 128 | 1,461.610 | 281.383 | 1,360,008 |
| 100k | 0.985 | 0.940 | 1,150.83 | 128 | 14,266.271 | 406.009 | 13,600,008 |

The 100k graph traversal is 35.1x faster than this probe's scalar full scan, while Recall@10 is 0.006 below the separately measured public Lite FP32 graph result of 0.946. This is a functional gate, not a stable performance comparison: each scale ran once, the scalar filter kernel is not SIMD-dispatched, and process peak RSS includes the input dataset plus temporary BruteForce and FP32 graph instances used to build/export the topology. Raw CSV, stderr, and `/usr/bin/time -v` evidence are in `/home/ubuntu/project/vsag-lite-rabitq-graph-validation-20260922-final`.

## GIST-960 RaBitQ validation

The RaBitQ codec probe accepts non-power-of-two dimensions using the same four-round transform shape as `FHTKacRotator`: alternating front/back floor-power-of-two FHT blocks, sign masks, Kac mixing, and final scaling. Its self-test covers dimensions 768 and 960, norm preservation, deterministic encoding, layout sizes, and snapshot round trips.

Prepare a documented prefix from the public ANN_GIST1M `fbin` files outside the repository. The converter requires 960-dimensional inputs, rejects truncated or non-finite payloads, and recomputes exact Top-10 independently for the 10k and 100k prefixes:

```bash
cmake -S lite -B build-lite-rabitq -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_RABITQ_LITE_PROBE=ON -DENABLE_TESTS=ON
cmake --build build-lite-rabitq --target lite_prepare_gist lite_rabitq_codec_probe
build-lite-rabitq/lite_prepare_gist base.fbin query.fbin /data/gist-prepared 100000 100
build-lite-rabitq/lite_rabitq_codec_probe /data/gist-prepared/scale-10000 32 512
build-lite-rabitq/lite_rabitq_codec_probe /data/gist-prepared/scale-100000 32 512
```

On the recorded 100-query prefixes, exhaustive full-code and lower-bound-filtered Recall@10 were both 0.997 at 10k and 100k. With `max_degree=32` and `ef_search=512`, graph Recall@10 was 0.984 at 10k and 0.949 at 100k; graph-search P50 was 5.408 ms and 9.139 ms. The 100k run encoded in 2.715 s, built the temporary FP32-derived graph in 180.568 s, used 1,408,628 KiB process peak RSS, and stored 36.0 MB filter payload, 60.0 MB supplement payload, 2.4 MB metadata, and 26.4 MB CSR topology (decimal bytes).

These are single-run experiment results, not a public-backend or SIMD performance claim. The scalar full scan keeps source vectors and a temporary FP32 graph backend resident, and the GIST subsets are prefix-specific datasets with recomputed ground truth. Full GIST1M is optional stress testing rather than a Lite acceptance requirement; high dimensionality is covered here without redefining Lite as a million-scale index.

## RaBitQ CSR snapshot experiment

The experiment-only `VSLRBQ01` format keeps its existing version 1 model/code payload unchanged. Version 2 appends the CSR graph as little-endian offset and neighbor arrays. Loading validates the offset count, zero origin, monotonic and bounded adjacency ranges, maximum degree 64, final edge count, neighbor range, self-loops, duplicate neighbors, truncation, and trailing bytes before graph search can run.

Use separate processes to save and restore a prepared dataset:

```bash
build-lite-rabitq/lite_rabitq_codec_probe --save \
  /data/sift-prepared/scale-10000 /data/sift-10k-rabitq-v2.bin 16 128
build-lite-rabitq/lite_rabitq_codec_probe --load \
  /data/sift-prepared/scale-10000 /data/sift-10k-rabitq-v2.bin 128
```

Recorded SIFT-10k and SIFT-100k snapshots were 2,880,648 and 28,800,648 bytes. Fresh-process load took 6.817 and 66.843 ms, and restored graph Recall@10 remained exactly 0.966 and 0.940. Mean visited/reordered counts also remained 829.15/128 and 1,150.83/128. The 100k loader used 32,176 KiB process peak RSS. These are single-run fresh-process measurements with uncontrolled page cache; they validate independent persistence and result stability rather than strict cold-load performance.

Version 2 is local to the opt-in probe. It does not change public Lite v1/v2/v3 snapshots or expose RaBitQ through `VectorStorage`.


### RaBitQ fixed-model CRUD experiment

The experiment now has a correctness-oriented mutable graph state. Add and Update encode with the existing trained centroid and FHT masks; the model is never retrained. Remove uses last-slot compaction, repairs every reference to the removed and moved slots, and preserves the external-ID-to-slot map. The mutation path expands persisted CSR into adjacency vectors and uses exhaustive full-code neighbor selection, so it is not a mutation-performance result.

VSLRBQ01 version 3 persists the fixed model, contiguous split records, external IDs, graph options, and CSR topology. Versions 1 and 2 remain unchanged. The self-test runs 180 deterministic mixed Add/Update/Remove operations, checks structural invariants after every operation, and requires byte-stable persistence plus identical graph-search results after a fresh restore.

This remains local to lite_rabitq_codec_probe; it does not change the public Lite API or public snapshot formats. Pairwise degree pruning currently reconstructs an approximate query from the stored 8-bit code, and SIMD filter dispatch is still open.


### RaBitQ fixed-model CRUD stability mode

The codec probe can measure the experiment-only mutable state with a fixed trained model:

    lite_rabitq_codec_probe --crud DATASET_DIR SNAPSHOT ROUNDS CRUD_OPS QUERIES MAX_DEGREE EF_SEARCH

Each operation updates one external ID, removes it, and adds the same ID and updated vector again, keeping the active count fixed. Full structural validation runs after each mutation batch rather than inside the timed operations. Each CSV row reports Update/Remove/Add P50 and P99, adjacency-to-CSR compaction, graph-search latency, exhaustive full-code and graph self-query Top-1, graph/full agreement, traversal work, snapshot save/load, snapshot bytes, RSS, and a deterministic result checksum.

The loader must reproduce every measured graph result and distance exactly. The reported state RSS is sampled before loading; round-trip RSS includes both the original and restored states. Process peak RSS also includes source vectors and the temporary FP32 graph used to create the initial reference topology. The mutation path remains a scalar correctness reference with exhaustive full-code neighbor selection, so these numbers must not be presented as a production update-throughput result.
