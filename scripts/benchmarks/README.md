# Streaming block reader microbenchmark

This Linux-only diagnostic compares the legacy materializing TLV reader and the forward-only reader used by HGraph. It does **not** measure complete HGraph loading or search performance.

Build the library first, then compile against the same library build:

```sh
g++ -std=c++17 -O2 -Iinclude -Isrc \
  -Ibuild/_deps/fmt-src/include -Ibuild/_deps/nlohmann_json-src/include \
  -Ibuild/_deps/tsl-src/include scripts/benchmarks/streaming_block_reader.cpp \
  -Lbuild/src -Wl,-rpath,"$PWD/build/src" -lvsag -o /tmp/streaming-block-benchmark
for mode in legacy forward legacy forward legacy forward; do
  /tmp/streaming-block-benchmark "$mode"
done
```

The input is a generated 129 MiB zero-filled payload, above the configured 128 MiB block limit. The callback consumes every byte through an 8 KiB buffer. Each sample runs in a fresh process; input generation and expected-checksum calculation happen before timing. `getrusage` measures the increase in peak RSS (KiB) and output blocks (512-byte units). A zero RSS delta means no new process high-water mark was observed, not zero allocation. Output blocks measure OS-accounted storage writes, not logical write calls; results depend on the temporary-filesystem backend.

## Local observation, 2026-09-10

Base commit `b144589fc1` with the HGraph forward-reader worktree changes; Linux x86-64, AMD EPYC 9T24, Debug library, no coverage instrumentation. Other test/build work ran concurrently, so timings are indicative only.

| Adapter | Seconds, three runs | Peak RSS increase | Output bytes per run |
| --- | --- | --- | --- |
| Legacy | 4.51888, 4.53509, 4.53647 | 133816–133876 KiB | 135266304 |
| Forward | 4.44432, 4.42120, 4.43077 | 0 KiB observed | 0 |

Both paths read exactly 135266304 bytes from the generated source. The evidence supports removal of payload staging and temporary writes; it is not an end-to-end speedup claim. The legacy helper remains because other indexes still depend on it.

## Real HGraph streaming-load comparison

`hgraph_streaming_load.cpp` creates a streaming artifact with 1024 vectors (dimension 4) and 131073 bytes of extra information per vector. `legacy_streaming_block_shim.cpp` is a Linux `LD_PRELOAD` adapter which routes calls to the new helper through the still-existing legacy helper. Thus both modes use identical component parsers and the same artifact, isolating the effect of block materialization. This is not a comparison against a complete historical checkout. The shim logs payload lengths so its activation is observable; it is only for this benchmark, never for production.

```sh
g++ -std=c++17 -O2 -Iinclude scripts/benchmarks/hgraph_streaming_load.cpp \
  -Lbuild/src -Wl,-rpath,"$PWD/build/src" -lvsag -o /tmp/hgraph-streaming-benchmark
g++ -std=c++17 -O2 -fPIC -shared -Iinclude -Isrc \
  -Ibuild/_deps/fmt-src/include -Ibuild/_deps/nlohmann_json-src/include \
  -Ibuild/_deps/tsl-src/include scripts/benchmarks/legacy_streaming_block_shim.cpp \
  -Lbuild/src -lvsag -o /tmp/legacy-streaming-block-shim.so
benchmark_dir=$(mktemp -d /tmp/vsag-streaming-benchmark.XXXXXX)
export GCOV_PREFIX="$benchmark_dir/profiles"
export OPENBLAS_NUM_THREADS=1
/tmp/hgraph-streaming-benchmark create "$benchmark_dir/hgraph.stream"
for iteration in 1 2 3; do
  LD_PRELOAD=/tmp/legacy-streaming-block-shim.so \
    /tmp/hgraph-streaming-benchmark load "$benchmark_dir/hgraph.stream"
  /tmp/hgraph-streaming-benchmark load "$benchmark_dir/hgraph.stream"
done
```

The local artifact was 134329419 bytes; its extra-info block was 134218776 bytes, exceeding the 134217728-byte limit. Every successful load restored 1024 elements. With the Debug **coverage-instrumented** library (unlike the microbenchmark above), observed old-helper loads took 4.33437, 4.33281 and 4.33939 seconds, increased peak RSS by 663360–663364 KiB and wrote 134221824 bytes each. Forward loads took 4.25345, 4.26048 and 4.24500 seconds, increased peak RSS by 532456–532460 KiB and wrote zero bytes. RSS here includes the restored index and ordinary component buffers; the approximately 128 MiB difference isolates the removed staging buffer. Profiling output is redirected outside the unit-test coverage tree and is written after the measured interval. These warm-cache, concurrent-work measurements demonstrate resource savings, not a production latency guarantee.
