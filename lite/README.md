# VSAG Lite

The standalone library defaults to exact FP32 squared-L2 BruteForce and does not
link Full VSAG. Callers may explicitly transition an
existing Lite Index with BuildGraph(max_degree, ef_search). Graph search is
approximate. BuildGraph builds a replacement before publishing it; a failure
leaves the BruteForce index unchanged. Add, Update and Remove remain available
after transition. Calls must be externally serialized.

BruteForce Save/Load keeps the byte-identical v1 snapshot. FP32 Graph Save writes
v2 and FP16 Graph Save writes v3 with records, graph options and adjacency; Load
detects all three versions and restores the corresponding backend. The format is not compatible
with Full VSAG. It has no checksum or crash-safe file replacement. The graph
is a standalone single-layer candidate, not Full HGraph or LazyHGraph.
BruteForce and FP32 graph use the runtime-selected FP32 distance kernel on supported
x86_64 builds; FP16 graph uses the corresponding half-precision dispatcher. Dimensions
below 16 and other platforms use Generic.
SQ8 and mmap are not included; FP16 graph storage is available only when explicitly selected.

Search(query, dim, k, IdFilter) accepts a callback on external IDs; returning true allows
an ID in the result. The graph still visits and scores rejected nodes for connectivity,
so the filtered result may contain fewer than k neighbors. The three-argument Search
retains its unfiltered behavior.

The supported graph options are max_degree 2–64 and ef_search at least
max_degree. Callers can check ActiveBackend() before and after BuildGraph().
For example, after creating an Index and adding vectors:

    auto built = index->BuildGraph(16, 128);
    if (!built) { /* handle built.error(); the flat index is unchanged */ }
    auto results = index->Search(query.data(), index->Dim(), 10);

This candidate trades faster approximate queries for graph construction time,
larger snapshots and a temporary flat-plus-graph memory peak during transition.

## Directory layout

- `include/vsag/lite/index.h`: public Lite API and external-ID search filter.
- `src/lite/index.cpp`: API dispatch and v1/v2/v3 snapshot handling.
- `src/lite/bruteforce_backend.cpp`, `src/lite/graph_backend.cpp`, and
  `src/lite/fp32_distance*.cpp`: exact and graph implementations plus FP32 distance dispatch.
- `src/lite/index_test.cpp` and `src/lite/graph_backend_test.cpp`: functional,
  CRUD, filtering, and snapshot tests.
- `lite/CMakeLists.txt` and `lite/example/`: standalone build, install, and consumer example.
- `docs/docs/{en,zh}/src/development/lite_first.md`: user-facing guides.

Benchmark runners and candidate probes are tracked with the experiment work; the public
Lite API supports FP32 and explicitly selected FP16 graph storage, while SQ8 remains experimental.

## Graph snapshot v2

All integers are little endian. The 48-byte header keeps the VSAGLT01 magic;
version and representation are 2, followed by dimension, active record count
and payload length. The payload contains max_degree and ef_search (uint64 each),
then all signed int64 IDs, all row-major FP32 values, and one adjacency record
per slot: uint64 link count followed by that many uint64 slot numbers. Thus the
payload length is 16 + count * (16 + 4 * dim) + 8 * total_links. The slot
numbers refer to the saved compact record order. Load checks the full length,
degree bounds, duplicate IDs, finite vectors, and adjacency bounds, self-links
and duplicates before publishing the new index. There is no checksum, so a
bit change that still describes a valid graph is not detectable. The caller
owns flushing, atomic file replacement and crash durability.

From the repository root, configure tests and run the regular suite:

    cmake -S lite -B build-lite-graph -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
    cmake --build build-lite-graph -j2
    ctest --test-dir build-lite-graph --output-on-failure -j2

The hidden SIFT probes require the prepared independent-query dataset from
the experiment workflow and write the graph snapshot outside the repository:

    VSAG_SIFT_DIR=/path/to/scale-100000 VSAG_GRAPH_SNAPSHOT=/path/to/graph.snapshot build-lite-graph/lite_graph_tests '[lite-sift]'
    VSAG_SIFT_DIR=/path/to/scale-100000 VSAG_GRAPH_SNAPSHOT=/path/to/graph.snapshot build-lite-graph/lite_graph_tests '[lite-sift-load]'

The second command loads the snapshot in a fresh process. The 2026-09-14
single-run 100k SIFT probe at degree 16 / ef 128 measured Recall@10 0.946,
search P50 442 us, build 35.1 s and a 65.6 MB graph snapshot after
caching graph-pruning distances. Three alternating 10k runs with the same
benchmark executable reduced median build from 2.50 s to 1.70 s while
producing byte-identical snapshots. These are same-host exploratory numbers,
not general performance guarantees. Raw
results and environment are kept outside Git. The default v0.1 guide remains
in the English and Chinese links below.

## SQ8 and FP16 candidate evaluation

The optional [quantization probe](benchmark/README.md) measures classic
per-dimension SQ8 and IEEE FP16 encoding on independent SIFT queries.
It is a selection experiment, not a quantized Lite backend: Graph, public
API and v1/v2 snapshots remain FP32. The code-byte count must not be reported
as current index RSS or snapshot size.

- [English v0.1 guide](../docs/docs/en/src/development/lite_first.md)
- [中文 v0.1 说明](../docs/docs/zh/src/development/lite_first.md)

The installed consumer and v0.1 measurements are documented in the linked English and Chinese guides.

## FP16 graph storage

Call `BuildGraph(VectorStorage::FP16, max_degree, ef_search)` to store graph vectors as IEEE binary16 while keeping the existing FP32 input and search API. The original `BuildGraph(max_degree, ef_search)` remains FP32. `ActiveVectorStorage()` reports the active representation. FP16 graphs use snapshot version 3; versions 1 and 2 remain readable and unchanged. Loading does not require the save host ISA because the stored representation is portable little-endian binary16. The loader bulk-reads v3 vectors directly into final FP16 storage, validates finite binary16 exponent fields, and converts byte order in place on non-little-endian hosts. Values outside the finite FP16 range are rejected when the graph is built or updated.
