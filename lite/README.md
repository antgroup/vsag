# VSAG Lite (phase-2 experimental graph branch)

The standalone library defaults to exact FP32 squared-L2 BruteForce and does not
link Full VSAG. On this phase-2 branch, callers may explicitly transition an
existing Lite Index with BuildGraph(max_degree, ef_search). Graph search is
approximate. BuildGraph builds a replacement before publishing it; a failure
leaves the BruteForce index unchanged. Add, Update and Remove remain available
after transition. Calls must be externally serialized.

BruteForce Save/Load keeps the byte-identical v1 snapshot. Graph Save writes
v2 with the FP32 records, graph options and adjacency; Load detects either
version and restores the corresponding backend. The format is not compatible
with Full VSAG. It has no checksum or crash-safe file replacement. The graph
is a standalone single-layer candidate, not Full HGraph or LazyHGraph.
Quantization and mmap are not included.

The supported graph options are max_degree 2–64 and ef_search at least
max_degree. Callers can check ActiveBackend() before and after BuildGraph().
For example, after creating an Index and adding vectors:

    auto built = index->BuildGraph(16, 128);
    if (!built) { /* handle built.error(); the flat index is unchanged */ }
    auto results = index->Search(query.data(), index->Dim(), 10);

This candidate trades faster approximate queries for graph construction time,
larger snapshots and a temporary flat-plus-graph memory peak during transition.

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
search P50 444 us, build 43.5 s and a 65.6 MB graph snapshot. These are
exploratory numbers on one host, not statistical performance claims. Raw
results and environment are kept outside Git. The default v0.1 guide remains
in the English and Chinese links below.

- [English v0.1 guide](../docs/docs/en/src/development/lite_first.md)
- [中文 v0.1 说明](../docs/docs/zh/src/development/lite_first.md)

The installed consumer and v0.1 measurements are documented in the linked English and Chinese guides.
