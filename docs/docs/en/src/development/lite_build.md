# VSAG Lite

VSAG Lite is a reduced-footprint build for dense-vector search. It preserves the
public C++ API, so applications using retained indexes can switch between Full
and Lite without source changes.

## Feature set

| Area | Lite status |
| --- | --- |
| HGraph, LazyHGraph, BruteForce and WARP | Included |
| Add, build, search, update and remove | Included |
| Serialize and deserialize | Included |
| Mark-deletion persistence | Included |
| IVF, Pyramid, SINDI, SINDI v2 and SIMQ | Excluded |
| SINDI and sparse-vector-only DataCells | Excluded |

Creating an excluded index through `Factory::CreateIndex` returns an error.

## Build and test

```bash
make lite
make test-lite
```

Equivalent CMake configuration:

```bash
cmake -S . -B build-lite -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_LITE=ON -DENABLE_TESTS=ON
cmake --build build-lite --parallel
./build-lite/tests/unittests -d yes --allow-running-no-tests
./build-lite/tests/functests -d yes --allow-running-no-tests
```

The Lite configuration uses the repository's existing Catch2 test binaries.
Tests that depend on excluded index implementations are omitted at build time.
`make test-lite` runs the focused `[lite]` cases after compiling the retained
test suite; run the binaries directly, as above, for the broader regression set.

## Compatibility and persistence

Lite retains the same headers, factory, datasets, result objects and BinarySet
API as Full. Marked removals are stored in an optional HGraph label-metadata
block. Readers remain compatible with older files that lack the block, although
old deletion information that was never stored cannot be recovered.

Writers omit the optional block when an index has no marked deletions, preserving
the legacy byte layout in that case. A file containing persisted deletion state
requires a reader that understands the extension; older readers cannot load that
file safely.

HGraph stores dense codes, graph adjacency, labels, visited-list state and
optional precise-reorder data. LazyHGraph starts flat and transitions to HGraph
at its configured threshold. Lite removes unused factories and dedicated
DataCells; it does not change retained indexes' runtime layout or accuracy.

## Benchmarking

Use the repository's `tools/eval` framework with identical Full and Lite YAML
configurations and standard ANN datasets. Measure stripped shared libraries and
linked deployment binaries rather than unstripped static archives. Preserve raw
results and report medians and dispersion across repeated runs. Sparse/SINDI
deployments require Full.
