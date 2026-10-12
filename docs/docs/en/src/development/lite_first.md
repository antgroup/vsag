# Standalone VSAG Lite

This is the current supported contract for PR #2904, updated 2026-10-12.
Historical experiment descriptions do not override this contract.

## Build, configuration and installation

```bash
cmake -S lite -B build-lite -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build-lite -j2
ctest --test-dir build-lite --output-on-failure
cmake --install build-lite --prefix "$PWD/install-lite"
```

The default is FP32 BruteForce. `ENABLE_RABITQ_LITE_BACKEND` defaults OFF;
explicitly add `-DENABLE_RABITQ_LITE_BACKEND=ON` to compile RaBitQ8 support.
`ENABLE_COVERAGE=ON` instruments library and test translation units.
Shared and static `libvsag-lite` libraries are built; installed CMake consumers
use `find_package(vsag-lite CONFIG REQUIRED)` and `vsag::lite`.
See `lite/example` for installed shared/static consumers and `rabitq_example.cpp`
for the optional backend lifecycle. No Full `libvsag.so` is linked.

## Functions mapped to implementation and tests

| Supported function | Public entry / implementation | Regression evidence |
| --- | --- | --- |
| Empty index, owned FP32 vectors and external int64 IDs | `Index::Create`, `Add`; `src/lite/index.cpp`, `bruteforce_backend.cpp` | `index_test.cpp`: validation, duplicate IDs, exact search, scale |
| Explicit graph conversion | `BuildGraph(storage, degree, ef)`; `graph_backend.cpp`, `rabitq_backend.cpp` | Floating graph and RaBitQ API suites; failure retains prior backend |
| Single-vector insert/update/physical delete | `Add`, `Update`, `Remove` | All three storage modes; moved-slot references, incoming-edge invariants and mutable persistence |
| L2 search, external-ID callback filter, per-call budget | `Search`, `SearchWithOptions`; graph search helpers | Ties/nonmonotonic IDs, empty/filter/large-boundary/reference comparisons |
| Save/Load, corruption and stream error handling | `index.cpp`, `rabitq_snapshot.h` | Truncation, duplicate IDs, invalid topology/metadata, short writes and round trips |
| Allocation-failure rollback | Floating graph and RaBitQ transaction journals | `graph_allocation_test.cpp`, `rabitq_backend_test.cpp`; logical state restored |
| Native SIMD reuse with generic fallback | `fp32_distance*`, `fp16_distance*`, `rabitq_filter_ip*` | ISA/scalar differential tests, short dimensions and tails |
| Bounded query workspace and snapshot staging | `query_heap.h`, `query_visited.h`, block/stream I/O | Heap reference, bitmap boundary, endian/block/byte-equivalence fixtures |

BuildGraph uses degree 2–64 and configured ef at least degree. SearchOptions zero
uses the stored budget; per-call overrides do not change build/CRUD/persistence
configuration. Graph search is approximate; a filter may produce fewer than k
results. Rejected IDs can still be traversed for connectivity. RaBitQ returned
distances are quantized L2 estimates, not exact FP32 distances.

## Architecture, retained and removed components

Public `vsag::lite::Index` owns a private backend. The default dense BruteForce
supports exact squared-L2 search. An explicit BuildGraph constructs a replacement
single-layer graph before publication; it is not native HGraph/LazyHGraph and does
not automatically transition at a size threshold. The FP32-input API can select
FP16 graph storage or opt-in RaBitQ8. No SQ8, mmap, sparse, bindings, general Full
factory or automatic backend selection is provided.

Floating storage uses contiguous vector/ID arrays, an external-ID-to-slot map,
outgoing and incoming adjacency. Delete compacts the last slot into the hole and
repairs references. Capacity may remain after deletion; no guarantee of returning
all freed bytes to the OS is made. Graph journals back up changed rows/records;
successful growth can retain extra capacity after a failed operation.
FP16 uses borrowed stored half rows for direct Save rather than decode/re-encode.

RaBitQ stores a fixed trained centroid/rotation model, 3-bit filter and 5-bit
supplement codes plus per-record metadata, IDs and graph adjacency. It does not
retain an exact FP32 vector matrix. Current public construction/loading enables
32-bit incoming adjacency and `CACHED` protection with safe link pruning;
the additional Add/Remove/displaced-target guard flags remain false.
Update protection is enabled by CACHED. This preserves an existing target's last
incoming edge when another edge can be safely removed; the farthest fallback
remains if none can. It is not a global connectivity or recall guarantee.
See [RaBitQ design](lite_rabitq_design.md) for the actual configuration and model limits.

## Compatibility, persistence and failure boundaries

This is a separate Lite API, not an ABI-compatible replacement for Full Index.
One instance supports externally serialized calls only, not concurrent reads or
read/write mixing. Inputs are borrowed during calls; results and stored state are owned.

FP32 BruteForce uses `VSAGLT01` v1, FP32 graph v2, FP16 graph v3. Optional RaBitQ
uses independent `VSAGLQ01` v1; existing floating formats are unchanged. All are
little-endian and not compatible with native Full snapshots. RaBitQ Load restores
model/codes/topology without training or reconstructing exact input vectors.
Load requires a seekable stream and validates remaining lengths and content.
Save has no checksum, fsync or atomic-file replacement guarantee. A failed Save
may leave a partial output; callers must arrange temporary-file replacement.

Add/Update/Search/Save/Load use the existing `tl::expected`/Error contract.
Remove returns bool: false means missing ID or failed graph allocation, so the
cause cannot be distinguished. Failed graph transactions preserve logical state,
not allocator capacity. Complete identical stored representations preserve graph
topology on Update; equal RaBitQ codes do not mean raw inputs are identical.

## Project output acceptance

| Required output | Present evidence | Remaining acceptance |
| --- | --- | --- |
| Reviewable standalone implementation with CRUD/persistence/config/tests | PR #2904, sources and examples above; ON/OFF regression suites | Community CI and maintainer review, not automatic merge |
| Architecture: modules/layout/memory/API boundaries | This guide and bilingual RaBitQ design | Keep synchronized when behavior changes |
| Reproducible multi-dataset benchmarks and raw results | [Versioned performance evidence](../../../../../lite/PERFORMANCE_EVIDENCE.md), pinned scripts/configs/raw archives for SIFT/GIST/Cohere 10k/100k | One unified latest-source CPU/peak/CRUD/quality matrix, cold I/O and hours-long interleaved changed-vector stability |
| At least one improvement against actual Full with stable retrieval | Historical warm load/RSS/ELF-closure results with explicit Full provenance | Quality-matched native Full comparison, complete deployment package; GIST maintenance quality remains incomplete |

`.github/workflows/lite.yml` defines default-OFF, RaBitQ-ON, ASan/UBSan and
coverage jobs, all-source formatting, configured-TU/header tidy, installation
and consumer checks. Coverage collector gates the production Lite line union at
90%, not Full or branch coverage. Local passes do not imply online CI has passed.

Old experiments and negative candidates remain in the [pinned experimental repository](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark).
Reported best values from different measured versions must not be combined into
a fictitious latest-version matrix. Functional support is not complete quality,
performance or long-term stability acceptance. The project is not declared finished.
