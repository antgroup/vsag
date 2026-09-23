# RaBitQ Lite public backend contract proposal

Status: design for maintainer and mentor review. This document does not change
the public API, public snapshots, or PR #2904.

## Decision summary

The smallest public shape is a third graph storage choice:

```cpp
enum class VectorStorage { FP32, FP16, RABITQ };
```

`Index::BuildGraph(VectorStorage::RABITQ, max_degree, ef_search)` converts a
non-empty BruteForce index into one graph backend with an immutable trained
model, contiguous 3-bit filter records, contiguous 5-bit supplement records,
external IDs, an ID-to-slot map, and bounded adjacency lists. The public
`Add`, `Update`, `Remove`, `Search`, filtered `Search`, `Save`, and `Load`
signatures remain unchanged.

The first public version fixes the algorithm profile to squared L2, FHT enabled,
3 filter bits, 5 supplement bits, a deterministic model seed, and error rate
1.9. These values are persisted. Exposing tuning knobs can be considered later
without blocking a minimal and reproducible backend. This is one global Lite
model, not the Full fused 16-cluster storage path.

## Source evidence

| Contract area | Existing source | Constraint carried into this proposal |
| --- | --- | --- |
| Public API | `include/vsag/lite/index.h`, `Index`, `BackendKind`, `VectorStorage` | Keep the existing Index methods and `tl::expected` boundary. RaBitQ remains a graph storage choice. |
| Conversion | `src/lite/index.cpp`, `Index::BuildGraph`; `src/lite/graph_backend.cpp`, `make_graph_backend` | Build from the owned BruteForce contents and replace the active backend only after complete success. |
| Backend lifecycle | `src/lite/backend.h`, `Backend`; `src/lite/graph_backend.cpp`, `GraphBackend` | Preserve owned storage, external IDs, last-slot compaction, ID tie ordering, and no concurrent-call guarantee. |
| Filtered search | `src/lite/graph_backend.cpp`, `GraphBackend::SearchImpl` | Filtering controls accepted output; rejected nodes may still be traversed to preserve connectivity. |
| Public persistence | `src/lite/index.cpp`, `Index::Save` and `Index::Load` | Retain `VSAGLT01`, add a new representation version, validate layout before allocation, reject trailing bytes, and restore without rebuilding. |
| Official model lifecycle | `src/datacell/rabitq_split_datacell.h`, `TrainFusedCodec`, `EncodeFused`, `ExportFusedCodec`, `ImportFusedCodec` | Training requires finite non-empty data; Add/Update encode through the frozen model; the exact model must be serialized. |
| Official two-stage search | `src/datacell/rabitq_split_datacell.h`, split distance methods; `src/impl/searcher/hgraph_rabitq_searcher.cpp`, `HGraphRaBitQSearcher` | Use filter distance/lower bound during traversal and supplement-backed full distance for the retained result candidates. |
| Official SIMD | `src/simd/kernels/rabitq_compute.h`; existing probe dispatch in `lite/benchmark/rabitq_filter_ip*.cpp` | Reuse the official centered 3-bit kernel through isolated ISA translation units with runtime fallback. |
| Experiment evidence | `lite/benchmark/rabitq_lite_codec_probe.cpp` and `RABITQ_LITE_FEASIBILITY.md` | The fixed-model CRUD, persistence, graph-quality, SIMD, and isolated-memory behavior already have bounded evidence. |

## Public behavior

### Construction and model lifetime

- `Index::Create(dim)` continues to create an empty FP32 BruteForce index.
- `BuildGraph(VectorStorage::RABITQ, ...)` requires at least one record. This
  follows the official non-empty training contract and avoids freezing a model
  trained from only the first later Add.
- All source vectors must be finite. Invalid graph options keep the existing
  limits: `2 <= max_degree <= 64` and `ef_search >= max_degree`.
- Build trains one deterministic model from the complete source batch, encodes
  every source vector, constructs the graph, and swaps the new backend into the
  Index only after all steps succeed.
- The model is immutable after Build or Load. Add and Update encode with that
  model; they never retrain it.
- `ActiveBackend()` returns `GRAPH`; `ActiveVectorStorage()` returns `RABITQ`.

### Search and filtering

- Query validation remains the existing finite FP32, exact-dimension check.
- Graph traversal uses the 3-bit filter distance and its lower bound. The final
  retained candidate set is reordered with the complete 3+5 code distance.
- The returned distance is the supplement-backed approximate squared-L2
  distance. Results are ordered by distance and then ascending external ID.
- `k == 0` and an empty loaded index return an empty result; `k` is capped at
  the record count.
- `IdFilter` applies to result acceptance only. Traversal may pass through IDs
  rejected by the filter, matching the existing graph contract.
- Candidate shortage during mutation neighbor discovery uses the exhaustive
  full-code path as a correctness fallback and records a test-visible counter;
  it is not exposed as a public API.

### CRUD

- Add rejects duplicate IDs and non-finite or wrong-dimension vectors. It
  encodes first, selects neighbors, reserves required storage, and mutates the
  backend only after all fallible preparation succeeds.
- Update rejects missing IDs and invalid vectors. It encodes and selects new
  neighbors before removing old inbound links, replacing the code, and
  relinking the slot.
- Remove returns false for a missing ID. Existing last-slot-to-hole compaction
  semantics are preserved. All references to the removed slot and moved last
  slot are repaired because degree pruning can make adjacency asymmetric.
- Failed Add, Update, Build, Save, or Load leaves the logical contents
  unchanged. No concurrent operations are supported, matching the public Lite
  class documentation.

## Storage and internal interfaces

The backend owns:

- dimension and fixed RaBitQ model (centroid and deterministic FHT state);
- fixed filter/supplement strides and contiguous byte arrays;
- fixed-size per-record distance metadata;
- external IDs and ID-to-slot map;
- graph options and adjacency vectors.

Persistent raw FP32 vectors are not retained. A decoded approximate vector may
be produced internally for pairwise degree pruning, but public search and
persistence operate on split codes. This is the behavior measured by the
experiment and is necessary for the Lite memory objective.

The internal `Backend` abstraction needs a representation-specific snapshot
hook rather than a growing list of RaBitQ getters. The recommended refactor is:

```cpp
virtual tl::expected<uint64_t, Error> SnapshotPayloadSize() const = 0;
virtual tl::expected<void, Error> SaveSnapshotPayload(std::ostream&) const = 0;
```

`Index::Save` keeps ownership of the common magic/version/dimension/count/
payload/representation header and selects versions from backend kind/storage.
Existing backends implement the hook with their current byte layout so
v1/v2/v3 output remains byte-identical. RaBitQ implements only the v4 payload.
Static restore factories continue to perform representation-specific validation
before constructing a backend.

## Public snapshot version 4

Version 4 remains under the `VSAGLT01` magic and uses little-endian fields. It
is separate from the experiment-only `VSLRBQ01` files. The payload contains:

1. graph options (`max_degree`, `ef_search`);
2. fixed algorithm identifiers and parameters (L2, filter bits, supplement
   bits, error-rate bits, FHT flag, model seed/version);
3. exact model payload length and bytes;
4. external IDs;
5. fixed record strides and contiguous filter, supplement, and metadata bytes;
6. per-slot adjacency counts and neighbor slots.

Load must reject unsupported parameters, arithmetic overflow, incorrect fixed
strides, wrong payload length, non-finite model/metadata, duplicate IDs,
degrees above the stored limit, out-of-range neighbors, self-loops, duplicate
neighbors, truncated blocks, and trailing bytes. It restores the encoded graph
directly and never retrains or rebuilds topology.

Save/load byte stability is required for an unchanged index. Existing v1/v2/v3
fixtures must retain their exact hashes after the internal snapshot-hook
refactor.

## Error translation

Public methods keep `tl::expected<..., Error>`. Validation failures use
`INVALID_ARGUMENT`; malformed snapshots use `INVALID_BINARY`; allocation and
capacity failures use `NO_ENOUGH_MEMORY`; stream failures retain the existing
`READ_ERROR` convention. Implementation boundaries translate
`std::bad_alloc`, `std::length_error`, `std::invalid_argument`, and stream
exceptions consistently. No exception from encoding, allocation, or IO crosses
the public API.

## Minimal implementation sequence

1. Add the internal model/record container and scalar differential tests under
   `src/lite`, reusing official pack and formula helpers.
2. Add `RaBitQGraphBackend` with Build, Search, filtered Search, and the runtime
   Generic/AVX2/AVX512 filter dispatcher.
3. Add fixed-model CRUD and the existing mutation boundary regressions.
4. Add the internal snapshot hook, prove v1/v2/v3 byte compatibility, then add
   strict v4 save/load.
5. Expose `VectorStorage::RABITQ` only after all preceding stages pass.

Each step should remain independently reviewable. The implementation should
reuse validated experiment logic selectively rather than copy the monolithic
probe class.

## Acceptance gates

- clang-format-15, clang-tidy-15, Release CTest, and ASan+UBSan CTest pass.
- Generic, AVX2, and AVX512 results match the scalar distance oracle within the
  established tolerance; unsupported CPUs use Generic.
- Deterministic model/code tests cover dimensions 1/7/8/9/31/128/129/768/960,
  constant and zero-norm vectors, and non-finite rejection.
- CRUD tests cover empty/one/two-element states, duplicate/missing IDs,
  last-slot and non-last-slot removal, asymmetric inbound links, filtered
  search, and save/load after mutation.
- Corruption tests cover every version 4 block and retain all v1/v2/v3 fixtures.
- SIFT-10k/100k and GIST-960 10k/100k reproduce the probe's quality within an
  agreed tolerance, with seven fresh processes for formal latency/RSS claims.
- The public backend links only `libvsag-lite`; Full `libvsag` is not a runtime
  dependency.
- Snapshot size and isolated steady RSS are compared against FP16 and the
  exact probe commit. Promotion requires a mentor/maintainer-approved Recall
  floor and latency budget rather than a new threshold chosen during coding.

## Explicit exclusions for the first public version

- cosine/IP metrics, PCA/MRQ, raw-vector reorder, disk supplement IO, mmap,
  fused datacell linkage, concurrent calls, and online model retraining;
- ARM-specific SIMD and batch-four traversal optimization;
- an inbound-edge index or flat-adjacency rewrite without a measured workload
  demonstrating that Remove or fragmentation is the dominant cost;
- changes to the Full VSAG API or snapshot formats.

## Review decisions requested

Maintainers and the mentor need to confirm only these public-boundary choices
before implementation is promoted:

1. accept `VectorStorage::RABITQ` with a fixed 3+5 profile for the first Lite
   release, rather than adding a new options object;
2. accept rejection of `BuildGraph(RABITQ)` on an empty source;
3. accept version 4 under `VSAGLT01` with exact model/code/topology persistence;
4. provide the SIFT/GIST Recall floor and latency/RSS budget used for promotion.
