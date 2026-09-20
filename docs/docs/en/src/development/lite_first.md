# Independent VSAG Lite v0.1

> Phase-2 branch note (2026-09-14): the default Index::Create(dim) still uses
> exact BruteForce and its byte-identical v1 snapshot. Call BuildGraph(degree,
> ef_search) explicitly to publish a standalone single-layer approximate graph
> after a successful build. Graph Add/Update/Remove and Search use the same
> Index API; Save writes v2 including FP32 vectors, IDs, options and adjacency,
> while Load accepts both v1 and v2. Graph v2 is not a Full VSAG snapshot.
> The filtered Search overload also applies after BuildGraph: rejected external IDs
> can be traversed to keep the graph connected, but are never returned.
> No concurrent calls, checksum, quantization or mmap are provided. The
> remainder of this page documents the original v0.1 default behavior.


This experimental entry point starts with exact FP32 squared-L2 BruteForce and
selective source reuse, rather than conditionally compiling the Full library.
It currently targets Linux x86_64, C++17 and the repository compiler baseline.
The default Full build is unchanged. Run these commands from the repository root:

```sh
cmake -S lite -B build-lite-first -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build-lite-first -j2
ctest --test-dir build-lite-first --output-on-failure
cmake --install build-lite-first --prefix "$PWD/install-lite-first"
cmake -S lite/example -B build-lite-consumer -DCMAKE_PREFIX_PATH="$PWD/install-lite-first"
cmake --build build-lite-consumer -j2
./build-lite-consumer/lite_example "$PWD/new-example.lite"
```

The installed consumer only includes `vsag/lite/index.h` and links `vsag::lite`;
it does not link `libvsag.so`. It demonstrates Add, Search, Update, Remove,
Save and Load. Use a new snapshot path for each run. Expected output:

```text
added id=42 squared_l2=0
filtered id=7 squared_l2=48
updated id=42 squared_l2=0
removed id=42 remaining=1
loaded id=7 squared_l2=0
```

Unlike the Full `make` targets, `cmake -S lite` is a separate dependency-isolated
entry point. Tests use the existing pinned Catch2 v3.7.1 configuration. For an
offline build, pass `-DFETCHCONTENT_SOURCE_DIR_CATCH2=/path/to/catch2-v3.7.1`.
The existing Catch2 cases are registered with CTest as `lite_unit` and `lite_scale`.
With tests disabled, no Catch2 download is needed.
The build produces both `libvsag-lite.so` and `libvsag-lite.a`. The shared
target remains `vsag::lite`; the static archive is also installed for embedded
or offline consumers that choose to link it explicitly.

## API and ownership

`vsag::lite::Index::Create(dim)` creates a fixed-dimension index. `Add`, `Update`,
`Search`, `Save`, and static `Load` use the existing `tl::expected` / `vsag::Error`
contract without calling Full's global logger. `Remove` returns a boolean.
Inputs must contain the specified number of finite float32 values. Calls must
be externally serialized. Results own their memory; internal slots are not exposed.

- Single-record Add rejects duplicate IDs; Update rejects absent IDs.
- Remove returns false for absent IDs. A removed external ID can be inserted again.
- Add failure leaves logical records unchanged, though reserved capacity may grow.
- Search returns up to `min(k, Size())` entries sorted by squared-L2 then ascending ID.
  BruteForce checks the external ID before computing distance. Graph search may score and
  traverse rejected IDs to preserve connectivity, but never returns them. An empty filter
  accepts every ID; filtering can return fewer than k entries. Valid queries with k=0 or
  an empty index return an empty result.
- FP32 accumulation can overflow to positive infinity for extreme finite inputs;
  those distances tie and are ordered by ID. This is not arbitrary-precision L2.
- NaN/Inf inputs and dimension mismatches are rejected even for k=0.
- Memory allocation failures are translated to Error where caught; constructing
  an error itself may still allocate. No hard real-time or no-throw OOM guarantee.

## Data organization and reuse

The private implementation owns contiguous FP32 records, external IDs, and an
ID-to-slot map. BruteForce scans and graph construction/search reuse the official
`ComputeL2SqrImpl` kernels through the same runtime distance dispatcher.
On x86_64 GNU/Clang builds, a small runtime dispatcher selects AVX512, AVX2,
SSE4.1, or Generic according to CPU/OS support; dimensions below 16 retain the
Generic path. Other supported toolchains use Generic. Physical removal uses the
same last-record-to-hole principle as Full
BruteForce, but retains capacity for reuse. It does not promise immediate RSS
reduction. The minimal implementation intentionally avoids Full Factory,
InnerIndexInterface, attribute and multi-vector dependencies.

The initial BruteForce backend is still the default. `BuildGraph` constructs a separate
private FP32 graph backend and publishes it only after success; graph CRUD and filtered
search remain available. BruteForce snapshots use v1, while graph snapshots use v2.
Quantization and mmap are not part of the public Lite index.

## Snapshot v1

The binary format is separate from Full VSAG. All numeric fields are little endian:

| Offset | Field |
| --- | --- |
| 0 | 8 bytes `VSAGLT01` |
| 8 | uint64 version = 1 |
| 16 | uint64 dimension |
| 24 | uint64 record count |
| 32 | uint64 payload bytes = count * (8 + 4 * dimension) |
| 40 | uint64 representation = 1 (IEEE754 FP32, squared-L2) |
| 48 | count signed int64 IDs encoded as two's-complement bits |
| 48 + 8 * count | contiguous row-major FP32 values |

Load requires a seekable stream and consumes its remaining bytes, rejecting
truncation, trailing bytes, invalid sizes/versions, duplicate IDs and non-finite
values before publishing a new index. It bulk-reads contiguous IDs, vectors and
each graph adjacency row directly into their final owned containers. Non-little-endian
hosts convert those payload values in place. Loading does not mutate an existing
index. This is an owned-memory load, not mmap or zero-copy. There is no checksum:
valid-looking bit corruption cannot always be detected.
Save writes at the current output position; the caller owns flushing/closing,
atomic replacement, permissions and crash durability. Partial output may remain
after failure. Write failures currently use existing `READ_ERROR` because the
shared ErrorType has no separate write-error code.

## Verification and limitations

`[lite]` covers CRUD, independent-reference search and malformed snapshots.
The opt-in `[lite-scale]` case runs a 100,000 x 128 synthetic end-to-end workflow;
it is a correctness smoke test, not a realistic retrieval benchmark or a claim
of performance improvement. The separate [benchmark PR #2926](https://github.com/antgroup/vsag/pull/2926)
compares the same exact FP32 BruteForce workload at 10k and 100k x 128,
using seven alternating Full/Lite runs on one machine. The figures below
come from its generated-data comparison at commit `d729426`; shared-library
sizes are measured after `strip`, and RSS is the median process-lifetime
peak reported by `getrusage(RUSAGE_SELF)`.

| Measure | Full | Lite |
| --- | ---: | ---: |
| Stripped shared library (bytes) | 40,463,344 | 39,488 |
| 10k process peak RSS (KiB) | 164,416 | 12,112 |
| 100k process peak RSS (KiB) | 214,200 | 72,912 |
| 100k Search P50 (us) | 2,401 | 4,392 |

The RSS figures include the benchmark's data buffers, CRUD, queries and
temporary allocations; they are not index-object memory. At 100k, Lite's
search, Save and warm Load are slower than Full. See #2926 for the complete
configuration, raw-result workflow and limitations. Strict cold load and
graph-index comparisons remain work.

A follow-up seven-run alternating comparison on the same AMD EPYC host measures
the scalar Lite build against the runtime-dispatched SIMD build. Search results
kept identical IDs and order for all 640 inspected Top-10 rows; the largest
absolute distance difference was `9.54e-6`.

| Dataset | Scalar Search P50 (us) | SIMD Search P50 (us) | Scalar peak RSS (KiB) | SIMD peak RSS (KiB) |
| --- | ---: | ---: | ---: | ---: |
| 10k x 128 | 436.0 | 100.1 | 12,108 | 12,120 |
| 100k x 128 | 4,344.0 | 3,331.8 | 72,908 | 72,924 |

The stripped Lite shared library changed from 39,488 to 47,704 bytes. These are
single-machine medians for this exact workload, not general performance claims.

For source coverage add `-DENABLE_COVERAGE=ON` to a Debug build, run the tests,
and collect gcov results. Only report actually measured coverage; Full coverage
is not implied. The installed consumer above checks that the new target can be
used without linking `libvsag.so`. Full API/ABI replacement, bindings, graph
search, sparse vectors, WARP, quantization, mmap and concurrent calls are out of scope.

## FP16 graph storage

Call `BuildGraph(VectorStorage::FP16, max_degree, ef_search)` to store graph vectors as IEEE binary16 while keeping the existing FP32 input and search API. The original `BuildGraph(max_degree, ef_search)` remains FP32. `ActiveVectorStorage()` reports the active representation. FP16 graphs use snapshot version 3; versions 1 and 2 remain readable and unchanged. Loading does not require the save host ISA because the stored representation is portable little-endian binary16. Values outside the finite FP16 range are rejected when the graph is built or updated.
