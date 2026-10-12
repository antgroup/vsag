# Independent VSAG Lite v0.1

> Phase-2 branch note (2026-09-14): the default Index::Create(dim) still uses
> exact BruteForce and its byte-identical v1 snapshot. Call BuildGraph(degree,
> ef_search) explicitly to publish a standalone single-layer approximate graph
> after a successful build. Graph Add/Update/Remove and Search use the same
> Index API; Save writes v2 for FP32 graphs and v3 for FP16 graphs, including
> vectors, IDs, options and adjacency. Load accepts v1, v2 and v3. Graph snapshots
> are not compatible with Full VSAG.
> The filtered Search overload also applies after BuildGraph: rejected external IDs
> can be traversed to keep the graph connected, but are never returned.
> No concurrent calls, checksum, SQ8 or mmap are provided. The
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
graph query id=7 budget=64
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
- Graph Update returns without changing adjacency or snapshots when validated FP32 bytes
  or encoded FP16 bits match the stored representation. FP16 range checks still run first;
  FP32 positive and negative zero have distinct byte representations. Same-value Update is
  not a graph repair request. BuildGraph only converts BruteForce to Graph; it does not
  rebuild an index that is already a graph.
- Remove returns false for absent IDs or failed graph repair allocation without changing contents. A removed external ID can be inserted again.
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
private graph backend and publishes it only after success; graph CRUD and filtered
search remain available. BruteForce snapshots use v1, FP32 graphs use v2, and explicitly
selected FP16 graphs use v3. SQ8 and mmap are not part of the public Lite index.

## Architecture design and compatibility

The standalone boundary is intentionally narrow:

```text
application
    |
    v
vsag::lite::Index             public ownership and error contract
    |
    v
detail::Backend               private CRUD/Search/storage abstraction
    |-------------------|
    v                   v
BruteForceBackend       GraphBackend
FP32 row-major          FP32 or FP16 row-major + adjacency rows
    |                   |
    +---------+---------+
              v
    runtime L2 dispatcher     Generic/SSE/AVX2/AVX512 as available
              |
              v
    versioned owned snapshot  v1 flat / v2 FP32 graph / v3 FP16 graph
```

`lite/CMakeLists.txt` builds only this source closure into shared and static
`vsag-lite` libraries. Lite reuses the repository's `Error`/`tl::expected`
contract and official L2 implementation traits, but does not link the Full
Factory, HGraph, thread pool, allocator orchestration, logging runtime, IO
framework, attribute system, sparse or multi-vector indexes. This keeps the
installed surface to `vsag/lite/index.h` plus the header-only error contract.

The index choices follow the small-data deployment goal:

- BruteForce is the default because it is exact, has no construction phase,
  and keeps the smallest state for small collections.
- The single-layer graph is explicit because it trades construction time,
  snapshot bytes and approximate recall for faster queries. `BuildGraph`
  creates the complete replacement first and publishes it only on success.
- FP16 is an explicit graph storage choice. It reduces vector and snapshot
  bytes, while inputs and results stay FP32. It may change distances and recall,
  so FP32 remains the graph default.

BruteForce stores vectors in one row-major `std::vector<float>`, IDs in a
parallel vector and the external-ID mapping in an `unordered_map`. Graph adds
one bounded outgoing-adjacency vector per slot; FP16 replaces the vector payload
with row-major binary16 words. Capacity grows geometrically. Remove physically
moves the last record into the hole and updates slot references, while retaining
allocated capacity for reuse; it does not promise an RSS drop. Pruning and old
snapshots may contain asymmetric adjacency, so removal scans stored edges rather
than assuming every incoming edge appears in the removed node's outgoing row.

Graph construction temporarily owns both the flat source and the replacement,
so peak memory can exceed either steady state. Search owns its visited bitmap and
candidate queues per call. FP16 decode scratch belongs to the caller inside the
implementation and is reused across serialization iterations. There is no shared
mutable decode buffer, background compaction or internal synchronization.

The compatibility boundary is behavioral rather than ABI compatibility:

| Concern | Lite behavior | Difference from Full |
| --- | --- | --- |
| Errors | `tl::expected<..., vsag::Error>` where applicable | Reuses the contract without the Full runtime |
| IDs and vectors | int64 external IDs, fixed dimension, FP32 call inputs | Single-vector dense squared-L2 only |
| CRUD | Add, Update, physical Remove and Search | No delete markers, attributes or multi-vector operations |
| Filtering | Callback on external IDs | Separate minimal callback type |
| Results | Owned `std::vector<Neighbor>` | Not the Full `Dataset` result ABI |
| Persistence | Seekable C++ streams and Lite v1/v2/v3 | Not Full serialization compatible |
| Concurrency | Calls must be externally serialized | No internal thread pool or concurrent-call guarantee |

Migrating code must therefore adapt construction, result handling and
persistence even when CRUD names and error handling are familiar. Full API/ABI
replacement is not claimed.

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
each graph adjacency row directly into their final owned containers. Version 3 binary16
vectors are also read directly into final FP16 storage and checked for finite exponent
fields without an FP32 staging copy. Non-little-endian hosts convert payload values in
place. Loading does not mutate an existing
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
used without linking `libvsag.so`. Full API/ABI replacement, bindings, sparse vectors,
WARP, SQ8, mmap and concurrent calls are out of scope.

## FP16 graph storage

Call `BuildGraph(VectorStorage::FP16, max_degree, ef_search)` to store graph vectors as IEEE binary16 while keeping the existing FP32 input and search API. The original `BuildGraph(max_degree, ef_search)` remains FP32. `ActiveVectorStorage()` reports the active representation. FP16 graphs use snapshot version 3; versions 1 and 2 remain readable and unchanged. Loading does not require the save host ISA because the stored representation is portable little-endian binary16. Version 3 values are bulk-read directly into final FP16 storage, validated by their binary16 exponent fields, and byte-swapped in place when required. Values outside the finite FP16 range are rejected when the graph is built or updated.


## Per-query graph budget

`SearchWithOptions(query, dim, k, SearchOptions{ef_search}, filter)` supplies a
budget for one graph query. Zero uses the configured default; positive values
are clamped to the live count and raised to the requested result count when
necessary. BruteForce ignores this option and remains exact. The optional
filter has the same external-ID semantics as Search. Existing Search calls,
including empty filters, are unchanged.

```cpp
vsag::lite::SearchOptions options{512};
auto result = index->SearchWithOptions(query, dim, 10, options);
```

The override does not change BuildGraph/Add/Update/Remove budgets or Save output.
Snapshots retain their configured default. The consumer example builds with
budget8 then queries with budget64. Calls still require external serialization;
this feature does not add a concurrent-call guarantee. Bigger query budgets
spend more work and may improve recall, but do not guarantee a quality target.

## Opt-in 8-bit RaBitQ candidate

Build the standalone target with `-DENABLE_RABITQ_LITE_BACKEND=ON` to enable
`BuildGraph(VectorStorage::RABITQ8, degree, ef_search)`. The option is OFF by
default. Disabled builds return `UNSUPPORTED_INDEX_OPERATION` for this selection;
the original flat index remains intact. This candidate requires a nonempty flat
training set, dimension at most 1,048,576, at most 1,000,000 live records, degree
2–64 and ef at least degree. The deterministic training seed is currently 47.

Inputs remain finite FP32 L2 vectors. The backend owns a fixed trained centroid,
four FHT/Kac sign-mask rounds and 3+5 split code planes. Add/Update use the fixed
model without retraining. Search traverses 3-bit filter estimates and ranks
candidates with 8-bit estimates. These are approximate distances, not exact
original-vector squared L2. Filtering accepts external IDs when the callback
returns true; rejected nodes remain traversable. Per-call budgets do not change
stored defaults. Filtered and unfiltered approximate paths need not return
identical rows even with an allow-all callback.

Save writes independent little-endian `VSAGLQ01` version 1 model/code/ID/graph
payloads, never decoded-and-reencoded vectors. Load preserves query results and
remains mutable, including after removing all records. Only enabled builds load
this candidate format. Existing `VSAGLT01` v1/v2/v3 retain their original paths;
experimental `VSLRBQ01` snapshots are not accepted as this candidate format.
Streams must be seekable. Header sizes are checked against remaining bytes before
large allocations. No checksum, atomic file replacement or concurrent calls are
provided. Extreme finite values that overflow model/transform arithmetic fail
without replacing the existing index.

Duplicate Add and absent Update/Remove IDs are checked before transaction copying.
Update encodes the requested vector and compares both planes plus all six metadata
fields against the stored record. Identical complete encodings return without
changing topology or snapshot bytes; plane identity alone is not sufficient.
This still performs encoding and can fail on allocation or internal arithmetic.
For a real change, the prepared normalized query and full code are moved into
the transaction; the mutation path does not normalize or encode that input again.

**Default public CRUD uses local rollback transactions.** Remove backs up the
hole/last code and ID, retains the removed map node, and journals affected rows.
On failure it restores lengths within retained capacity and reinserts the node
without exceeding the map's old load; no rollback allocation is required.
Public Add prepares codes
and neighbors, reserves geometric container growth, then journals touched rows.
Failure erases the inserted ID, truncates appended lengths and restores old rows;
capacity/bucket growth can remain, but logical contents are preserved. Normal
buffer expansion can still copy existing data. Public Update instead
journals the replaced code and each adjacency row before its first write. Failure
restores codes and swaps saved rows without allocating; IDs/model/container sizes
are unchanged by Update. It still scans directed adjacency for inbound references
and can journal many rows in a high-indegree case. The experimental inbound-index
state uses the full-copy Add/Update/Remove fallback. This reduces mutation copying but does not make
all CRUD constant-time or remove temporary mutation memory. This is a correctness candidate, not a validated
performance improvement. Remove's bool result cannot distinguish a missing ID
from allocation failure; either failure retains the state. Error construction
itself can allocate, as with the existing Lite error contract. Build retains
temporary FP32 vectors and a temporary FP32 graph; peak build RSS is not steady
quantized RSS. Three-distribution quality/performance and long-CRUD acceptance
remain pending. Do not infer those results from older standalone probe numbers.

Use the opt-in `lite_rabitq_example` build target for the public API lifecycle.

The integrated 10k SIFT/GIST/Cohere pilot is recorded in `lite/benchmark/results/integrated-storage-pilot-20261009`. It establishes a snapshot-size reduction at the measured settings, but queries and whole-process peak memory are worse than FP32. Historical queries and short churn do not replace100k, fresh-process isolated RSS or long changed-vector acceptance. Read the experiment report for raw evidence and measured-version boundaries.

A fresh-process10k loaded-only study is recorded at `lite/benchmark/results/integrated-load-final-20261009`: RaBitQ loaded RSS is about41–66% below FP32 on these three distributions, with no original base matrix resident. Load/first-query regressions remain; these are warm-uncontrolled RAM measurements, not cold I/O or100k/long-CRUD acceptance. The report separates current-image VmHWM from the launcher-affected getrusage high-water metric.

The integrated100k pilot records lower loaded RaBitQ RSS but quality failures at stored degree16/ef128. A separate observed-query grid meets the study floors at SIFT ef512 and GIST/Cohere ef2048 for RaBitQ, with slower queries than floating modes. See `lite/benchmark/results/integrated-100k-pilot-20261009` and `integrated-100k-budget-grid-20261009`. This selection is not blind final acceptance, exact equal-recall timing or a new default; long changed-vector CRUD and Full comparisons remain pending.


[RaBitQ architecture, snapshot layout and remaining acceptance](lite_rabitq_design.md).

## Mutation failure and RaBitQ distance computation

Published FP32 and FP16 graphs use a local mutation journal. Allocation failure
in Add or Update preserves logical records and adjacency; Remove returns false
and preserves contents if its graph repair cannot allocate. Reserved capacity
may grow. Only affected rows and the overwritten records are backed up, rather
than copying the entire graph. Private BuildGraph construction is discarded on
failure and enables journaling before publishing the completed graph.

RaBitQ computes the query sum once per search or maintenance query. Its 5-bit
supplement calls Full's shared scalar and SIMD kernels with runtime
AVX2/AVX512 selection and a scalar fallback. SIMD uses the native fused multiply-add
and vector reduction, so distance rounding can differ from the former scalar
dimension-order accumulation. Tests cover independent scalar estimates, unaligned
inputs, plane tails, query quality and mutation rollback. Encoding, persistence
and search budgets are unchanged. A speed improvement over the previous Lite implementation does not
establish parity with Full VSAG or with FP32/FP16.

## Expanded native-reference acceptance and ELF scope

The current Lite/native HGraph 600-query-per-dataset study is recorded in `lite/benchmark/results/acceptance-full-expanded-r1-20261009`. It uses a pinned installed Full reference, not old Lite, with explicit source/header/binary provenance. Query budgets differ (Lite512/Full128); this is not an exact matched-recall or same-budget comparison. The study records warm-load/total-RSS advantages but slower queries, larger snapshots and higher construction peaks. These observed queries are regression data, not blind holdout or long changed-vector CRUD acceptance.

`python3 lite/benchmark/measure_elf_closure.py --lite LITE_SO --full FULL_SO --output NEW_OUTPUT --scratch /dev/shm` measures trusted shared-library ELF DT_NEEDED closures. It resolves recursive dependencies, counts each realpath once (including common system runtime), and applies identical `strip --strip-unneeded` only to scratch copies; originals are not changed. This scope excludes executables, headers, Python bindings, containers and dynamically loaded plugins. The recorded reduction is not a complete wheel/container/SDK size claim. Run `python3 lite/benchmark/test_measure_elf_closure.py` for dependency/deduplication/copy-only fixtures.

Query-local graph heaps preserve the existing admission and traversal order.
After unfiltered FP32/FP16 traversal ends, Lite takes the owned retained buffer,
selects the best k candidates, and sorts only that prefix by distance then
external ID. RaBitQ scores the same retained candidates directly instead of
popping the entire coarse heap; its scoring formula and final ID tie-break are
unchanged. Filtered search keeps its existing path. This follows the native
DistanceHeap retained-buffer access pattern, without changing ef, topology,
the public API, or snapshots. The paired regression evidence is in
`lite/benchmark/results/query-selection-expanded-20261009`; its control is
old Lite, not Full VSAG.

For persistent whole-row CRUD replay, set `VSAG_GRAPH_REPLACEMENTS` to fvecs in operation order and supply QUERY_RESULTS. Every replacement must be finite and differ from the current row; the row count must equal ROUNDS times CRUD_OPS. Final truth comes from `changed-groundtruth.ivecs`, while initial truth stays in `groundtruth.ivecs`. Remove/re-add retain the new row instead of restoring original values. The tool writes `.initial` query evidence and `.operations.csv` (operation 0/1/2 means Update/Remove/Add); legacy CLI and CSV columns are unchanged without this opt-in. `lite/benchmark/prepare_persistent_crud.py` freezes three full-ID passes and computes direct-difference FP64 truth. `persistent-whole-crud-20261009` records lower maintained quality than fresh construction on the same final data, especially RaBitQ. Shorter queries are not a speed win when quality drops. Fresh RaBitQ also retrains the model, so that comparison does not isolate graph maintenance. These are serialized initial/final queries, warm cache, and one run per case, not per-operation interleaved reads or100k/final acceptance.

RaBitQ removal still scans all adjacency rows without reverse edges. The scan now
checks for either the removed slot or the relocated last slot once, skipping
unchanged rows before erase/remap. Affected-row ordering, transactional backups,
repair candidates and persisted results are unchanged; this is not an
asymptotic-complexity improvement or a maintenance-quality repair. Fixed replay
evidence and the legacy-scan golden path are recorded in
`lite/benchmark/results/rabitq-remove-skip-20261009`. Its control is old Lite,
not native Full; post-CRUD recall failures remain explicit.

For persistent replay only, VSAG_GRAPH_CRUD_MODE=update runs Update without deletion, replace runs Remove/Add without the redundant preceding Update, and all keeps the combined protocol. Explicit selection appends mutation_mode to the opt-in CSV; default schema is unchanged. The same final data/truth isolation is recorded in lite/benchmark/results/persistent-crud-isolation-20261010. Both isolated RaBitQ paths still fail the study recall floors; this separates workloads but does not establish or repair a sole cause.

VSAG_GRAPH_DIAGNOSTIC_EF can request the full node count during persistent
benchmark replay, emitting separate initial/final diagnostic evidence without
changing formal ef. The frozen-model ceiling study in
lite/benchmark/results/rabitq-model-ceiling-20261010 reaches about99.55--99.65%
after this workload, while formal maintenance quality remains lower. An isolated
Update incoming-protection candidate improves recall but adds about33% mutation
cost and fails the GIST study floor; it is not enabled in production. See the
adjacent candidate results. Diagnostic full-budget recall is not formal acceptance.

## Experimental runtime incoming-count maintenance

RaBitQ now contains an internal NONE/RECOUNT/CACHED incoming-protection policy; the public adapter remains NONE. CACHED lazily maintains one uint64_t incoming count per slot through adjacency changes instead of recounting the whole graph on every Update. Successful Add appends a zero count and applies edge deltas; Remove subtracts deleted outgoing edges and transfers the last slot count during compaction. Failed journal mutations invalidate it; the next Update rebuilds. Reverse-adjacency copy rollback preserves the original cache. Runtime counts consume 8N logical bytes, are included in internal memory accounting, and do not change snapshots/API/model/budgets. Disabling retains capacity. Continuous Update replay at10k reduced the protected prototype's mutation block by15.79% on GIST and18.50% on Cohere, with byte-identical paired snapshots and ordered results. GIST Recall .897667 still fails the study target, so the policy is not default-enabled; mixed replay still fails maintenance-quality targets;100k and native Full benefits are unproven. See lite/benchmark/results/rabitq-incoming-count-cache-20261010 for raw data, golden tests, scoped coverage and limitations.

The structural mixed replay in lite/benchmark/results/rabitq-mixed-count-cache-20261010 preserves the same protected-policy snapshots and ordered results after three changed Update/Remove/Add passes. Relative to the previous cache that invalidated on each successful Add/Remove, mutation blocks decrease10.05%/11.61% on GIST/Cohere10k. Final recalls .811667/.908000 still fail study targets: this is bookkeeping cost reduction, not a maintenance-quality fix, a public-default speedup or native Full comparison. Operation timings are not uniformly improved and the evidence is single-pair serialized replay.

Internal ConfigureIncomingProtection also accepts optional Add/Remove stage flags, both false by default. Add can protect a new slot after link; Remove can protect sorted affected orphan targets after compaction/repair, with the same degree-safe local guard and mutation journal. Public adapter remains NONE. The fixed factorial study in lite/benchmark/results/rabitq-structural-guard-20261010 raises mixed GIST/Cohere Recall .811667/.908000 to .847500/.922000 when both flags are on, at mutation-block costs +6.39%/+3.30% and higher query latency. Add-only accounts for most observed gains. These recalls still fail study targets; the internal flags are not public configuration/default acceptance, and this borrows the existing Lite floating GraphBackend, not native Full HGraph.

The further internal Add displaced-target guard records old targets evicted by reverse-link pruning and repairs their incoming edges inside the existing transaction. Fixed-budget10k mixed changed-vector CRUD improves GIST Recall .847500 to .855167 and Cohere .922000 to .924500,with CRUD block cost +2.86%/+1.52%; one-pair query point estimates improve but are not a repeatable latency claim. Both still miss study floors,so the public policy remains NONE. The regression demonstrates the old-target omission,not a global connectivity guarantee. Evidence: `lite/benchmark/results/rabitq-add-displaced-20261010`.

The bounded RaBitQ link-ranking stack-scratch trial was rejected: three fixed-budget real changed-vector CRUD pairs per dataset showed no stable benefit (median mutation block +1.77% GIST/+1.07% Cohere). Complete snapshots and ordered results were identical. Production was restored; only the degree64 saturation/tied-code regression and negative evidence remain. See `lite/benchmark/results/rabitq-link-scratch-20261010`; scoped restored Lite coverage is95.16%,not a new performance gain.

## RaBitQ load-only ID validation optimization (2026-10-11)

The [matched loader study](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-load-id-map-20261011/README.md) removes a temporary duplicate-ID set;the required constructor slot map still rejects duplicates. Default reverse=false/NONE and snapshot bytes are unchanged. Fifteen alternating10k pairs per dataset show GIST/Cohere warm-load medians6.483486->6.243492ms (-3.70%) and 5.721933->5.503448ms (-3.82%),with baseline-subtracted RSS472/416KiB lower and60 exact native round trips. This is allocator-retention/warm-load evidence,not index capacity,query,CRUD,cold I/O,Full or100k improvement. Malformed duplicates are rejected later after graph parsing;INVALID_BINARY remains but validation order/error text is not preserved. Quality and final acceptance remain pending.

## Experimental RaBitQ incoming-safe link pruning (2026-10-11)

The [fixed-budget paired study](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-safe-link-20261011/README.md) adapts floating Lite's incoming-safe edge removal to RaBitQ cached counts. An existing target's last incoming edge is preserved when another ranked edge can be safely dropped; if none exists, the original farthest fallback remains. This is not general diversity pruning or complete connectivity protection. The internal final `protect_link` argument requires CACHED and defaults false; the public adapter remains NONE/reverse=false, with unchanged API, model, degree/ef and snapshot format.

Both measured adapters use CACHED, with only the candidate enabling this gate. Three alternating10k pairs each use GIST/Cohere600 observed queries and three full-ID true whole-vector Update/Remove/Add passes (1.08M calls total). Final recall improves .811667 to .899167 / .908 to .947 (+8.75/+3.90 percentage points), while CRUD median rises1.20/2.29%. Post-CRUD query P50 also rises407.651 to439.810 /403.991 to428.569us; this is a quality/cost tradeoff, not an equal-recall speed win. Both pre-existing study floors .90/.95 are still missed. These deterministic observed cohorts are not blind or independent quality replications.

Keep this internal strategy opt-in pending further maintenance-quality work and100k/interleaved acceptance. Release6,ASan/UBSan5,fresh coverage5,RaBitQ-OFF4,format/tidy15 and30000 allocation-fault positions pass; scoped Lite coverage is95.44%,not whole Full. Source/dependency/adapter receipts and raw query/operation evidence are retained. No public-default, Full, cold I/O or package improvement is claimed.

## RaBitQ owned build adjacency handoff (2026-10-11)

The [paired build diagnostic](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-owned-topology-20261011/README.md) removes row adjacency → CSR → row adjacency copies. Build shrinks spare row capacity individually then transfers ownership; loading keeps the CSR constructor. API, edges, quantization, query budgets, incoming defaults and format remain unchanged. Reverse adjacency checks targets before count-array indexing.

Synthetic dim17, three alternating pairs each10k/100k: median100k build HWM87628→65460KiB (-25.30%);10k both18856KiB. Build-time medians increase0.47%/1.60%, not a stable speed win. Cumulative ordinary-new requested bytes decrease4760184/38653816, not retained memory. Six paired snapshots and ordered eight-query IDs/distances match. Not real high-dimensional data, long CRUD, cold loading, recall acceptance or Full comparison.10k HWM can be obscured by launcher floor; real-data and fresh Full validation remain pending.

## Real100k RaBitQ ownership acceptance slice (2026-10-11)

The [real-data study](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-owned-real100k-20261011/README.md) compares production4f73870 against0444bf9, BOTH Lite public NONE/reverse=false, not native Full. GIST960 and normalized Cohere768 each100k/100 observed queries/k10; degree16/build128/query512; three alternating pairs,12 builders/36 fresh loaders.

Whole-process peak medians fall928420→903892KiB (-2.64%) and759556→734956 (-3.24%); this includes inputs/construction/query/native reload, not a separate build-stage peak. Build medians are essentially unchanged. Six paired native snapshots/ordered-query outputs match exactly;Recall.859/.937 stays unchanged and insufficient. Snapshot sizes112804416/93603552bytes unchanged. Query P50 medians increase1088.753→1115.092us (+2.42%) and1068.973→1134.902 (+6.17%);Save47.454→65.928ms (+38.93%) and38.101→60.065 (+57.65%) regresses. Fresh warm load62.696→60.714/53.161→53.228ms is not a cold or universal speed result.

This validates a modest real memory tradeoff, NOT overall speed acceptance, Full advantage or durable changed-value CRUD.100 observed queries make exploratory tails only. Row allocation/cache locality and Save's GetGraph CSR materialization are follow-up hypotheses, not measured causes. Release6 and the initial driver fixture passed; prior4f73870 sanitizer/OFF/coverage evidence is inherited, not rerun. Final collection after RAM control expiry restored the first successful fixture log from the archive and retains the failed replay log. Fresh Full/config-SIMD-quality alignment, real long CRUD and broader query cohorts remain open.

## RaBitQ bounded streamed Save (2026-10-11)

The [streamed Save study](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-stream-save-20261011/README.md) replaces whole-graph GetGraph CSR materialization with read-only row traversal in original offsets/edges order using at most8192 uint64 entries (64KiB). Validate,scalar little-endian fallback,short-write errors and wire format remain;no public API/topology/model/budget/default changes.64KiB is adjacency staging only,not all Save scratch in reverse/cached modes.

Real GIST960/Cohere768 each100k,parentb09631c versus candidate,seven alternating fresh-process pairs/dataset,28 saves. Load precedes timing of ofstream open/Save/close;CPU0/threads1/RAM warm uncontrolled cache,not fdatasync durability or cold I/O. Save medians52.819 to45.774ms (-13.34%) and44.151 to37.441 (-15.20%). After-Save VmHWM151216 to134964/132460 to116216KiB,16252/16244KiB less;load/launcher HWM can obscure Save-only peaks. All28 output hashes equal input;input hashes match previous real100k study.

Release6/SAN5/freshCov5/OFF4/format-tidy15 pass;scoped production Lite2983/3125=95.456%,not wholeFull. Tests span0/1/8191/8192/8193/9000rows,block boundaries,bulk/scalar endian and short writes. Initial naming lint failed,corrected before timing;log retained. This is loaded-state Save optimization,NOT same-protocol remeasurement of prior build-then-Save regression,query regression,long CRUD quality or Full acceptance;those remain open.

## Build-row capacity candidate not adopted (2026-10-11)

The [isolated study](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-row-capacity-20261011/README.md) removes only the per-row shrink_to_fit loop in a candidate adapter. Both variants share bf50173 headers/objects/streamed Save; no production source or public option changes. GIST960/Cohere768 each100k/100 observed queries,degree16/build128/query512/CPU0/threads1,three alternating pairs,12 builders/36 warm fresh loaders. Six paired snapshots and ordered query outputs match;Recall .859/.937 unchanged.

Median query P50 GIST1081.373 to1084.913us (+0.33%),Cohere1100.393 to1146.963 (+4.23%);P99+3.46%/+0.94%. Build+0.60%/+0.02%,whole-process peak+0.005%/0%,no clear win. Small Save/load changes are not stable algorithm gains. Keeping capacity does not establish query improvement or causality of the prior regression. Do not adopt;retain shrinking default and streamed Save. Raw parameters/source/inputs/latencies/GT/commands archived and independently verified. Fresh Release6 passed;SAN/OFF/coverage inherited from bf50173,NOT candidate sanitizer/fault acceptance. Not Full,long CRUD or blind tail validation. Next priority is route workspace/heap allocation diagnosis.

## Promoted implementation and historical evidence

The implementation promoted on 2026-10-12 is derived from personal source `a62e62e08db3f0d7da2e4439785feed6fcf90257`. Experimental diversity switches and diagnostic benchmark targets are excluded. Benchmark commands and `lite/benchmark/` paths below refer to that pinned experimental repository, not targets shipped in this standalone PR. The reports retain each measured source version: numbers from different stages are not a single latest-version acceptance matrix. See [the pinned reproducibility sources](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark) and [promotion evidence](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results). Current CPU-time/utilization, quality-matched native Full comparisons, cold I/O and hours-long interleaved changed-vector CRUD remain incomplete. Externally concurrent calls require caller serialization.
